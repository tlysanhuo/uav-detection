"""
Custom modules for YOLOv8 improvement.
Programmatic approach: build standard model, then replace backbone C2f -> C2f_CBAM.

Usage:
    import custom_modules
    model = custom_modules.build_cbam_model('yolov8n.pt')
    model = custom_modules.build_cbam_p2_model('yolov8n-p2.yaml', 'yolov8n.pt')
"""
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, C2f, Bottleneck
from ultralytics import YOLO


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.sa(x * self.ca(x))


class C2f_CBAM(nn.Module):
    """C2f + CBAM at output."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )
        self.cbam = CBAM(c2, reduction)
        # Attributes required by ultralytics _predict_once
        self.f = -1
        self.i = 0

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cbam(self.cv2(torch.cat(y, 1)))


def c2f_to_cbam(c2f_mod, reduction=16):
    """Convert C2f -> C2f_CBAM, copying weights and metadata."""
    c1 = c2f_mod.cv1.conv.in_channels
    c2 = c2f_mod.cv2.conv.out_channels
    n = len(c2f_mod.m)
    shortcut = c2f_mod.m[0].add if n > 0 else False
    g = c2f_mod.m[0].cv2.conv.groups if n > 0 else 1

    new_mod = C2f_CBAM(c1, c2, n, shortcut, g, reduction=reduction)

    # Copy weights
    old_sd = c2f_mod.state_dict()
    new_sd = new_mod.state_dict()
    for k in new_sd:
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
    new_mod.load_state_dict(new_sd)

    # Copy ultralytics metadata
    for attr in ('f', 'i', 'type'):
        if hasattr(c2f_mod, attr):
            setattr(new_mod, attr, getattr(c2f_mod, attr))

    return new_mod


def _get_backbone_c2f_indices(model):
    """Find backbone C2f layer indices by checking the model yaml."""
    yaml = model.model.yaml
    backbone = yaml.get('backbone', [])
    indices = []
    for i, layer_def in enumerate(backbone):
        if len(layer_def) >= 3 and layer_def[2] == 'C2f':
            indices.append(i)
    return indices


def build_cbam_model(source='yolov8n.pt', backbone_indices=None, reduction=16):
    """Build YOLOv8 with CBAM on backbone C2f blocks."""
    model = YOLO(source)
    nn_model = model.model
    if backbone_indices is None:
        backbone_indices = _get_backbone_c2f_indices(model)
    for idx in backbone_indices:
        if isinstance(nn_model.model[idx], C2f):
            nn_model.model[idx] = c2f_to_cbam(nn_model.model[idx], reduction)
    return model


def build_cbam_p2_model(p2_yaml='yolov8n-p2.yaml', pretrained='yolov8n.pt', reduction=16):
    """Build YOLOv8n-P2 with CBAM on backbone C2f blocks.
    
    Strategy: Build P2 model from YAML (random init), then load pretrained
    YOLOv8n weights for matching layers. P2-specific layers use random init.
    """
    # Build P2 model from YAML
    model = YOLO(p2_yaml)
    nn_model = model.model

    # Load pretrained weights for transfer learning
    if pretrained:
        import torch
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        pretrained_model = ckpt.get('model', ckpt)
        if hasattr(pretrained_model, 'float'):
            pretrained_model = pretrained_model.float()
        pretrained_sd = pretrained_model.state_dict()
        
        # Load matching weights (backbone layers that exist in both models)
        current_sd = nn_model.state_dict()
        loaded, skipped = 0, 0
        for k in current_sd:
            if k in pretrained_sd and pretrained_sd[k].shape == current_sd[k].shape:
                current_sd[k] = pretrained_sd[k]
                loaded += 1
            else:
                skipped += 1
        nn_model.load_state_dict(current_sd)
        print(f"Loaded {loaded} pretrained params, skipped {skipped} (P2-specific or shape mismatch)")

    # Replace backbone C2f with C2f_CBAM
    backbone_indices = _get_backbone_c2f_indices(model)
    for idx in backbone_indices:
        if isinstance(nn_model.model[idx], C2f):
            nn_model.model[idx] = c2f_to_cbam(nn_model.model[idx], reduction)
            print(f"  Replaced layer {idx}: C2f -> C2f_CBAM")

    return model


if __name__ == '__main__':
    base = YOLO('yolov8n.pt')
    base_p = sum(p.numel() for p in base.model.parameters())

    cbam = build_cbam_model('yolov8n.pt')
    cbam_p = sum(p.numel() for p in cbam.model.parameters())

    print(f'YOLOv8n:  {base_p/1e6:.2f}M params')
    print(f'+CBAM:    {cbam_p/1e6:.2f}M params  (+{(cbam_p-base_p)/1e3:.1f}K)')

    # Forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = cbam.model(x)
    if isinstance(y, (list, tuple)):
        print(f"Forward OK, pred shape: {y[0].shape}, extra type: {type(y[1])}")
    elif isinstance(y, dict):
        print(f'Forward OK, output keys: {list(y.keys())}')
    else:
        print(f'Forward OK, output type: {type(y)}')

    # Test P2+CBAM model
    print()
    p2_model = build_cbam_p2_model('yolov8n-p2.yaml', 'yolov8n.pt')
    p2_p = sum(p.numel() for p in p2_model.model.parameters())
    print(f'YOLOv8n-P2+CBAM: {p2_p/1e6:.2f}M params')
    
    # Verify CBAM layers
    for i, m in enumerate(p2_model.model.model):
        if 'CBAM' in type(m).__name__:
            print(f'  Layer {i}: {type(m).__name__}')
