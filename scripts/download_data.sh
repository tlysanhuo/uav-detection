#!/bin/bash
# Download and extract VisDrone-DET dataset
set -e

DATA_DIR="/workspace/tly/uav-detection/data/VisDrone"
mkdir -p "$DATA_DIR"

# VisDrone2019-DET train/val/test-dev from GitHub releases
BASE_URL="https://github.com/VisDrone/VisDrone-Dataset/releases/download"

download_and_extract() {
    local name=$1
    local url="${BASE_URL}/${name}"
    local zipfile="${DATA_DIR}/${name}"
    local dest="${DATA_DIR}"

    if [ -d "${dest}/${name%.zip}" ]; then
        echo "${name%.zip} already exists, skipping"
        return
    fi

    echo "Downloading ${name}..."
    wget -q --show-progress -O "$zipfile" "$url"
    echo "Extracting ${name}..."
    unzip -q -o "$zipfile" -d "$dest"
    rm "$zipfile"
    echo "${name} done"
}

# Train set (~1.4GB)
download_and_extract " VisDrone2019-DET-train.zip" "VisDrone2019-DET-train.zip"

# Val set (~0.15GB)
download_and_extract "visdrone2019-DET-val.zip" "VisDrone2019-DET-val.zip"

# Test-dev set (~0.5GB)
download_and_extract "VisDrone2019-DET-test-dev.zip" "VisDrone2019-DET-test-dev.zip"

echo ""
echo "=== Download complete ==="
echo "Converting annotations to YOLO format..."

source /workspace/tly/miniconda3/bin/activate uav_detect
python /workspace/tly/uav-detection/scripts/convert_visdrone.py --src "$DATA_DIR"

echo "=== All done ==="
