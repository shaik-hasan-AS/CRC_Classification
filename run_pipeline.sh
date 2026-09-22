#!/bin/bash
set -e

echo "Starting full 3-seed KD validation pipeline..."
cd data
if [ ! -d "NCT-CRC-HE-100K" ]; then
    echo "Downloading NCT-CRC-HE-100K..."
    wget -c "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip" -O NCT-CRC-HE-100K.zip
    echo "Extracting..."
    unzip -q NCT-CRC-HE-100K.zip
    rm NCT-CRC-HE-100K.zip
else
    echo "Dataset already exists."
fi
cd ..

echo "Running training for seeds 43 and 44..."
.venv/bin/python scripts/run_kd_seeds_43_44.py
echo "Done."
