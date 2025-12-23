#!/bin/bash

# Script to extract data and create GNUPlot visualization

echo "Step 1: Extracting chunk-by-chunk PSNR data from log files..."
python3 extract_chunk_psnr.py

if [ $? -ne 0 ]; then
    echo "Error: Data extraction failed."
    exit 1
fi

echo ""
echo "Step 2: Checking for GNUPlot..."

if ! command -v gnuplot &> /dev/null; then
    echo "GNUPlot이 설치되어 있지 않습니다."
    echo ""
    echo "설치 방법:"
    echo "  1. Conda 사용: conda install -c conda-forge gnuplot"
    echo "  2. 또는 설치 스크립트 실행: bash install_gnuplot.sh"
    echo ""
    read -p "지금 설치하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash install_gnuplot.sh
        if [ $? -ne 0 ]; then
            echo "설치 실패. 수동으로 설치해주세요."
            exit 1
        fi
    else
        echo "GNUPlot 설치 후 다시 실행해주세요."
        exit 1
    fi
fi

echo "GNUPlot 버전:"
gnuplot --version

echo ""
echo "Step 3: Creating GNUPlot visualization..."
gnuplot plot_chunk_psnr.gp

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! Plot saved as: chunk_psnr_comparison.png"
    echo "Data files are in: ./plot_data/"
else
    echo "Error: GNUPlot failed."
    exit 1
fi

