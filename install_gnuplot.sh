#!/bin/bash

# GNUPlot 설치 스크립트

echo "GNUPlot 설치를 시도합니다..."

# Conda 환경 확인
if command -v conda &> /dev/null; then
    echo "Conda를 사용하여 설치합니다..."
    conda install -y -c conda-forge gnuplot
    if [ $? -eq 0 ]; then
        echo "GNUPlot 설치 완료!"
        gnuplot --version
        exit 0
    fi
fi

# 시스템 패키지 매니저로 설치 시도
if command -v apt-get &> /dev/null; then
    echo "apt-get을 사용하여 설치합니다..."
    sudo apt-get update && sudo apt-get install -y gnuplot
elif command -v yum &> /dev/null; then
    echo "yum을 사용하여 설치합니다..."
    sudo yum install -y gnuplot
else
    echo "패키지 매니저를 찾을 수 없습니다."
    echo "수동으로 설치해주세요:"
    echo "  - Ubuntu/Debian: sudo apt-get install gnuplot"
    echo "  - CentOS/RHEL: sudo yum install gnuplot"
    echo "  - Conda: conda install -c conda-forge gnuplot"
    exit 1
fi

