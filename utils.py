"""
유틸리티 함수 모듈

학습 및 평가에 필요한 보조 함수들을 제공합니다.
"""

import numpy as np
import math
import torch
import random
import os
import logging
import time


class AverageMeter:
    """
    실행 평균을 계산하는 클래스.
    
    학습 중 loss, accuracy 등의 메트릭을 추적하는 데 사용됩니다.
    """
    
    def __init__(self):
        """AverageMeter 초기화"""
        self.val = 0      # 현재 값
        self.avg = 0      # 평균 값
        self.sum = 0      # 누적 합
        self.count = 0    # 누적 개수

    def update(self, val, n=1):
        """
        새로운 값을 추가하여 평균을 업데이트합니다.
        
        Args:
            val (float): 새로운 값
            n (int): 값의 개수 (배치 크기 등). Default: 1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        """모든 값을 초기화합니다."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def logger_configuration(config, save_log=False, test_mode=False):
    """
    로거를 설정합니다.
    
    Args:
        config: 설정 객체 (workdir, samples, models, log 속성 필요)
        save_log (bool): 로그를 파일에 저장할지 여부. Default: False
        test_mode (bool): 테스트 모드 여부 (workdir에 '_test' 추가). Default: False
    
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    logger = logging.getLogger("Deep joint source channel coder")
    
    # 테스트 모드인 경우 workdir 수정
    if test_mode:
        config.workdir += '_test'
    
    # 로그 저장 모드인 경우 디렉토리 생성
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    
    # 로그 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    
    # 표준 출력 핸들러 (콘솔 출력)
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    
    # 파일 핸들러 (로그 파일 저장)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    
    logger.setLevel(logging.INFO)
    config.logger = logger
    
    return config.logger


def makedirs(directory):
    """
    디렉토리를 생성합니다 (존재하지 않는 경우에만).
    
    Args:
        directory (str): 생성할 디렉토리 경로
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    """
    모델의 state_dict를 저장합니다.
    
    Args:
        model (nn.Module): 저장할 모델
        save_path (str): 저장 경로
    """
    torch.save(model.state_dict(), save_path)


def seed_torch(seed=1029):
    """
    재현성을 위해 모든 랜덤 시드를 고정합니다.
    
    Python, NumPy, PyTorch (CPU 및 GPU)의 시드를 설정합니다.
    
    Args:
        seed (int): 시드 값. Default: 1029
    """
    # Python 내장 random
    random.seed(seed)
    
    # Python hash 시드
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy 시드
    np.random.seed(seed)
    
    # PyTorch CPU 시드
    torch.manual_seed(seed)
    
    # PyTorch GPU 시드
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경
    
    # CuDNN 비결정적 동작 비활성화 (재현성 보장)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
