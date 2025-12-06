from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image
import os
import torch
import numpy as np
import requests
import zipfile
from tqdm import tqdm

# Number of workers for data loading
NUM_DATASET_WORKERS = 8

# --- [Download Helper Functions] (기존과 동일) ---
def download_file(url, save_path):
    print(f"[*] Downloading {url} to {save_path}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    response = requests.get(url, stream=True, headers=headers)

    if response.status_code != 200:
        print(f"[!] Failed to download. Status Code: {response.status_code}")
        return 

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def unzip_file(zip_path, extract_to):
    print(f"[*] Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def check_and_download_div2k(root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    expected_path = os.path.join(root_dir, 'DIV2K_train_HR')
    if not os.path.exists(expected_path) or len(glob(os.path.join(expected_path, '*.png'))) < 800:
        print("[!] DIV2K Train dataset not found. Downloading...")
        url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        zip_path = os.path.join(root_dir, "DIV2K_train_HR.zip")
        
        if not os.path.exists(zip_path):
            download_file(url, zip_path)
            
        unzip_file(zip_path, root_dir)
    else:
        print(f"[*] DIV2K Train dataset found at {expected_path}")

def check_and_download_kodak(root_dir):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    existing_imgs = glob(os.path.join(root_dir, '*.png'))
    if len(existing_imgs) < 24:
        print("[!] Kodak dataset not found or incomplete. Downloading...")
        base_url = "http://r0k.us/graphics/kodak/kodim{:02d}.png"
        for i in range(1, 25):
            url = base_url.format(i)
            save_path = os.path.join(root_dir, f"kodim{i:02d}.png")
            if not os.path.exists(save_path):
                download_file(url, save_path)
    else:
        print(f"[*] Kodak dataset found at {root_dir}")

# --- [Dataset Classes] ---

class HR_image(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.imgs = []
        self.is_train = is_train

        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        for d in data_dir:
            self.imgs += glob(os.path.join(d, '**', '*.jpg'), recursive=True)
            self.imgs += glob(os.path.join(d, '**', '*.png'), recursive=True)
        
        self.imgs = sorted(list(set(self.imgs)))
        
        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {data_dir}. Check the download logic!")
        
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.is_train:
            # 학습 시: 설정된 transform (RandomCrop 256 등) 적용
            if self.transform:
                img = self.transform(img)
        else:
            # 테스트 시: 128의 배수 크기로 Center Crop (OLD 코드 로직 적용)
            w, h = img.size
            new_w = w - (w % 128)
            new_h = h - (h % 128)
            
            # 128 배수가 되도록 자르기 (Center Crop)
            # 만약 원본이 이미 128 배수면 그대로 유지됨 (예: Kodak 768x512)
            transform_test_dynamic = transforms.Compose([
                transforms.CenterCrop((new_h, new_w)), # (Height, Width) 순서 주의
                transforms.ToTensor()
            ])
            img = transform_test_dynamic(img)

        return img

    def __len__(self):
        return len(self.imgs)
    
def get_loader(args, config):
    # 1. Download Check
    if args.trainset == 'DIV2K':
        check_and_download_div2k(config.train_data_dir)
    
    if args.testset == 'Kodak':
        check_and_download_kodak(config.test_data_dir)
    elif args.testset == 'DIV2K':
        pass 

    # 2. Define Transforms (Train Only)
    # Test용 Transform은 __getitem__ 내부에서 동적으로 처리함
    transform_train = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor() 
    ])

    # 3. Dataset Init
    if args.trainset == 'DIV2K':
        # Train: is_train=True
        train_dataset = HR_image(config.train_data_dir, transform=transform_train, is_train=True)
    else:
        raise NotImplementedError(f"Trainset {args.trainset} not implemented.")

    if args.testset in ['Kodak', 'DIV2K']:
        # Test: is_train=False (Transform은 내부 로직 따름)
        test_dataset = HR_image(config.test_data_dir, transform=None, is_train=False)
    else:
         raise NotImplementedError(f"Testset {args.testset} not implemented.")

    def worker_init_fn_seed(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)

    # 4. DataLoader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn_seed,
                              drop_last=True)

    # [중요] Test Loader는 무조건 batch_size=1 이어야 함 (이미지 크기가 다를 수 있으므로)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1, 
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader