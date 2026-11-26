"""데이터 불러오기와 전처리. 02_preprocessing에서 확인하고, 03_modeling에서 import해서 사용한다."""
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils import DATA_DIR

IMG_SIZE = 224                              # 교안 기본 입력 크기
IMAGENET_MEAN = [0.485, 0.456, 0.406]       # 사전학습 가중치에 맞는 정규화 값
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_COLOR = (255, 255, 255)                 # 배경이 대부분 흰색이라 흰색으로 패딩


class PadToSquare:
    """비율을 유지한 채 긴 변을 IMG_SIZE로 줄이고, 남는 부분을 흰색으로 채워 정사각형을 만든다."""
    def __init__(self, size=IMG_SIZE, color=PAD_COLOR):
        self.size = size
        self.color = color

    def __call__(self, img):
        return ImageOps.pad(img, (self.size, self.size), color=self.color)


def get_transform(train=False, augment=False):
    """Train/Val/Test 공통: 패딩 → 텐서 → 정규화. 증강은 train=True, augment=True일 때만."""
    steps = [PadToSquare()]
    if train and augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class FashionDataset(Dataset):
    """metadata.csv의 경로와 target_id만 사용 (category, subcategory, product_id는 입력으로 쓰지 않음)."""
    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["target_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(DATA_DIR / self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def load_meta():
    return pd.read_csv(DATA_DIR / "metadata.csv", encoding="utf-8-sig")


def get_loaders(batch_size=32, augment=False, num_workers=0, seed=42):
    """제공된 분할(train/val/test)을 그대로 사용해 DataLoader 3개를 만든다."""
    meta = load_meta()
    loaders = {}
    for split in ["train", "val", "test"]:
        df = meta[meta["split"] == split].reset_index(drop=True)
        is_train = split == "train"
        ds = FashionDataset(df, get_transform(train=is_train, augment=augment))
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,                                          # Train만 섞기
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed) if is_train else None,
            persistent_workers=num_workers > 0,
        )
    return loaders