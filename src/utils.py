"""모든 노트북이 공통으로 쓰는 설정 (경로, 시드, 디바이스, 그림 저장)."""
import os
import random
import platform
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------- 경로 ----------
def find_root(marker: str = "src") -> Path:
    """현재 위치에서 위로 올라가며 프로젝트 루트(src 폴더가 있는 곳)를 찾는다.
    노트북이 notebooks/ 안에 있어도, 팀원 컴퓨터 경로가 달라도 동작한다."""
    p = Path.cwd().resolve()
    for cand in [p, *p.parents]:
        if (cand / marker).is_dir():
            return cand
    raise FileNotFoundError("프로젝트 루트를 찾지 못했습니다.")


ROOT = find_root()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["원피스", "재킷", "바지", "스커트", "상의"]   # target_id 0~4 순서


# ---------- 재현성 ----------
def set_seed(seed: int = 42) -> None:
    """같은 시드면 같은 결과가 나오도록 고정 (전략 간 비교를 공정하게)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------- 디바이스 ----------
def get_device():
    """CUDA(Colab GPU) > MPS(Apple Silicon Mac) > CPU 순으로 선택."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- 그래프 ----------
def setup_korean_font() -> None:
    """한글 클래스명이 네모(□)로 깨지지 않게 운영체제별 한글 폰트 지정."""
    system = platform.system()
    if system == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    elif system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "NanumGothic"   # Colab은 별도 설치 필요
    plt.rcParams["axes.unicode_minus"] = False        # 음수 기호 깨짐 방지


def save_fig(name: str, dpi: int = 150) -> Path:
    """현재 그림을 figures/에 저장. README에 바로 쓸 수 있게 처음부터 파일로 남긴다."""
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"저장됨: {path.relative_to(ROOT)}")
    return path
