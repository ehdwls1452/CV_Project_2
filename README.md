# 상품 이미지 카테고리 분류 (원피스 / 재킷 / 바지 / 스커트 / 상의)

ResNet18 기반 Scratch 학습과 Transfer Learning 전략을 비교하고, 오류 분석을 통해 개선 실험을 수행합니다.

## 실행 방법
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
데이터는 저장소에 포함하지 않습니다. 수업에서 배포한 데이터를 `data/` 아래에 풀어 주세요.
```
data/
├── metadata.csv
├── class_mapping.csv
├── class_counts.csv
└── images/{train,val,test}/
```

## 폴더 구조
```
notebooks/  01_EDA → 02_preprocessing → 03_modeling
src/        공통 설정(utils.py)
figures/    README용 그래프
results/    실험 결과 기록
```

> 진행하면서 채워 나갈 예정입니다.
