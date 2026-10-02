# EPOCH 5th Datathon : YouTube Rising Channel Prediction

데이터 사이언스 연합 동아리 EPOCH 제5회 데이터톤 참가 프로젝트입니다.

---

## 1. 문제 정의

어떤 YouTube 채널이 앞으로 성장 국면("Rising")에 들어갈지 예측한다.

- 예측 단위: `채널 × 기준일` 한 줄마다 Rising 확률 하나
- "Rising" = 현재 규모가 아니라, 자기 과거 대비 성장세가 강해진 채널 (구독자 500만 채널이 평소대로면 Rising 아님 / 구독자 2만 채널이 눈에 띄게 치고 나가면 Rising)

정답(라벨) 계산 방식

```
1단계) view_5d(영상) = 업로드 후 108~132시간 구간 스냅샷의 조회수 최댓값
2단계) past_score   = median(view_5d)  [기준일 이전 업로드된 롱폼 영상]
	future_score = median(view_5d)  [기준일 이후 업로드된 롱폼 영상, 없으면 0]
3단계) growth_score  = log1p(future_score) − log1p(past_score)
4단계) 같은 기준일 안에서 growth_score 상위 20% → label = 1
```

- 표본 조건: 기준일 이전 롱폼 영상(view_5d 있는 것) 3편 이상 & past_score ≥ 100
- 라벨 계산은 쇼츠 제외, 롱폼만 사용

---

## 2. 진행 방식

랜덤 분리 기반 일반적인 ML 파이프라인이 아니라, 시간 순서가 있는 데이터라는 점에 맞춰 아래처럼 진행합니다.

```
[1주차] 문제 정의 → 데이터 확인 → ★테스트 분리(날짜 기준) → EDA(train만)
[2주차] 전처리 → 파생변수 → 베이스라인 점수 기록 (+ 연습 리더보드 제출)
[3주차] 모델 추가 → 교차검증 → 비교표 → 최적 모델 선정
[4주차] 튜닝 → ★홀드아웃 1회 평가 → 노트북 통합 → 발표
```

핵심 원칙

1. 테스트 분리는 랜덤이 아니라 날짜로 한다. `train_labels.csv`에는 두 기준일(08-11, 08-25)이 섞여 있고 채널도 대부분 겹친다. 랜덤 분리 시 같은 채널 정보가 양쪽에 새서 검증 점수가 부풀려진다. → 08-11 = train / 08-25 = holdout(4주차 전까지 열람 금지)
2. 피처는 항상 "기준일 시점에 알 수 있던 정보"로만 만든다. `make_features(D)`처럼 기준일을 인자로 받는 함수로 작성해, 08-11 / 08-25 / 09-01(본선)에 동일하게 적용한다. 기준일 이후 정보가 섞이면 리더보드 성능과 무관하게 실격 사유가 될 수 있다.
3. 교차검증 대신 "08-11 학습 → 08-25 검증"을 주 지표로 쓴다. 기준일이 2개뿐이라 시간 기준 폴드가 사실상 1개. PR-AUC를 주 지표, ROC-AUC를 보조로 기록.
4. 리더보드 점수는 ±0.03 정도 흔들린다(표본 694행). 0.01 차이로 모델을 고르지 않는다. 판단은 항상 로컬 검증 기준으로.

---

## 3. 폴더 구조

```
datathon/
├── data/
│   ├── raw/
│   │   └── epoch_data/            ← 압축 해제한 원본 csv 17개 + MANIFEST.txt
│   │       (train_labels.csv, sample_submission.csv, videos.csv, ...)
│   └── processed/                 ← 직접 만든 파일 (parquet 변환본, 피처, split 등)
├── notebooks/
│   ├── 01_data_check.ipynb        ← 데이터 확인
│   ├── 02_split_eda.ipynb         ← 테스트 분리 + EDA
│   └── ...                        ← 02_features, 03_model, 04_tuning 등 예정
├── submissions/                   ← 제출 파일 (submit_NN_설명.csv)
└── README.md
```