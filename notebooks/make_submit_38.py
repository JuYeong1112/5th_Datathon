# %% [markdown]
# # submit_38 (최종 제출, 리더보드 0.6707)
#
# **submit_38 = 0.5·rank(submit_21) + 0.5·rank(08-25 라벨 규칙)**
#
# - submit_21 : 0.35·s4 + 0.35·s3 + 0.30·submit_15 순위평균 (11_ensemble.ipynb 출력, 리더보드 0.6420)
# - 08-25 라벨 규칙 : 08-25 공식 라벨=1 채널을 위로, 같은 라벨 안에서는 9/1 과거 중앙값이 낮은 채널을 위로
#
# 입력: `submissions/submit_21_ens_s4_s3_s15.csv`, 원본 `video_5d_views.csv`, `train_labels.csv`, `sample_submission.csv`
# 출력: `submissions/submit_38_ens21_prevrule50.csv`

# %%
import os
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

RAW = os.environ.get('EPOCH_RAW', '../data/raw/epoch_data')
SUB = os.environ.get('EPOCH_SUB', '../submissions')
D, PREV = '2026-09-01', '2026-08-25'       # 본선 기준일 / 전주 공식 라벨 기준일

# %% 1. 입력 준비
# 9/1 과거 중앙값: 기준일 이전 롱폼 log(1 + view_5d)의 중앙값
lp = duckdb.sql(f"""
    SELECT channel_id, MEDIAN(LN(1 + view_5d)) AS lp
    FROM read_csv('{RAW}/video_5d_views.csv', header=true, auto_detect=true)
    WHERE is_shorts = 0 AND view_5d IS NOT NULL AND published_at < TIMESTAMP '{D}'
    GROUP BY 1""").df()

# 채점 대상 694행 (row_id = '채널ID_날짜')
sample = pd.read_csv(f'{RAW}/sample_submission.csv', encoding='utf-8-sig')[['row_id']]
sample['channel_id'] = sample.row_id.str.rsplit('_', n=1).str[0]

# 공식 08-25 라벨
tl = pd.read_csv(f'{RAW}/train_labels.csv')
tl['channel_id'] = tl.row_id.str.rsplit('_', n=1).str[0]
prev = tl[tl.row_id.str.endswith(PREV)][['channel_id', 'target']].rename(columns={'target': 'prev'})

m = sample.merge(lp, on='channel_id', how='left').merge(prev, on='channel_id', how='left')
m['s21'] = pd.read_csv(f'{SUB}/submit_21_ens_s4_s3_s15.csv').set_index('row_id').prediction.loc[m.row_id].values

# %% 2. 순위 계산
N = len(m)
def rank01(x):
    """값을 0~1 순위로 (결측은 중앙값으로 채움)"""
    x = pd.Series(np.asarray(x, float))
    return rankdata(x.fillna(x.median())) / N

prev_f = m.prev.fillna(0.2).values                  # 라벨 없는 채널은 기저율 0.2
r_rule = rank01(prev_f + 1e-3 * rank01(-m.lp))      # 라벨 우선, 같은 라벨 안에서는 과거 중앙값 낮은 순
r21 = rank01(m.s21)

# %% 3. 블렌드 · 저장
score = 0.5 * r21 + 0.5 * r_rule + 1e-6 * r21       # 동점은 submit_21 순위로 깸
score = (score - score.min()) / (score.max() - score.min())
sub = pd.DataFrame({'row_id': m.row_id.values, 'prediction': score})

# 제출 형식 검사
assert list(sub.columns) == ['row_id', 'prediction'] and len(sub) == 694
assert set(sub.row_id) == set(sample.row_id) and sub.row_id.is_unique
assert sub.prediction.notna().all() and sub.prediction.between(0, 1).all() and not sub.prediction.duplicated().any()

sub.to_csv(f'{SUB}/submit_38_ens21_prevrule50.csv', index=False)
print('저장 완료:', len(sub), '행')
