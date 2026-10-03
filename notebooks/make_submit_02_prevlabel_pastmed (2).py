# =====================================================================
# 9/1 제출 v2: 8/25 라벨(전주 라벨) + 0.5 * rank(-past_med)
#   - 8/25 라벨: 정답 구간(8/25~9/12)이 9/1 정답 구간(9/1~9/19)과 11일 겹쳐서 강한 신호
#   - past_med: 라벨이 log(미래) - log(과거)라서 과거 중앙값이 낮을수록 Rising 되기 쉬움
# =====================================================================
import duckdb, pandas as pd, numpy as np          # duckdb: CSV를 SQL로 집계 / pandas: 표 처리
from scipy.stats import rankdata                  # 값 -> 순위 변환 (동점은 평균 순위)

RAW = '/mnt/project'                              # 데이터 폴더 (로컬에서는 '../data/raw/epoch_data' 로 바꾸기)
D, PREV = '2026-09-01', '2026-08-25'              # D: 예측 기준일 / PREV: 피처로 쓸 전주 라벨 기준일

con = duckdb.connect()                            # 메모리 위의 DuckDB 연결 생성
# video_5d_views.csv 를 'v'라는 이름의 뷰로 등록 (파일을 통째로 메모리에 올리지 않고 SQL로 조회)
con.sql(f"create view v as select * from read_csv('{RAW}/video_5d_views.csv', header=true, auto_detect=true)")

# 채점 대상 행 목록 불러오기 (utf-8-sig: 파일 앞의 BOM 문자 제거)
ss = pd.read_csv(f'{RAW}/sample_submission.csv', encoding='utf-8-sig')
# row_id = '채널ID_날짜' 형태 -> 마지막 '_' 기준으로 잘라 채널ID만 추출
ss['channel_id'] = ss.row_id.str.rsplit('_', n=1).str[0]

# past_med: 기준일 이전 롱폼(is_shorts=0) 영상들의 view_5d 중앙값을 채널별로 계산
#   - 라벨 정의와 똑같이 직전 영상까지 전부 포함 (컷 없음)
#   - ln(1+x): 조회수 스케일을 로그로 줄임 (라벨도 로그 차이로 정의됨)
past = con.sql(f"""select channel_id, ln(1+median(view_5d)) lp from v
                   where is_shorts=0 and published_at < timestamp '{D}' group by 1""").df()

tl = pd.read_csv(f'{RAW}/train_labels.csv')      # 학습용 정답 (8/11, 8/25 기준일)
tl['channel_id'] = tl.row_id.str.rsplit('_', n=1).str[0]   # row_id 에서 채널ID 추출
# 8/25 기준일 행만 골라 target 컬럼 이름을 prev(전주 라벨)로 변경
prev = tl[tl.row_id.str.endswith(PREV)][['channel_id', 'target']].rename(columns={'target': 'prev'})

# 채점 대상 694행에 past_med, 전주 라벨을 채널ID 기준으로 붙이기 (how='left': 채점 행은 하나도 빠지지 않게)
m = ss[['row_id', 'channel_id']].merge(past, on='channel_id', how='left').merge(prev, on='channel_id', how='left')

lp = m.lp.fillna(m.lp.median())                  # past_med 가 없는 채널은 전체 중앙값으로 채움
r_lp = rankdata(-lp) / len(m)                     # 과거 중앙값이 낮을수록 1에 가까운 순위 점수 (0~1)
score = m.prev.fillna(0.2) + 0.5 * r_lp           # 전주 라벨(0/1) + 0.5*순위 / 8/25 라벨 없는 채널은 기저율 0.2
m['prediction'] = score / 1.5                     # 최댓값 1.5로 나눠 0~1 범위로 맞춤 (순위는 그대로라 PR-AUC 영향 없음)

# 점검용 출력: 전체 행 수, 전주 라벨이 없는 채널 수, past_med 가 없는 채널 수
print('rows', len(m), '| 8/25 라벨 없음', m.prev.isna().sum(), '| past_med 없음', m.lp.isna().sum())
# 제출 형식(row_id, prediction)으로 저장, 순서는 sample_submission 그대로
m[['row_id', 'prediction']].to_csv('submit_02_prevlabel_pastmed.csv', index=False)
