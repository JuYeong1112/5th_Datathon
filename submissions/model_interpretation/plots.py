import os, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
S = os.environ['S']; O = f'{S}/interp'
plt.rcParams.update({'font.family': 'AppleGothic', 'axes.unicode_minus': False, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8C8577',
                     'axes.labelcolor': '#3A3A35', 'xtick.color': '#5A5850', 'ytick.color': '#2D2D28'})
INK, MUTED, GREEN, RED, GRAY = '#2D2D28', '#6B6A60', '#3D6B41', '#9A3F35', '#B8B3A7'
NAME = {'log_past_med': '과거 중앙값', 'c_lb': '08-25 라벨=1 기준선 높이', 'c_ub': '08-25 라벨=0 기준선 높이',
        'log_short_med_rel': '쇼츠 성과(롱폼 대비)', 'like_rate': '좋아요율', 'log_view_std': '조회수 변동성',
        'n_long_28': '최근 28일 롱폼 수', 'comment_rate': '댓글률', 'avg_dur': '평균 영상 길이', 'log_q90': '상위 10% 영상 수준',
        'mom_7': '최근 7일 모멘텀', 'prev_x_known': '라벨=1 × 아는 구간 부진', 'n_short_14': '최근 14일 쇼츠 수', 'prev_f': '08-25 라벨'}

# ---- 1. 그룹 Feature importance (백테스트 permutation) ----
gp = pd.read_csv(f'{O}/perm_group.csv', index_col=0).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
cols = [GREEN if g == gp['mean'].idxmax() else GRAY for g in gp.index]
ax.barh(gp.index, gp['mean'], color=cols, height=0.55, xerr=gp['std'], error_kw={'ecolor': '#8C8577', 'lw': 1, 'capsize': 2})
for y, (v, s) in enumerate(zip(gp['mean'], gp['std'])):
    ax.text(v + s + 0.006, y, f'{v:.3f}', va='center', fontsize=10, color=INK)
ax.set_xlabel('섞었을 때 PR-AUC 하락폭 (클수록 중요)', fontsize=10, color=MUTED)
ax.set_xlim(0, gp['mean'].max() + gp['std'].max() + 0.04); ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color='#E4E0D6', lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{O}/fig1_group_importance.png'); plt.close()

# ---- 2. SHAP 요약 (HGB, 9/1 테스트 694채널) ----
d = pickle.load(open(f'{O}/shap.pkl', 'rb')); F = d['F']; sv = np.asarray(d['shap_hgb']); X = d['X']
order = pd.Series(np.abs(sv).mean(0), index=F).sort_values(ascending=False).index[:8][::-1]
cmap = LinearSegmentedColormap.from_list('v', ['#E8E2D6', '#C98F7F', RED])
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=220)
rng = np.random.default_rng(0)
for y, f in enumerate(order):
    s = sv[:, F.index(f)]; x = X[f].values.astype(float)
    ok = ~np.isnan(x); c = np.full(len(x), np.nan)
    if ok.any():
        lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
        c[ok] = np.clip((x[ok] - lo) / (hi - lo + 1e-9), 0, 1)
    jit = rng.normal(0, 0.09, len(s))
    ax.scatter(s[~ok], y + jit[~ok], s=6, color='#CFCAC0', alpha=0.6, lw=0)
    ax.scatter(s[ok], y + jit[ok], s=7, c=c[ok], cmap=cmap, vmin=0, vmax=1, alpha=0.85, lw=0)
ax.axvline(0, color='#8C8577', lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME.get(f, f) for f in order]); ax.tick_params(axis='y', length=0)
ax.set_xlabel('SHAP 값 (← Rising 확률 낮춤 · 높임 →)', fontsize=10, color=MUTED)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)); cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cb.set_ticks([0, 1]); cb.set_ticklabels(['낮음', '높음']); cb.set_label('피처 값', fontsize=9, color=MUTED); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f'{O}/fig2_shap_summary.png'); plt.close()

# ---- 3. 개별 예측 설명: 08-25 라벨=0인데 상위 20%에 든 채널 (38 순위 121위) ----
m = pd.read_pickle(f'{O}/m38.pkl').set_index('row_id').loc[d['row_id']].reset_index()
m['rank38'] = (-m.r38).rank().astype(int)
i = m[(m.prev == 0) & (m.rank38 <= 139)].sort_values('rank38').index[0]
s = pd.Series(sv[i], index=F); top = s.reindex(s.abs().sort_values(ascending=False).index)[:6][::-1]
NAME.update({'log_prev_past': '08-25 시점 과거 중앙값'})
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
ax.barh([NAME.get(f, f) for f in top.index], top.values, color=[GREEN if v > 0 else RED for v in top.values], height=0.55)
for y, v in enumerate(top.values):
    ax.text(v + (0.02 if v > 0 else -0.02), y, f'{v:+.2f}', va='center', ha='left' if v > 0 else 'right', fontsize=10, color=INK)
ax.axvline(0, color='#8C8577', lw=0.8); ax.tick_params(axis='y', length=0)
ax.set_xlim(min(top.min() - 0.25, -0.4), top.max() + 0.25)
ax.set_xlabel('SHAP 값 (+ = Rising 쪽으로 밀어올림)', fontsize=10, color=MUTED)
fig.tight_layout(); fig.savefig(f'{O}/fig3_local_example.png'); plt.close()
print('rank', m.loc[i, 'rank38'], 'r21', int((-m.r21).rank()[i]), 'rule', int((-m.r_rule).rank()[i]))
