import os, sys, time, pickle
S = os.environ['S']
src = open(f'{S}/nb10_core.py').read().replace('from IPython.display import display', 'display = print')
exec(src)
import shap
from sklearn.inspection import permutation_importance
F = PAIR + CONS
t0 = time.time()

# ---- 1. 최종 구성 모델(s2 구조) 재현 확인 ----
s2_re, _ = predict(TR_dl, TE_defl, F)
sub_s2 = pd.read_csv(f'{SUB}/submit_10_s2_cons_defl.csv').set_index('row_id').prediction.loc[TE_defl.row_id].values
print('s2 재현 spearman', round(spearmanr(s2_re, sub_s2)[0], 4))

# ---- 2. Permutation importance (백테스트 5개 기준일, 앙상블 순위 점수의 PR-AUC 하락) ----
rng = np.random.default_rng(0)
rows = []
for td in EVAL_DATES:
    te = TR_dl[TR_dl.ref_date == td].reset_index(drop=True)
    tr = TR_dl[TR_dl.ref_date <= (pd.Timestamp(td) - pd.Timedelta(days=4)).strftime('%Y-%m-%d')]
    hs = [hgb(s).fit(tr[F], tr.target) for s in range(5)]
    lr = logreg().fit(tr[F], tr.target)
    def score(X):
        ph = np.mean([h.predict_proba(X)[:, 1] for h in hs], axis=0)
        pl = lr.predict_proba(X)[:, 1]
        return average_precision_score(te.target_raw, (rank01(ph) + rank01(pl)) / 2)
    base = score(te[F])
    for f in F:
        for r in range(5):
            X = te[F].copy(); X[f] = rng.permutation(X[f].values)
            rows.append({'date': td, 'feat': f, 'rep': r, 'drop': base - score(X)})
    print(td, 'base', round(base, 4), f'{time.time()-t0:.0f}s')
pi = pd.DataFrame(rows).groupby('feat')['drop'].agg(['mean', 'std']).sort_values('mean', ascending=False)
pi.to_csv(f'{S}/interp/perm.csv')
print(pi.head(15).round(4))

# ---- 3. SHAP: 최종 학습(전체 학습쌍) → 9/1 테스트 예측 설명 ----
hs = [hgb(s).fit(TR_dl[F], TR_dl.target) for s in range(5)]
X = TE_defl[F].reset_index(drop=True)
sv = np.mean([shap.TreeExplainer(h).shap_values(X) for h in hs], axis=0)
if isinstance(sv, list): sv = sv[1]
lr = logreg().fit(TR_dl[F], TR_dl.target)
Z = lr.named_steps['sc'].transform(lr.named_steps['imp'].transform(X))
coef = lr.named_steps['m'].coef_[0]
lr_contrib = Z * coef                                   # 선형 모델의 정확한 SHAP (표준화 공간)
pickle.dump({'F': F, 'X': X, 'shap_hgb': sv, 'lr_contrib': lr_contrib, 'coef': coef,
             'row_id': TE_defl.row_id.values, 'TR_prev': TR_dl[['prev_f', 'has_prev', 'target']].copy()},
            open(f'{S}/interp/shap.pkl', 'wb'))
print('done', f'{time.time()-t0:.0f}s')
