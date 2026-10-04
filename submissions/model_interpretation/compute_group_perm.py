import os, time, pickle
S = os.environ['S']
exec(open(f'{S}/nb10_core.py').read().replace('from IPython.display import display', 'display = lambda *a, **k: None').replace("validate_raw(TR_dl", "(lambda *a: {})(TR_dl"))
F = PAIR + CONS
G = {
 '08-25 라벨·역추론': PREVF + CONS,
 '과거 수준': ['log_past_med', 'log_q10', 'log_q90', 'log_view_std', 'n_long_5d', 'log_n_long_5d'],
 '업로드 활동': ['n_long_7', 'n_long_14', 'n_long_28', 'n_short_14', 'short_ratio', 'days_since_long', 'days_since_meas', 'avg_dur'],
 '최근 모멘텀': ['mom_7', 'mom_14', 'mom_old', 'trend_slope', 'log_short_med_rel'],
 '시청자 반응': ['like_rate', 'comment_rate'],
 '게임': ['n_games', 'has_game'],
}
assert sorted(sum(G.values(), [])) == sorted(F), set(F) ^ set(sum(G.values(), []))
rng = np.random.default_rng(0); rows = []
for td in EVAL_DATES:
    te = TR_dl[TR_dl.ref_date == td].reset_index(drop=True)
    tr = TR_dl[TR_dl.ref_date <= (pd.Timestamp(td) - pd.Timedelta(days=4)).strftime('%Y-%m-%d')]
    hs = [hgb(s).fit(tr[F], tr.target) for s in range(5)]; lr = logreg().fit(tr[F], tr.target)
    def score(X):
        return average_precision_score(te.target_raw, (rank01(np.mean([h.predict_proba(X)[:, 1] for h in hs], 0)) + rank01(lr.predict_proba(X)[:, 1])) / 2)
    base = score(te[F])
    for g, cols in G.items():
        for r in range(10):
            X = te[F].copy(); idx = rng.permutation(len(X))
            X[cols] = X[cols].values[idx]          # 그룹 전체를 같은 순서로 섞음 (그룹 내부 관계 유지)
            rows.append({'grp': g, 'drop': base - score(X)})
gp = pd.DataFrame(rows).groupby('grp')['drop'].agg(['mean', 'std']).sort_values('mean', ascending=False)
gp.to_csv(f'{S}/interp/perm_group.csv'); print(gp.round(4))
pickle.dump(G, open(f'{S}/interp/groups.pkl', 'wb'))
