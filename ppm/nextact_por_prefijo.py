"""Accuracy of next-activity prediction (Task 1, F) by prefix length k, test set, mean of three seeds.
Same data, split, models and seeds as run_final_v3 / run_experiments_v3.run_nextact."""
import os, sys, time
os.environ["PPM_OUT"] = "resultados_final"
import numpy as np, pandas as pd
import ppm_data_v3 as D
import run_experiments_v3 as R
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

R.LOG = open(os.devnull, "w")
ev, cases = D.load_log()
meta, X, sa, sn, y = D.build_prefixes(ev, cases, "nextact", variant="B")
tr, va, te = D.temporal_split(meta, cases)
classes = np.unique(y); remap = {c: i for i, c in enumerate(classes)}; yr = np.array([remap[v] for v in y]); n_out = len(classes)
static_cols = [c for c in X.columns if not (c.startswith("n_") or c.startswith("last_") or c in ("k", "elapsed_h", "since_last_h"))]
sc = StandardScaler().fit(X.loc[tr, static_cols]); st = sc.transform(X[static_cols]).astype(np.float32); Xa = X.values.astype(np.float32)
kte = meta.k.values[te]; yte = yr[te]
# baseline
last = sa[np.arange(len(sa)), meta.k.values - 1]
tab = pd.crosstab(last[tr], yr[tr]); guess = tab.idxmax(1); base_pred = np.array([guess.get(l, tab.sum().idxmax()) for l in last[te]])
rows = []
for k in sorted(set(kte)):
    m = kte == k; rows.append(dict(modelo="Baseline", semilla=0, k=k, n=int(m.sum()), accuracy=accuracy_score(yte[m], base_pred[m])))
rows.append(dict(modelo="Baseline", semilla=0, k=0, n=len(yte), accuracy=accuracy_score(yte, base_pred)))
for name in ["DecisionTree", "RandomForest", "XGBoost", "LSTM", "Transformer"]:
    for seed in R.SEEDS:
        t0 = time.time()
        if name in ("LSTM", "Transformer"):
            predict, ep = R.train_dl(name, sa, sn, st, yr, tr, va, n_out, seed); pt = predict(np.where(te)[0])
        else:
            mdl = R.make_tab(name, seed, n_out, 1.0)
            if name == "XGBoost": mdl.fit(Xa[tr], yr[tr], eval_set=[(Xa[va], yr[va])], verbose=False)
            else: mdl.fit(Xa[tr], yr[tr])
            pt = mdl.predict_proba(Xa[te]); full = np.zeros((len(pt), n_out)); full[:, mdl.classes_] = pt; pt = full
        pred = pt.argmax(1)
        rows.append(dict(modelo=name, semilla=seed, k=0, n=len(yte), accuracy=accuracy_score(yte, pred)))
        for k in sorted(set(kte)):
            m = kte == k; rows.append(dict(modelo=name, semilla=seed, k=k, n=int(m.sum()), accuracy=accuracy_score(yte[m], pred[m])))
        print(name, seed, f"acc {accuracy_score(yte, pred):.3f}", f"{time.time()-t0:.1f}s", flush=True)
res = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_final"); os.makedirs(out, exist_ok=True)
res.to_csv(os.path.join(out, "nextact_por_prefijo.csv"), sep=";", index=False)
summ = res.groupby(["modelo", "k"]).agg(n=("n", "first"), accuracy=("accuracy", "mean")).reset_index()
piv = summ.pivot(index="k", columns="modelo", values="accuracy")[["Baseline", "DecisionTree", "RandomForest", "XGBoost", "LSTM", "Transformer"]]
piv["n"] = summ[summ.modelo == "Baseline"].set_index("k")["n"]
print(piv.round(3).to_string())
piv.to_csv(os.path.join(out, "nextact_por_prefijo_resumen.csv"), sep=";")
