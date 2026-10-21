"""
Variante C (label_uci_no_planificada) con validación cruzada estratificada por caso (5 particiones).
Sólo hay 21 casos positivos: con la partición temporal el test tiene 3 casos, así que se estima con CV
fuera de muestra (out-of-fold) sobre los 1.365 casos sin destino UCI previsto.
Métricas: AUC-ROC y AUC-PR agrupando las predicciones de las 5 particiones, (a) sobre todos los prefijos y
(b) en el prefijo que termina en el evento DESTINO_* (la salida de quirófano), que es el momento de decisión clínica.
"""
import os, time, warnings
import numpy as np, pandas as pd
import ppm_data_v3 as D
import run_experiments_v3 as R
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT = R.OUT
t0 = time.time()
ev, cases = D.load_log()
casesC = cases[cases.destino_postop_grupo.fillna("") != "UCI"].reset_index(drop=True)
meta, X, sa, sn, y = D.build_prefixes(ev, casesC, "outcome", variant="B", target="label_uci_no_planificada")
last_act = np.array([D.ACTS[a - 1] for a in sa[np.arange(len(sa)), meta.k.values - 1]])
at_dest = np.isin(last_act, list(D.DESTINO_ACTS))
lab = casesC.set_index("case_id")["label_uci_no_planificada"]
cid = casesC.case_id.values; cl = lab.loc[cid].values
static_cols = [c for c in X.columns if not (c.startswith("n_") or c.startswith("last_") or c in ("k", "elapsed_h", "since_last_h"))]
Xa = X.values.astype(np.float32)
lines = [f"Variante C con CV 5-fold por caso: {len(casesC)} casos ({cl.sum()} positivos), {len(meta)} prefijos ({y.sum()} positivos), "
         f"{at_dest.sum()} prefijos en la salida de quirófano ({y[at_dest].sum()} positivos)"]
print(lines[0])
res, oof_all = [], {}
for name in ["DecisionTree", "RandomForest", "XGBoost", "LSTM", "Transformer"]:
    per_seed = []
    for seed in R.SEEDS:
        oof = np.zeros(len(meta)); skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for f, (itr, ite) in enumerate(skf.split(cid, cl)):
            tr_c, te_c = cid[itr], cid[ite]
            # validación interna (early stopping) = 20 % de los casos de entrenamiento, estratificado
            skf2 = StratifiedKFold(5, shuffle=True, random_state=100 + f); i_in, i_va = next(skf2.split(tr_c, cl[itr]))
            tr = meta.case_id.isin(tr_c[i_in]).values; va = meta.case_id.isin(tr_c[i_va]).values; te = meta.case_id.isin(te_c).values
            sc = StandardScaler().fit(X.loc[tr, static_cols]); st = sc.transform(X[static_cols]).astype(np.float32)
            spw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
            if name in ("LSTM", "Transformer"):
                predict, _ = R.train_dl(name, sa, sn, st, y, tr, va, 1, seed); oof[te] = predict(np.where(te)[0])
            else:
                m = R.make_tab(name, seed, 1, spw)
                if name == "XGBoost": m.fit(Xa[tr], y[tr], eval_set=[(Xa[va], y[va])], verbose=False)
                else: m.fit(Xa[tr], y[tr])
                oof[te] = m.predict_proba(Xa[te])[:, 1]
        r = dict(modelo=name, semilla=seed, auc_roc=roc_auc_score(y, oof), auc_pr=average_precision_score(y, oof),
                 auc_roc_salida_qx=roc_auc_score(y[at_dest], oof[at_dest]), auc_pr_salida_qx=average_precision_score(y[at_dest], oof[at_dest]))
        res.append(r); per_seed.append(oof)
        print(f"  {name:13s} seed {seed}: AUC-ROC {r['auc_roc']:.3f} AUC-PR {r['auc_pr']:.3f} | salida quirófano AUC-ROC {r['auc_roc_salida_qx']:.3f} AUC-PR {r['auc_pr_salida_qx']:.3f}")
    oof_all[name] = np.mean(per_seed, 0)
df = pd.DataFrame(res); df.to_csv(os.path.join(OUT, "outcome_C_cv_por_semilla.csv"), sep=";", index=False)
summ = df.groupby("modelo")[["auc_roc", "auc_pr", "auc_roc_salida_qx", "auc_pr_salida_qx"]].agg(["mean", "std"])
summ.to_csv(os.path.join(OUT, "outcome_C_cv_resumen.csv"), sep=";")
# riesgo medio predicho por destino previsto en la salida de quirófano (Random Forest)
rf = oof_all["RandomForest"]
tab = pd.DataFrame({"destino": last_act[at_dest], "y": y[at_dest], "p": rf[at_dest]}).groupby(["destino", "y"])["p"].agg(["count", "mean"])
tab.to_csv(os.path.join(OUT, "outcome_C_cv_riesgo_por_destino_RF.csv"), sep=";")
# figura: ROC y PR out-of-fold
from sklearn.metrics import roc_curve, precision_recall_curve
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for name, pm in oof_all.items():
    fpr, tpr, _ = roc_curve(y, pm); pr, rc, _ = precision_recall_curve(y, pm)
    ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, pm):.3f})"); ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y, pm):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC out-of-fold — outcome_C (UCI no planificada)"); ax[0].legend(fontsize=8)
ax[1].axhline(y.mean(), color="k", ls="--", lw=0.8); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall out-of-fold — outcome_C"); ax[1].legend(fontsize=8)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "outcome_C_cv_curvas.png"), dpi=150); plt.close(fig)
out = "\n".join(lines) + "\n" + summ.round(3).to_string() + "\n\nRiesgo medio predicho (RF) en la salida de quirófano por destino y resultado:\n" + tab.round(3).to_string() + f"\n\nTiempo {time.time()-t0:.0f} s\n"
print(out); open(os.path.join(OUT, "outcome_C_cv_log.txt"), "w", encoding="utf-8").write(out)
