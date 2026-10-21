"""
Predictive Process Monitoring sobre el event log v3 (destino postoperatorio como actividad) con los 5 algoritmos del esquema
(Decision Tree, Random Forest, XGBoost, LSTM, Transformer), configuración básica, 3 semillas.

Experimentos:
  outcome_A : label_ingreso_uci, sin planificación (sin eventos DESTINO_*, anestesia Gral-UCI/REA -> General)
  outcome_B : label_ingreso_uci, con DESTINO_* como actividad (cota superior: DESTINO_UCI = reserva de cama UCI)
  outcome_C : label_uci_no_planificada en los casos sin DESTINO_UCI (con DESTINO_* en el prefijo)
  nextact   : siguiente actividad (tarea de Zare et al. 2025)
Salidas en PPM/resultados/
"""
import os, time, json, warnings, sys
import numpy as np, pandas as pd
import ppm_data_v3 as D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score,
                             brier_score_loss, accuracy_score, precision_recall_curve, roc_curve, confusion_matrix)
import xgboost as xgb
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("PPM_OUT", "resultados")); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
torch.set_num_threads(2)
LOG = None
def p(*a):
    s = " ".join(str(x) for x in a); print(s)
    if LOG: LOG.write(s + "\n"); LOG.flush()


# ------------------------------------------------------------------ modelos secuenciales
class LSTMNet(nn.Module):
    def __init__(self, n_act, n_static, n_out, emb=16, hid=64):
        super().__init__()
        self.emb = nn.Embedding(n_act + 1, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb + 2, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + n_static, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_out))
    def forward(self, a, x, s):
        lengths = (a > 0).sum(1).clamp(min=1)
        h = torch.cat([self.emb(a), x], -1)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.head(torch.cat([hn[-1], s], -1))


class TransformerNet(nn.Module):
    def __init__(self, n_act, n_static, n_out, d=32, heads=4, layers=2, maxlen=D.MAXLEN):
        super().__init__()
        self.emb = nn.Embedding(n_act + 1, d, padding_idx=0)
        self.num = nn.Linear(2, d)
        self.pos = nn.Parameter(torch.randn(1, maxlen, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d + n_static, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_out))
    def forward(self, a, x, s):
        mask = a == 0
        h = self.emb(a) + self.num(x) + self.pos[:, :a.shape[1]]
        h = self.enc(h, src_key_padding_mask=mask)
        m = (~mask).unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(torch.cat([pooled, s], -1))


def train_dl(kind, sa, sn, st, y, tr, va, n_out, seed, epochs=80, patience=12, lr=1e-3, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    n_static = st.shape[1]
    net = LSTMNet(len(D.ACTS), n_static, n_out) if kind == "LSTM" else TransformerNet(len(D.ACTS), n_static, n_out)
    T = lambda z, dt=torch.float32: torch.tensor(z, dtype=dt)
    A, X, S = T(sa, torch.long), T(sn), T(st)
    if n_out == 1:
        Y = T(y).unsqueeze(1); pos_w = torch.tensor([(y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        Y = T(y, torch.long); loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    idx_tr = np.where(tr)[0]; idx_va = np.where(va)[0]
    best, best_state, bad = -1, None, 0
    def score(idx):
        net.eval()
        with torch.no_grad():
            out = net(A[idx], X[idx], S[idx])
        if n_out == 1:
            return roc_auc_score(y[idx], out.squeeze(1).numpy())
        return accuracy_score(y[idx], out.argmax(1).numpy())
    for ep in range(epochs):
        net.train(); rng = np.random.default_rng(seed + ep); perm = rng.permutation(idx_tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = loss_fn(net(A[b], X[b], S[b]), Y[b]); loss.backward(); opt.step()
        sc = score(idx_va)
        if sc > best: best, bad, best_state = sc, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    net.load_state_dict(best_state); net.eval()
    def predict(idx):
        with torch.no_grad():
            out = net(A[idx], X[idx], S[idx])
        return torch.sigmoid(out).squeeze(1).numpy() if n_out == 1 else torch.softmax(out, 1).numpy()
    return predict, ep + 1


# ------------------------------------------------------------------ modelos tabulares
def make_tab(name, seed, n_out, spw):
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                      n_jobs=2, random_state=seed)
    if n_out == 1:
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="auc",
                                 early_stopping_rounds=30, random_state=seed, n_jobs=2)
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                             objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=30,
                             random_state=seed, n_jobs=2)


def bin_metrics(y, prob, thr):
    pred = (prob >= thr).astype(int)
    return dict(auc_roc=roc_auc_score(y, prob), auc_pr=average_precision_score(y, prob),
                f1=f1_score(y, pred), bal_acc=balanced_accuracy_score(y, pred), brier=brier_score_loss(y, prob),
                recall=(pred[y == 1] == 1).mean(), precision=(y[pred == 1] == 1).mean() if pred.sum() else 0.0)


def best_threshold(y, prob):
    pr, rc, th = precision_recall_curve(y, prob); f = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.argmax(f[:-1])]) if len(th) else 0.5


# ------------------------------------------------------------------ experimento outcome
def run_outcome(tag, variant, ev, cases, target=D.TARGET):
    p(f"\n=== {tag} (variante {variant}, objetivo {target}, casos {len(cases)}) ===")
    meta, X, sa, sn, y = D.build_prefixes(ev, cases, "outcome", variant=variant, target=target)
    tr, va, te = D.temporal_split(meta, cases)
    p(f"prefijos {len(meta)} (train {tr.sum()} / val {va.sum()} / test {te.sum()}), positivos {y.sum()} "
      f"(test {y[te].sum()}), features agregadas {X.shape[1]}")
    static_cols = [c for c in X.columns if not (c.startswith("n_") or c.startswith("last_") or c in ("k", "elapsed_h", "since_last_h"))]
    sc = StandardScaler().fit(X.loc[tr, static_cols]); st = sc.transform(X[static_cols]).astype(np.float32)
    Xa = X.values.astype(np.float32); spw = (y[tr] == 0).sum() / (y[tr] == 1).sum()
    rows, curves, per_k, importances = [], {}, [], {}
    for name in ["DecisionTree", "RandomForest", "XGBoost", "LSTM", "Transformer"]:
        probs = []
        if y[va].sum() < 1 or y[te].sum() < 1:
            p("  sin positivos en validación o test: se omite"); break
        for seed in SEEDS:
            t0 = time.time()
            if name in ("LSTM", "Transformer"):
                predict, ep = train_dl(name, sa, sn, st, y, tr, va, 1, seed)
                pv, pt = predict(np.where(va)[0]), predict(np.where(te)[0]); extra = f"epochs={ep}"
            else:
                m = make_tab(name, seed, 1, spw)
                if name == "XGBoost": m.fit(Xa[tr], y[tr], eval_set=[(Xa[va], y[va])], verbose=False); extra = f"trees={m.best_iteration + 1}"
                else: m.fit(Xa[tr], y[tr]); extra = ""
                pv, pt = m.predict_proba(Xa[va])[:, 1], m.predict_proba(Xa[te])[:, 1]
                if name in ("RandomForest", "XGBoost") and seed == 0:
                    importances[name] = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
            dt = time.time() - t0; thr = best_threshold(y[va], pv)
            met = bin_metrics(y[te], pt, thr); met.update(modelo=name, semilla=seed, umbral=thr, tiempo_s=dt, detalle=extra)
            rows.append(met); probs.append(pt)
            p(f"  {name:13s} seed {seed}: AUC-ROC {met['auc_roc']:.3f}  AUC-PR {met['auc_pr']:.3f}  F1 {met['f1']:.3f} "
              f"(umbral {thr:.2f}, recall {met['recall']:.2f}, prec {met['precision']:.2f})  bal.acc {met['bal_acc']:.3f}  {dt:.1f}s {extra}")
        pm = np.mean(probs, 0); curves[name] = pm
        for k in sorted(meta.k[te].unique()):
            mk = (meta.k.values[te] == k)
            if y[te][mk].sum() >= (3 if tag != "outcome_C" else 1) and (y[te][mk] == 0).sum() >= 3:
                per_k.append(dict(experimento=tag, modelo=name, k=int(k), n=int(mk.sum()), positivos=int(y[te][mk].sum()),
                                  auc_roc=roc_auc_score(y[te][mk], pm[mk]), auc_pr=average_precision_score(y[te][mk], pm[mk])))
    res = pd.DataFrame(rows); res.insert(0, "experimento", tag)
    res.to_csv(os.path.join(OUT, f"{tag}_por_semilla.csv"), sep=";", index=False)
    summ = res.groupby("modelo")[["auc_roc", "auc_pr", "f1", "bal_acc", "recall", "precision", "brier", "tiempo_s"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(OUT, f"{tag}_resumen.csv"), sep=";")
    pd.DataFrame(per_k).to_csv(os.path.join(OUT, f"{tag}_por_prefijo.csv"), sep=";", index=False)
    for n, s in importances.items():
        s.head(20).to_csv(os.path.join(OUT, f"{tag}_importancia_{n}.csv"), sep=";", header=["importancia"])
    # figuras
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, pm in curves.items():
        fpr, tpr, _ = roc_curve(y[te], pm); pr, rc, _ = precision_recall_curve(y[te], pm)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y[te], pm):.3f})"); ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y[te], pm):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(xlabel="FPR", ylabel="TPR", title=f"ROC test — {tag}"); ax[0].legend(fontsize=8)
    ax[1].axhline(y[te].mean(), color="k", ls="--", lw=0.8); ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall test — {tag}"); ax[1].legend(fontsize=8)
    plt.tight_layout(); fig.savefig(os.path.join(OUT, f"{tag}_curvas.png"), dpi=150); plt.close(fig)
    pk = pd.DataFrame(per_k)
    if len(pk):
        piv = pk.pivot(index="k", columns="modelo", values="auc_roc")
        ax = piv.plot.bar(figsize=(8, 4), rot=0); ax.set(ylabel="AUC-ROC", xlabel="longitud del prefijo k", title=f"AUC-ROC por longitud de prefijo — {tag}", ylim=(0.4, 1))
        ax.legend(fontsize=8, ncol=3); plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{tag}_auc_por_prefijo.png"), dpi=150); plt.close()
    if "XGBoost" in importances:
        s = importances["XGBoost"].head(15)[::-1]
        plt.figure(figsize=(7, 4.5)); plt.barh(s.index, s.values, color="#3b6ea5"); plt.title(f"Importancia de variables XGBoost — {tag}")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"{tag}_importancia_xgb.png"), dpi=150); plt.close()
    p(summ.round(3).to_string())
    return summ


# ------------------------------------------------------------------ experimento next activity
def run_nextact(ev, cases):
    tag = "nextact"; p(f"\n=== {tag} ===")
    meta, X, sa, sn, y = D.build_prefixes(ev, cases, "nextact", variant="B")
    tr, va, te = D.temporal_split(meta, cases)
    classes = np.unique(y); remap = {c: i for i, c in enumerate(classes)}; yr = np.array([remap[v] for v in y]); n_out = len(classes)
    names = [D.ACTS[c] for c in classes]
    p(f"prefijos {len(meta)}, clases {n_out}: " + ", ".join(f"{D.ACTS[c]} {int((y == c).sum())}" for c in classes))
    static_cols = [c for c in X.columns if not (c.startswith("n_") or c.startswith("last_") or c in ("k", "elapsed_h", "since_last_h"))]
    sc = StandardScaler().fit(X.loc[tr, static_cols]); st = sc.transform(X[static_cols]).astype(np.float32); Xa = X.values.astype(np.float32)
    rows, cms = [], {}
    for name in ["DecisionTree", "RandomForest", "XGBoost", "LSTM", "Transformer"]:
        for seed in SEEDS:
            t0 = time.time()
            if name in ("LSTM", "Transformer"):
                predict, ep = train_dl(name, sa, sn, st, yr, tr, va, n_out, seed); pt = predict(np.where(te)[0]); extra = f"epochs={ep}"
            else:
                m = make_tab(name, seed, n_out, 1.0)
                if name == "XGBoost": m.fit(Xa[tr], yr[tr], eval_set=[(Xa[va], yr[va])], verbose=False); extra = f"trees={m.best_iteration + 1}"
                else: m.fit(Xa[tr], yr[tr]); extra = ""
                pt = m.predict_proba(Xa[te]); full = np.zeros((len(pt), n_out)); full[:, m.classes_] = pt; pt = full
            pred = pt.argmax(1); dt = time.time() - t0
            met = dict(modelo=name, semilla=seed, accuracy=accuracy_score(yr[te], pred), f1_macro=f1_score(yr[te], pred, average="macro"),
                       f1_weighted=f1_score(yr[te], pred, average="weighted"), tiempo_s=dt, detalle=extra)
            rows.append(met); cms[(name, seed)] = confusion_matrix(yr[te], pred, labels=range(n_out))
            p(f"  {name:13s} seed {seed}: accuracy {met['accuracy']:.3f}  F1-macro {met['f1_macro']:.3f}  F1-pond {met['f1_weighted']:.3f}  {dt:.1f}s {extra}")
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, f"{tag}_por_semilla.csv"), sep=";", index=False)
    summ = res.groupby("modelo")[["accuracy", "f1_macro", "f1_weighted", "tiempo_s"]].agg(["mean", "std"]); summ.to_csv(os.path.join(OUT, f"{tag}_resumen.csv"), sep=";")
    # baseline: actividad siguiente más frecuente dada la última actividad
    last = sa[np.arange(len(sa)), meta.k.values - 1]
    tab = pd.crosstab(last[tr], yr[tr]); guess = tab.idxmax(1); base_pred = np.array([guess.get(l, tab.sum().idxmax()) for l in last[te]])
    p(f"  baseline (siguiente más frecuente dada la última actividad): accuracy {accuracy_score(yr[te], base_pred):.3f}  F1-macro {f1_score(yr[te], base_pred, average='macro'):.3f}")
    best = summ["accuracy"]["mean"].idxmax(); cm = cms[(best, 0)]; cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_out)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(n_out)); ax.set_yticklabels(names, fontsize=7)
    for i in range(n_out):
        for j in range(n_out):
            if cm[i, j]: ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if cmn[i, j] > 0.6 else "black")
    ax.set(xlabel="predicha", ylabel="real", title=f"Siguiente actividad — matriz de confusión normalizada ({best}, test)")
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); fig.savefig(os.path.join(OUT, "nextact_confusion.png"), dpi=150); plt.close(fig)
    p(summ.round(3).to_string())
    return summ


if __name__ == "__main__":
    LOG = open(os.path.join(OUT, "log_ejecucion.txt"), "w", encoding="utf-8")
    t_all = time.time()
    ev, cases = D.load_log()
    for se in (True, False):
        h2, n = D.process_entropy(ev, se, 2); he, _ = D.process_entropy(ev, se, "e")
        p(f"Entropía del proceso ({'con' if se else 'sin'} START/END): {h2:.3f} bits (log2) / {he:.3f} nats (ln), {n} transiciones distintas")
    p(f"Referencia Zare 2025 (tabla 1): Helpdesk 2,60 · Road Traffic Fine 2,96 · BPI2020 RfP 3,21 · BPI2017_O 3,24 · BPI2020 PTC 3,64 · Sepsis 5,07")
    sA = run_outcome("outcome_A", "A", ev, cases)
    sB = run_outcome("outcome_B", "B", ev, cases)
    casesC = cases[cases.destino_postop_grupo.fillna("") != "UCI"].reset_index(drop=True)
    sC = run_outcome("outcome_C", "B", ev, casesC, target="label_uci_no_planificada")
    sN = run_nextact(ev, cases)
    p(f"\nTiempo total: {time.time() - t_all:.0f} s")
    LOG.close()
