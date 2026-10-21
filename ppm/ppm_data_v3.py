"""
Preparación de datos para Predictive Process Monitoring sobre el event log v3
(proceso de ingreso programado quirúrgico).

Dos tareas:
  outcome  -> predecir un label de caso a partir de un prefijo (sin fuga: el prefijo termina antes del
              primer INGRESO_UCI y nunca en un evento de fin ALTA/EXITUS/TRASLADO_OTRO_HOSPITAL).
              Variantes (v3):
                A  label_ingreso_uci SIN información de planificación: se quitan los eventos DESTINO_* y la
                   anestesia Gral-UCI/Gral-REA se recodifica como General
                B  label_ingreso_uci CON el destino postoperatorio como actividad (DESTINO_*) y anestesia original
                   (cota superior: DESTINO_UCI es la reserva de cama UCI)
                C  label_uci_no_planificada: casos sin destino UCI previsto (se excluyen los 110 con DESTINO_UCI);
                   con DESTINO_* en el prefijo. ¿Se puede anticipar la UCI que nadie había previsto?
  nextact  -> predecir la siguiente actividad (tarea de Zare et al. 2025)

Dos codificaciones:
  agregada   (Decision Tree, Random Forest, XGBoost): recuentos de actividades, última actividad,
             tiempos, atributos del caso conocidos al ingreso y atributos de cirugía si ya ha ocurrido
  secuencial (LSTM, Transformer): secuencia de actividades + 2 rasgos temporales por evento, con padding,
             más el vector estático del caso

Partición temporal por fecha de ingreso: 80 % train (del que el 20 % final es validación) / 20 % test.
"""
import numpy as np, pandas as pd, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTS = ["INGRESO", "INTERCONSULTA_ESPECIALISTA", "INTERCONSULTA_UCI", "QUIROFANO_INICIO", "QUIROFANO_FIN",
        "DESTINO_PLANTA", "DESTINO_UCI", "DESTINO_REANIMACION", "DESTINO_OTROS",
        "INGRESO_UCI", "TRASLADO_PLANTA", "TRASLADO_OTRO_HOSPITAL", "EXITUS", "ALTA"]
DESTINO_ACTS = {"DESTINO_PLANTA", "DESTINO_UCI", "DESTINO_REANIMACION", "DESTINO_OTROS"}
A2I = {a: i + 1 for i, a in enumerate(ACTS)}          # 0 = padding
END_ACTS = {"ALTA", "EXITUS", "TRASLADO_OTRO_HOSPITAL"}
MAXLEN = 6
SURG_ATTRS = ["categoria", "abordaje", "anestesia"]
TARGET = "label_ingreso_uci"
# n_diag (nº de diagnósticos CIE-10 del episodio) se codifica al alta -> posible fuga; se excluye por defecto.
# Análisis de sensibilidad con n_diag: PPM_EXCLUDE="" PPM_OUT=resultados_con_ndiag
EXCLUDE_STATIC = set(os.environ.get("PPM_EXCLUDE", "n_diag").split(",")) - {""}


def load_log():
    ev = pd.read_csv(os.path.join(BASE, "eventlog_pq.csv"), sep=";", parse_dates=["timestamp"], low_memory=False)
    ev = ev.sort_values(["case_id", "timestamp", "event_id"]).reset_index(drop=True)
    # duración de la cirugía en el evento QUIROFANO_FIN (minutos desde el QUIROFANO_INICIO anterior)
    qi = ev.timestamp.where(ev.activity == "QUIROFANO_INICIO").groupby(ev.case_id).ffill()
    ev["duracion_cirugia_min"] = np.where(ev.activity == "QUIROFANO_FIN", (ev.timestamp - qi).dt.total_seconds() / 60, np.nan)
    cases = pd.read_csv(os.path.join(BASE, "eventlog_pq_casos.csv"), sep=";", parse_dates=["fecha_ingreso"])
    return ev, cases


def process_entropy(ev, with_start_end=True, base=2):
    """Entropía del proceso según Zare et al. 2025: H = -sum p(t) log p(t) sobre las transiciones directas."""
    seqs = ev.groupby("case_id")["activity"].apply(list)
    trans = {}
    for s in seqs:
        s2 = ["<START>"] + s + ["<END>"] if with_start_end else s
        for a, b in zip(s2[:-1], s2[1:]):
            trans[(a, b)] = trans.get((a, b), 0) + 1
    p = np.array(list(trans.values()), dtype=float); p /= p.sum()
    log = np.log2 if base == 2 else np.log
    return float(-(p * log(p)).sum()), len(trans)


# ---------------------------------------------------------------- estáticos del caso
def static_frame(cases):
    """Atributos conocidos en el momento del ingreso (sin nada que dependa del resultado)."""
    st = pd.DataFrame({"case_id": cases.case_id})
    st["edad"] = cases.edad_ingreso.astype(float)
    st["sexo_H"] = (cases.sexo == "H").astype(int)
    st["n_diag"] = cases.n_diagnosticos.fillna(0).astype(float)
    st["paliativo"] = (cases.proceso_paliativo == "Sí").astype(int)
    st["ing_hora"] = cases.fecha_ingreso.dt.hour.astype(float)
    st["ing_dow"] = cases.fecha_ingreso.dt.dayofweek.astype(float)
    top_serv = cases.servicio_ingreso.value_counts().index[:8]
    for s in top_serv:
        st["serv_" + s.split()[0].replace("Í","I").replace("Ó","O")] = (cases.servicio_ingreso == s).astype(int)
    top_cap = cases.cie_capitulo.value_counts().index[:8]
    for c in top_cap:
        st["cie_" + str(c)] = (cases.cie_capitulo == c).astype(int)
    return st.drop(columns=[c for c in EXCLUDE_STATIC if c in st.columns]).set_index("case_id")


# ---------------------------------------------------------------- prefijos
def build_prefixes(ev, cases, task="outcome", maxlen=MAXLEN, variant="B", target=TARGET):
    """Devuelve (meta, X_agg, seq_act, seq_num, y).
    meta: case_id, k (longitud del prefijo), fecha_ingreso, label"""
    st = static_frame(cases)
    lab = cases.set_index("case_id")[target]
    fing = cases.set_index("case_id")["fecha_ingreso"]
    rows, seqa, seqn, ys = [], [], [], []
    ev = ev[ev.case_id.isin(cases.case_id)].copy()
    if variant == "A":   # sin información de planificación: fuera DESTINO_* y anestesia Gral-UCI / Gral-REA -> General
        ev = ev[~ev.activity.isin(DESTINO_ACTS)]
        ev["anestesia"] = ev["anestesia"].replace({"Gral-UCI": "General", "Gral-REA": "General"})
    surg_levels = {a: sorted(ev[a].dropna().unique()) for a in SURG_ATTRS}
    for cid, g in ev.groupby("case_id", sort=False):
        acts = g.activity.tolist(); ts = g.timestamp.tolist(); n = len(acts)
        if task == "outcome":
            cut = acts.index("INGRESO_UCI") if "INGRESO_UCI" in acts else n
            ks = [k for k in range(1, min(cut, maxlen) + 1) if acts[k - 1] not in END_ACTS]
        else:
            ks = [k for k in range(1, min(n - 1, maxlen) + 1)]
        base_static = st.loc[cid]
        for k in ks:
            pa, pt = acts[:k], ts[:k]
            elapsed = (pt[-1] - pt[0]).total_seconds() / 3600
            since_last = (pt[-1] - pt[-2]).total_seconds() / 3600 if k > 1 else 0.0
            r = {"k": k, "elapsed_h": elapsed, "since_last_h": since_last}
            for a in ACTS:
                r["n_" + a] = pa.count(a); r["last_" + a] = int(pa[-1] == a)
            # atributos de cirugía: sólo si QUIROFANO_INICIO ya está en el prefijo
            gq = g.iloc[:k]; q = gq[gq.activity == "QUIROFANO_INICIO"]
            r["cirugia_en_prefijo"] = int(len(q) > 0)
            for a in SURG_ATTRS:
                val = q[a].iloc[0] if len(q) else None
                for lv in surg_levels[a]:
                    r[f"{a}_{lv}"] = int(val == lv)
            qf = gq[gq.activity == "QUIROFANO_FIN"]
            r["dur_cirugia_min"] = float(qf["duracion_cirugia_min"].iloc[0]) if len(qf) and pd.notna(qf["duracion_cirugia_min"].iloc[0]) else 0.0
            r.update(base_static.to_dict())
            rows.append((cid, k, fing[cid], r))
            # secuencial
            sa = np.zeros(maxlen, dtype=np.int64); sn = np.zeros((maxlen, 2), dtype=np.float32)
            for i in range(k):
                sa[i] = A2I[pa[i]]
                sn[i, 0] = np.log1p((pt[i] - pt[0]).total_seconds() / 3600)
                sn[i, 1] = np.log1p((pt[i] - pt[i - 1]).total_seconds() / 3600) if i > 0 else 0.0
            seqa.append(sa); seqn.append(sn)
            ys.append(int(lab[cid]) if task == "outcome" else A2I[acts[k]] - 1)
    meta = pd.DataFrame([(c, k, f) for c, k, f, _ in rows], columns=["case_id", "k", "fecha_ingreso"])
    X = pd.DataFrame([r for *_, r in rows]).fillna(0.0)
    return meta, X, np.stack(seqa), np.stack(seqn), np.array(ys)


def temporal_split(meta, cases, train_frac=0.8, val_frac=0.2):
    """Partición por caso según fecha de ingreso: primeros 80 % train (últimos 20 % de ellos = val), 20 % test."""
    order = cases.sort_values("fecha_ingreso").case_id.tolist()
    n = len(order); n_tr = int(n * train_frac); n_val = int(n_tr * val_frac)
    tr, va, te = set(order[:n_tr - n_val]), set(order[n_tr - n_val:n_tr]), set(order[n_tr:])
    return meta.case_id.isin(tr).values, meta.case_id.isin(va).values, meta.case_id.isin(te).values
