"""
Predictive Process Monitoring sobre el event log v3: DOS OBJETIVOS NUEVOS, con la misma receta que la variante A
del 29/09 (5 algoritmos, configuración básica, 3 semillas, prefijos k = 1..6, partición temporal, sin n_diag).
No modifica nada de resultados/ ni de resultados_con_ndiag/: todo sale en resultados_targets/.

Experimentos:
  outcome_D_complicacion        label_complicacion_postop (240 casos, 16,3 %) SIN planificación (sin DESTINO_*,
                                anestesia Gral-UCI/REA -> General). PRINCIPAL. El prefijo termina antes del primer
                                evento que define la etiqueta: INGRESO_UCI o INTERCONSULTA_ESPECIALISTA en fase
                                postoperatoria, EXITUS o TRASLADO_OTRO_HOSPITAL. Las interconsultas preoperatorias
                                y los ingresos en UCI preoperatorios / sin cirugía no definen la etiqueta y se mantienen.
  outcome_D_complicacion_plan   lo mismo CON el destino postoperatorio (DESTINO_*) y la anestesia original: cota
                                superior, porque 105 de los 240 positivos son UCI planificada (DESTINO_UCI).
  outcome_E_estancia            label_estancia_prolongada (estancia > p90 = 8,74 d; 148 casos, 10 %) con el log
                                completo (el destino previsto no es fuga para la estancia). PRINCIPAL. Además de no
                                terminar en ALTA/EXITUS/TRASLADO, se descartan los prefijos cuyo tiempo transcurrido
                                desde el ingreso ya supera el umbral (ahí la etiqueta ya se conoce).
  outcome_E_estancia_sinplan    lo mismo sin planificación (estilo A), para comparar con la variante A.

Reutiliza los modelos, métricas y figuras de run_experiments_v3.run_outcome; sólo cambia la regla de corte de los
prefijos, que aquí se generaliza (build_prefixes_target) y se inyecta en ppm_data_v3.build_prefixes.

Uso:  python run_experiments_targets.py        (salida en PPM/resultados_targets/)
"""
import os, sys, time, warnings
os.environ.setdefault("PPM_OUT", "resultados_targets")      # run_experiments_v3 lee PPM_OUT al importarse
import numpy as np, pandas as pd
import ppm_data_v3 as D
import run_experiments_v3 as R

warnings.filterwarnings("ignore")
OUT = R.OUT
ACTS, A2I, END_ACTS, DESTINO_ACTS, SURG_ATTRS = D.ACTS, D.A2I, D.END_ACTS, D.DESTINO_ACTS, D.SURG_ATTRS
POSTOP_LABEL_ACTS = {"INGRESO_UCI", "INTERCONSULTA_ESPECIALISTA"}   # cuentan para la etiqueta sólo en fase postoperatoria
HARD_LABEL_ACTS = {"EXITUS", "TRASLADO_OTRO_HOSPITAL"}


# ------------------------------------------------------------------ reglas de corte
def cut_complicacion(acts, ts, fases, n, maxlen):
    """k válidos para label_complicacion_postop: antes del primer evento que define la etiqueta."""
    cut = n
    for i, (a, f) in enumerate(zip(acts, fases)):
        if (a in POSTOP_LABEL_ACTS and f == "postoperatorio") or a in HARD_LABEL_ACTS:
            cut = i; break
    return [k for k in range(1, min(cut, maxlen) + 1) if acts[k - 1] not in END_ACTS]


def make_cut_estancia(umbral_dias):
    def cut_estancia(acts, ts, fases, n, maxlen):
        """k válidos para label_estancia_prolongada: no termina en fin y el tiempo transcurrido no supera el umbral."""
        return [k for k in range(1, min(n, maxlen) + 1)
                if acts[k - 1] not in END_ACTS and (ts[k - 1] - ts[0]).total_seconds() / 86400 < umbral_dias]
    return cut_estancia


# ------------------------------------------------------------------ prefijos con regla de corte genérica
def build_prefixes_target(ev, cases, task="outcome", maxlen=D.MAXLEN, variant="B", target=D.TARGET, cut_rule=None):
    """Copia de ppm_data_v3.build_prefixes (tarea outcome) con la regla de corte parametrizada."""
    st = D.static_frame(cases)
    lab = cases.set_index("case_id")[target]
    fing = cases.set_index("case_id")["fecha_ingreso"]
    rows, seqa, seqn, ys = [], [], [], []
    ev = ev[ev.case_id.isin(cases.case_id)].copy()
    if variant == "A":
        ev = ev[~ev.activity.isin(DESTINO_ACTS)]
        ev["anestesia"] = ev["anestesia"].replace({"Gral-UCI": "General", "Gral-REA": "General"})
    surg_levels = {a: sorted(ev[a].dropna().unique()) for a in SURG_ATTRS}
    for cid, g in ev.groupby("case_id", sort=False):
        acts = g.activity.tolist(); ts = g.timestamp.tolist(); fases = g.fase.tolist(); n = len(acts)
        ks = cut_rule(acts, ts, fases, n, maxlen)
        base_static = st.loc[cid]
        for k in ks:
            pa, pt = acts[:k], ts[:k]
            elapsed = (pt[-1] - pt[0]).total_seconds() / 3600
            since_last = (pt[-1] - pt[-2]).total_seconds() / 3600 if k > 1 else 0.0
            r = {"k": k, "elapsed_h": elapsed, "since_last_h": since_last}
            for a in ACTS:
                r["n_" + a] = pa.count(a); r["last_" + a] = int(pa[-1] == a)
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
            sa = np.zeros(maxlen, dtype=np.int64); sn = np.zeros((maxlen, 2), dtype=np.float32)
            for i in range(k):
                sa[i] = A2I[pa[i]]
                sn[i, 0] = np.log1p((pt[i] - pt[0]).total_seconds() / 3600)
                sn[i, 1] = np.log1p((pt[i] - pt[i - 1]).total_seconds() / 3600) if i > 0 else 0.0
            seqa.append(sa); seqn.append(sn)
            ys.append(int(lab[cid]))
    meta = pd.DataFrame([(c, k, f) for c, k, f, _ in rows], columns=["case_id", "k", "fecha_ingreso"])
    X = pd.DataFrame([r for *_, r in rows]).fillna(0.0)
    return meta, X, np.stack(seqa), np.stack(seqn), np.array(ys)


def run_with_cut(tag, variant, ev, cases, target, cut_rule):
    """Ejecuta run_experiments_v3.run_outcome con la regla de corte indicada (inyectada en ppm_data_v3)."""
    D.build_prefixes = lambda ev_, cases_, task="outcome", maxlen=D.MAXLEN, variant="B", target=D.TARGET: \
        build_prefixes_target(ev_, cases_, task, maxlen, variant, target, cut_rule=cut_rule)
    # descriptivo de los prefijos: cuántos casos positivos conservan algún prefijo y hasta qué k
    meta, X, sa, sn, y = D.build_prefixes(ev, cases, variant=variant, target=target)
    lab = cases.set_index("case_id")[target]
    casos_con_prefijo = meta.case_id.nunique(); pos_con_prefijo = meta[meta.case_id.map(lab) == 1].case_id.nunique()
    R.p(f"\n--- {tag}: casos con algún prefijo {casos_con_prefijo}/{len(cases)}; positivos con algún prefijo "
        f"{pos_con_prefijo}/{int(lab.sum())}; prefijos por k: {meta.k.value_counts().sort_index().to_dict()}; "
        f"positivos por k: {meta[meta.case_id.map(lab) == 1].k.value_counts().sort_index().to_dict()}")
    return R.run_outcome(tag, variant, ev, cases, target=target)


if __name__ == "__main__":
    R.LOG = open(os.path.join(OUT, "log_ejecucion_targets.txt"), "w", encoding="utf-8")
    t_all = time.time()
    ev, cases = D.load_log()
    umbral = float(cases.estancia_dias.quantile(0.90))
    R.p(f"Event log v3: {len(cases)} casos, {len(ev)} eventos. Receta de la variante A: 5 algoritmos, 3 semillas, "
        f"k = 1..{D.MAXLEN}, partición temporal 64/16/20, estáticos excluidos {sorted(D.EXCLUDE_STATIC)}")
    R.p(f"label_complicacion_postop: {int(cases.label_complicacion_postop.sum())} positivos "
        f"(UCI postop {int(cases.ingreso_uci_postop.sum())}, interconsulta especialista postop "
        f"{int(cases.interconsulta_especialista_postop.sum())}, exitus {int(cases.label_exitus_intrahosp.sum())}, "
        f"traslado {int(cases.label_traslado_otro_hosp.sum())}); con DESTINO_UCI: "
        f"{int(((cases.label_complicacion_postop == 1) & (cases.destino_postop_grupo == 'UCI')).sum())}")
    R.p(f"label_estancia_prolongada: {int(cases.label_estancia_prolongada.sum())} positivos; umbral p90 = {umbral:.3f} días")

    run_with_cut("outcome_D_complicacion", "A", ev, cases, "label_complicacion_postop", cut_complicacion)
    run_with_cut("outcome_D_complicacion_plan", "B", ev, cases, "label_complicacion_postop", cut_complicacion)
    run_with_cut("outcome_E_estancia", "B", ev, cases, "label_estancia_prolongada", make_cut_estancia(umbral))
    run_with_cut("outcome_E_estancia_sinplan", "A", ev, cases, "label_estancia_prolongada", make_cut_estancia(umbral))
    R.p(f"\nTiempo total: {time.time() - t_all:.0f} s")
    R.LOG.close()
