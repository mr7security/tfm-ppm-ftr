#!/usr/bin/env python3
"""
conformance_bpmn_v3_destinos.py — BPMN de referencia v3: el modelo con "Trasladar a planta" opcional (el mismo que
se validó en v2 con pm4py y Celonis) al que se añade la decisión "¿Riesgo post-quirúrgico?" como tareas explícitas
DESTINO_* tras la cirugía:
    cirugía -> XOR -> DESTINO_UCI -> INGRESO_UCI -> ...
                   -> DESTINO_PLANTA | DESTINO_REANIMACION | DESTINO_OTROS -> (traslado a planta opcional) -> ...
Genera Discovery\\celonis\\bpmn_referencia_v3_destinos.bpmn (para Celonis, tareas con los nombres del log) y
mide la conformance en pm4py (log proyectado: QUIROFANO_FIN se excluye, igual que la desviación permitida en Celonis).
Uso: python conformance_bpmn_v3_destinos.py   (carpeta EventLogV3)
"""
import os, warnings; warnings.filterwarnings("ignore")
from collections import Counter
import pandas as pd, pm4py
from pm4py.objects.bpmn.obj import BPMN

SRC = os.path.join("Discovery", "celonis", "bpmn_referencia_celonis_traslado_opcional.bpmn")   # BPMN v2 validado
OUT_BPMN = os.path.join("Discovery", "celonis", "bpmn_referencia_v3_destinos.bpmn")
bp = pm4py.read_bpmn(SRC)
nodes = {n.get_id(): n for n in bp.get_nodes()}
flows = {f.get_id(): f for f in bp.get_flows()}
gw = nodes["Gateway_0sb461k"]            # ¿Complicación post-quirúrgica?
uci = nodes["Activity_08sdzat"]          # INGRESO_UCI
gw_planta = nodes["Gateway_0q50ke2"]     # antes de TRASLADO_PLANTA (opcional)
proc = gw.get_process()
for fid in ["Flow_19wpu48", "Flow_0l188eh"]:
    bp.remove_flow(flows[fid])
def task(name):
    t = BPMN.Task(name=name, process=proc); bp.add_node(t); return t
d_uci = task("DESTINO_UCI")
bp.add_flow(BPMN.SequenceFlow(gw, d_uci, name="Riesgo: reserva cama UCI", process=proc))
bp.add_flow(BPMN.SequenceFlow(d_uci, uci, process=proc))
for nm in ["DESTINO_PLANTA", "DESTINO_REANIMACION", "DESTINO_OTROS"]:
    t = task(nm)
    bp.add_flow(BPMN.SequenceFlow(gw, t, process=proc))
    bp.add_flow(BPMN.SequenceFlow(t, gw_planta, process=proc))
gw.set_name("¿Riesgo post-quirúrgico? (destino postoperatorio)")
from pm4py.objects.bpmn.layout import layouter
bp = layouter.apply(bp)
pm4py.write_bpmn(bp, OUT_BPMN)
pm4py.save_vis_bpmn(bp, os.path.join("conformance", "04_bpmn_referencia_v3_destinos.png"))

# conformance pm4py
log = pd.read_parquet("eventlog_pq.parquet")
proj = log[log["activity"] != "QUIROFANO_FIN"].copy()
df = pm4py.format_dataframe(proj.rename(columns={"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"}),
                            case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
df["case:concept:name"] = df["case:concept:name"].astype(str)
net, im, fm = pm4py.convert_to_petri_net(bp)
pm4py.write_pnml(net, im, fm, os.path.join("conformance", "bpmn_referencia_v3_destinos.pnml"))
al = pm4py.conformance_diagnostics_alignments(df, net, im, fm)
fa = pm4py.fitness_alignments(df, net, im, fm); pa = pm4py.precision_alignments(df, net, im, fm)
L = [f"BPMN v3 con destinos (traslado a planta opcional) — {len(net.places)} lugares, {len(net.transitions)} transiciones",
     f"Log proyectado sin QUIROFANO_FIN: {len(df):,} eventos",
     f"Alineaciones: fitness media {fa['average_trace_fitness']:.3f}, log fitness {fa['log_fitness']:.3f}, "
     f"trazas conformes {fa['percentage_of_fitting_traces']:.1f} %; precisión {pa:.3f}"]
sm, sl = Counter(), Counter()
for a in al:
    for (l, m) in a["alignment"]:
        if l == ">>" and m not in (None, ">>"): sm[m] += 1
        elif m == ">>" and l not in (None, ">>"): sl[l] += 1
L.append("  sólo en el modelo: " + ", ".join(f"{k} {v}" for k, v in sm.most_common()))
L.append("  sólo en el log: " + ", ".join(f"{k} {v}" for k, v in sl.most_common()))
cases = df["case:concept:name"].unique()
var = df.groupby("case:concept:name")["concept:name"].agg(">".join)
tab = pd.DataFrame({"variante": var, "fitness": pd.Series([a["fitness"] for a in al], index=cases)}) \
        .groupby("variante").agg(casos=("fitness", "size"), fitness=("fitness", "mean")).sort_values("casos", ascending=False)
tab.to_csv(os.path.join("conformance", "variantes_fitness_v3_destinos.csv"), sep=";")
L.append(tab.head(15).round(3).to_string())
txt = "\n".join(L); open(os.path.join("conformance", "conformance_bpmn_v3_destinos.txt"), "w", encoding="utf-8").write(txt); print(txt)
