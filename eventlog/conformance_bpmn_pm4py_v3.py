#!/usr/bin/env python3
"""
conformance_bpmn_pm4py_v3.py — Comprueba el event log v3 contra el modelo de referencia
mapaHospitalizacionProgramadoQuirurgico.bpmn con pm4py (token replay + alineaciones) y
genera el DFG del log. Uso: python conformance_bpmn_pm4py_v3.py  (en la carpeta EventLogV3)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pm4py
from pm4py.objects.conversion.bpmn import converter as bpmn_converter

OUT = "conformance"
os.makedirs(OUT, exist_ok=True)

# 1. log: para compararlo con el BPMN se proyecta la cirugía a un único evento (QUIROFANO_INICIO).
#    Los eventos DESTINO_* (v3) no son tareas del BPMN sino la decisión "¿Riesgo post-quirúrgico?": se excluyen de la
#    proyección y la decisión se comprueba aparte como regla (destino UCI => ingreso en UCI).
log = pd.read_parquet("eventlog_pq.parquet")
proj = log[(log["activity"] != "QUIROFANO_FIN") & (~log["activity"].str.startswith("DESTINO_"))].copy()
proj["activity"] = proj["activity"].replace({"QUIROFANO_INICIO": "CIRUGIA"})
df = pm4py.format_dataframe(proj.rename(columns={"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"}),
                            case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
df["case:concept:name"] = df["case:concept:name"].astype(str)

# 2. BPMN -> nombres de actividad del log
MAPA = {"Ingresar paciente en hospital": "INGRESO",
        "Realizar interconsulta preferente a UCI": "INTERCONSULTA_UCI",
        "Realizar cirugía programada": "CIRUGIA",
        "Ingresar paciente en UCI": "INGRESO_UCI",
        "Trasladar post-quirúgicamente paciente a planta": "TRASLADO_PLANTA",
        "Realizar interconsulta normal especialista": "INTERCONSULTA_ESPECIALISTA",
        "Realizar traslado a otro hospital": "TRASLADO_OTRO_HOSPITAL",
        "Recibir alta hospitalaria por exitus": "EXITUS",
        "Recibir alta hospitalaria": "ALTA"}
bpmn = pm4py.read_bpmn("mapaHospitalizacionProgramadoQuirurgico.bpmn")
from pm4py.objects.bpmn.obj import BPMN
for n in list(bpmn.get_nodes()):
    if isinstance(n, BPMN.Task):
        if n.get_name() in MAPA:
            n.set_name(MAPA[n.get_name()])
        elif n.get_name().startswith("Ingresar  paciente en UCI si hay"):
            bpmn.remove_node(n)   # business rule task aislada (anotación), no forma parte del flujo
net, im, fm = bpmn_converter.apply(bpmn)
pm4py.save_vis_petri_net(net, im, fm, os.path.join(OUT, "01_bpmn_como_petri.png"))
pm4py.write_pnml(net, im, fm, os.path.join(OUT, "bpmn_referencia.pnml"))

res = []
p = res.append
p("Conformance del event log v3 (1 evento de cirugía = CIRUGIA) contra mapaHospitalizacionProgramadoQuirurgico.bpmn")
p(f"Red de Petri derivada del BPMN: {len(net.places)} lugares, {len(net.transitions)} transiciones ({sum(1 for t in net.transitions if t.label)} visibles)")
p(f"Log proyectado: {len(df):,} eventos, {df['case:concept:name'].nunique():,} casos")

# 3. token replay
tr = pm4py.conformance_diagnostics_token_based_replay(df, net, im, fm)
fit_tr = pm4py.fitness_token_based_replay(df, net, im, fm)
prec_tr = pm4py.precision_token_based_replay(df, net, im, fm)
p(f"\nToken replay: fitness media {fit_tr['average_trace_fitness']:.3f}, log fitness {fit_tr['log_fitness']:.3f}, "
  f"trazas conformes {fit_tr['percentage_of_fitting_traces']:.1f} %; precisión {prec_tr:.3f}")

# 4. alineaciones (log pequeño: cabe entero)
al = pm4py.conformance_diagnostics_alignments(df, net, im, fm)
fit_al = pm4py.fitness_alignments(df, net, im, fm)
prec_al = pm4py.precision_alignments(df, net, im, fm)
p(f"Alineaciones: fitness media {fit_al['average_trace_fitness']:.3f}, log fitness {fit_al['log_fitness']:.3f}, "
  f"trazas conformes {fit_al['percentage_of_fitting_traces']:.1f} %; precisión {prec_al:.3f}")

# 5. desviaciones: movimientos sólo en log / sólo en modelo
from collections import Counter
solo_log, solo_modelo = Counter(), Counter()
for a in al:
    for (l, m) in a["alignment"]:
        if l == ">>" and m is not None and m != ">>":
            solo_modelo[m] += 1
        elif m == ">>" and l is not None and l != ">>":
            solo_log[l] += 1
p("\nMovimientos sólo en el modelo (actividad que el BPMN exige y el log no registra):")
for k, v in solo_modelo.most_common():
    p(f"  {k:28s} {v:6,d}")
p("Movimientos sólo en el log (evento observado que el BPMN no admite en ese punto):")
for k, v in solo_log.most_common():
    p(f"  {k:28s} {v:6,d}")

# 6. conformidad por variante
cases = df["case:concept:name"].unique()
fit_por_caso = pd.Series([a["fitness"] for a in al], index=cases)
var = df.groupby("case:concept:name")["concept:name"].agg(">".join)
tab = pd.DataFrame({"variante": var, "fitness": fit_por_caso}).groupby("variante").agg(casos=("fitness", "size"), fitness=("fitness", "mean")) \
        .sort_values("casos", ascending=False)
tab.to_csv(os.path.join(OUT, "variantes_fitness.csv"), sep=";")
p("\nVariantes más frecuentes y su fitness (alineaciones):")
p(tab.head(12).round(3).to_string())

# 6b. Variante del modelo con "Trasladar a planta" opcional (el traslado a planta tras despertar no se registra en traslados)
import copy
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils
net2 = copy.deepcopy(net)
t_pl = [t for t in net2.transitions if t.label == "TRASLADO_PLANTA"][0]
skip = PetriNet.Transition("skip_traslado_planta", None)
net2.transitions.add(skip)
for a in list(t_pl.in_arcs):
    petri_utils.add_arc_from_to(a.source, skip, net2)
for a in list(t_pl.out_arcs):
    petri_utils.add_arc_from_to(skip, a.target, net2)
im2 = pm4py.objects.petri_net.obj.Marking({[p_ for p_ in net2.places if p_.name == list(im.keys())[0].name][0]: 1})
fm2 = pm4py.objects.petri_net.obj.Marking({[p_ for p_ in net2.places if p_.name == list(fm.keys())[0].name][0]: 1})
pm4py.write_pnml(net2, im2, fm2, os.path.join(OUT, "bpmn_referencia_traslado_opcional.pnml"))
al2 = pm4py.conformance_diagnostics_alignments(df, net2, im2, fm2)
fit_al2 = pm4py.fitness_alignments(df, net2, im2, fm2)
prec_al2 = pm4py.precision_alignments(df, net2, im2, fm2)
p(f"\nBPMN con 'Trasladar a planta' opcional — alineaciones: fitness media {fit_al2['average_trace_fitness']:.3f}, "
  f"log fitness {fit_al2['log_fitness']:.3f}, trazas conformes {fit_al2['percentage_of_fitting_traces']:.1f} %; precisión {prec_al2:.3f}")
sl2, sm2 = Counter(), Counter()
for a in al2:
    for (l, m) in a["alignment"]:
        if l == ">>" and m not in (None, ">>"):
            sm2[m] += 1
        elif m == ">>" and l not in (None, ">>"):
            sl2[l] += 1
p("  sólo en el modelo: " + ", ".join(f"{k} {v}" for k, v in sm2.most_common()))
p("  sólo en el log: " + ", ".join(f"{k} {v}" for k, v in sl2.most_common()))
fit2_caso = pd.Series([a["fitness"] for a in al2], index=cases)
tab2 = pd.DataFrame({"variante": var, "fitness": fit2_caso}).groupby("variante").agg(casos=("fitness", "size"), fitness=("fitness", "mean")).sort_values("casos", ascending=False)
tab2.to_csv(os.path.join(OUT, "variantes_fitness_traslado_opcional.csv"), sep=";")
p(tab2.head(12).round(3).to_string())

# 7. DFG del log completo (con QUIROFANO_INICIO/FIN) y del proyectado
full = pm4py.format_dataframe(log.rename(columns={"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"}),
                              case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
full["case:concept:name"] = full["case:concept:name"].astype(str)
dfg, sa, ea = pm4py.discover_dfg(full)
pm4py.save_vis_dfg(dfg, sa, ea, os.path.join(OUT, "02_dfg_log_v3.png"))
pt = pm4py.discover_process_tree_inductive(full, noise_threshold=0.2)
n2, i2, f2 = pm4py.convert_to_petri_net(pt)
pm4py.save_vis_bpmn(pm4py.convert_to_bpmn(pt), os.path.join(OUT, "03_inductive_bpmn_log_v3.png"))
fit2 = pm4py.fitness_alignments(full, n2, i2, f2); prec2 = pm4py.precision_alignments(full, n2, i2, f2)
p(f"\nModelo descubierto (Inductive Miner infrequent 0,2) sobre el log completo: fitness {fit2['log_fitness']:.3f}, precisión {prec2:.3f}")
p(f"DFG del log v3: {len(dfg)} aristas")
# 8. Decisión del BPMN "¿Riesgo post-quirúrgico?" a partir de DESTINO_* (v3)
dd = log[log["activity"].str.startswith("DESTINO_")]
uci_c = set(log.loc[log["activity"] == "INGRESO_UCI", "case_id"])
p("\nRegla de decisión 'destino previsto' -> ingreso real en UCI (por intervención):")
for a, g in dd.groupby("activity"):
    n = len(g); k = g["case_id"].isin(uci_c).sum()
    p(f"  {a:22s} intervenciones {n:5,d}  con INGRESO_UCI en el episodio {k:4,d} ({k/n*100:5.1f} %)")
txt = "\n".join(res)
open(os.path.join(OUT, "conformance_bpmn.txt"), "w", encoding="utf-8").write(txt)
print(txt)
