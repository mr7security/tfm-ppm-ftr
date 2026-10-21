#!/usr/bin/env python3
"""
make_inputs_v3.py — Ficheros de entrada del event log v3 (ingreso programado quirúrgico) para cada herramienta.
El log ya es pequeño (7.780 eventos, 14 actividades): no hace falta vista de alto nivel; se usa completo.

  disco\disco_eventlog_pq.csv          CSV coma, Case ID / Activity / Timestamp + atributos (Disco también abre el XES)
  disco\eventlog_pq.xes.gz             copia del XES
  prom\eventlog_pq_prom.xes.gz         XES sin atributos numéricos vacíos (XESLite) — log completo
  prom\eventlog_pq_uci_prom.xes.gz     subconjunto de casos con ingreso en UCI
  celonis\celonis_eventlog_pq.csv/.parquet   tabla de actividades (CASE_ID, ACTIVITY, EVENTTIME, SORTING + atributos)
  celonis\celonis_casos_pq.csv/.parquet      tabla de casos
  pm4py\eventlog_pq.csv.gz             CSV genérico (;)
"""
import os, shutil, sys
import pandas as pd, numpy as np

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # EventLogV3
OUT = os.path.dirname(os.path.abspath(__file__))  # EventLogV3\\Discovery
CASE_COLS = ["case_id", "id_anonim_paciente", "edad_ingreso", "sexo", "servicio_ingreso", "cie_principal", "cie_capitulo",
             "n_diagnosticos", "proceso_paliativo", "cirugia_realizada", "tipo_intervencion", "categoria", "anestesia",
             "destino_postoperatorio", "destino_postop_grupo", "duracion_cirugia_min", "estancia_dias", "motivo_alta",
             "label_ingreso_uci", "label_uci_no_planificada", "label_complicacion_postop", "label_ftr_compuesto",
             "label_exitus_intrahosp", "label_estancia_prolongada", "label_reingreso_7d"]
EV_COLS = ["case_id", "activity", "timestamp", "event_id", "fase", "servicio", "prioridad", "servicio_solicitante",
           "servicio_destino", "tipo_intervencion", "categoria", "abordaje", "anestesia", "destino_postoperatorio",
           "destino_grupo", "cama", "num_traslado", "motivo_alta"]

log = pd.read_parquet(os.path.join(SRC, "eventlog_pq.parquet"))[EV_COLS]
cs = pd.read_parquet(os.path.join(SRC, "eventlog_pq_casos.parquet"))[CASE_COLS]
for d in ["disco", "prom", "celonis", "pm4py"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

full = log.merge(cs, on="case_id", how="left", suffixes=("", "_caso"))
full["case_id"] = full["case_id"].astype(str)

# pm4py / genérico
full.to_csv(os.path.join(OUT, "pm4py", "eventlog_pq.csv.gz"), sep=";", index=False, date_format="%Y-%m-%d %H:%M:%S", encoding="utf-8")

# Disco
d = full.rename(columns={"case_id": "Case ID", "activity": "Activity", "timestamp": "Timestamp"})
d.to_csv(os.path.join(OUT, "disco", "disco_eventlog_pq.csv"), sep=",", index=False, date_format="%Y-%m-%d %H:%M:%S", encoding="utf-8")
shutil.copy(os.path.join(SRC, "eventlog_pq.xes.gz"), os.path.join(OUT, "disco", "eventlog_pq.xes.gz"))

# Celonis
c = log.copy(); c["case_id"] = c["case_id"].astype(str)
c = c.rename(columns={"case_id": "CASE_ID", "activity": "ACTIVITY", "timestamp": "EVENTTIME"})
c = c.sort_values(["CASE_ID", "EVENTTIME", "event_id"])
c["SORTING"] = c.groupby("CASE_ID").cumcount() + 1
c["EVENTTIME"] = c["EVENTTIME"].astype("datetime64[ms]")
c.to_csv(os.path.join(OUT, "celonis", "celonis_eventlog_pq.csv"), sep=",", index=False, date_format="%Y-%m-%dT%H:%M:%S", encoding="utf-8")
c.to_parquet(os.path.join(OUT, "celonis", "celonis_eventlog_pq.parquet"), index=False)
cc = cs.rename(columns={"case_id": "CASE_ID"}).copy(); cc["CASE_ID"] = cc["CASE_ID"].astype(str)
cc.to_csv(os.path.join(OUT, "celonis", "celonis_casos_pq.csv"), sep=",", index=False, encoding="utf-8")
cc.to_parquet(os.path.join(OUT, "celonis", "celonis_casos_pq.parquet"), index=False)

# ProM: XES sólo con atributos de texto (sin nan) + etiquetas como enteros
import pm4py
x = full.rename(columns={"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"})
keep_ev = ["case:concept:name", "concept:name", "time:timestamp", "fase", "servicio", "prioridad", "servicio_destino",
           "tipo_intervencion", "destino_postoperatorio", "destino_grupo", "motivo_alta"]
keep_case = ["edad_ingreso", "sexo", "servicio_ingreso", "cie_capitulo", "cirugia_realizada", "destino_postop_grupo", "label_ingreso_uci",
             "label_uci_no_planificada", "label_complicacion_postop", "label_exitus_intrahosp"]
x = x[keep_ev + keep_case].rename(columns={k: "case:" + k for k in keep_case})
for col in x.columns:
    if col == "time:timestamp":
        continue
    if pd.api.types.is_numeric_dtype(x[col]):
        x[col] = x[col].fillna(-1).astype(int)
    else:
        x[col] = x[col].fillna("").astype(str)
x = pm4py.format_dataframe(x, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
pm4py.write_xes(x, os.path.join(OUT, "prom", "eventlog_pq_prom.xes.gz"))
uci = x[x["case:label_ingreso_uci"] == 1]
pm4py.write_xes(uci, os.path.join(OUT, "prom", "eventlog_pq_uci_prom.xes.gz"))
print(f"Entradas generadas: {len(full):,} eventos, {full['case_id'].nunique():,} casos; subconjunto UCI {uci['case:concept:name'].nunique()} casos")
