#!/usr/bin/env python3
"""
build_event_log_v3.py — Event log del PROCESO DE INGRESO PROGRAMADO QUIRÚRGICO
(alcance acordado en la reunión de dirección del 17/09/2026).

Población : episodios de ftr_hosp con tipo_ingreso = P (programado) y tipoproceso = Q (quirúrgico).
Tablas    : ftr_hosp, ftr_hosp_interconsultas, ftr_hosp_int_quirurgicas, ftr_hosp_traslados
            (+ catálogos codigo_servicio, motivo_alta_pk, destinos_postoperatorio_quirofano,
            tipo_anestesia_pk y ftr_hosp_codificacion sólo como atributo de caso CIE-10).
Caso      : episodio de hospitalización (id_anonim_episodio).
Actividades (alineadas con mapaHospitalizacionProgramadoQuirurgico.bpmn):
    INGRESO                     ftr_hosp.fecha_ingreso            "Ingresar paciente en hospital"
    INTERCONSULTA_UCI           interconsultas, destino = UCI (1)  "Realizar interconsulta preferente a UCI"
    INTERCONSULTA_ESPECIALISTA  interconsultas, destino != UCI     "Realizar interconsulta normal especialista"
    QUIROFANO_INICIO / _FIN     int_quirurgicas fecha_inicio/fin   "Realizar cirugía programada"
    DESTINO_PLANTA              int_quirurgicas.codigo_destino 1 PLANTA / 2 DESPERTAR   (timestamp = fecha_fin)
    DESTINO_UCI                 int_quirurgicas.codigo_destino 7 UCI                    (timestamp = fecha_fin)
    DESTINO_REANIMACION         int_quirurgicas.codigo_destino 9 REANIMACIÓN            (timestamp = fecha_fin)
    DESTINO_OTROS               int_quirurgicas.codigo_destino 3 DOMICILIO / 4 UCSI / 5 OTROS / 8 URGENCIAS
                                -> destino postoperatorio PREVISTO al salir de quirófano (decisión "¿Riesgo post-quirúrgico?")
    INGRESO_UCI                 traslados, codigo_servicio = 1     "Ingresar paciente en UCI"
    TRASLADO_PLANTA             traslados num>1, servicio != UCI   "Trasladar post-quirúrgicamente paciente a planta"
                                -> ftr_hosp_traslados = movimiento REAL dentro del hospital, con su hora real
                                   (también los pocos registrados antes o durante la cirugía)

Cambios 28/09/2026: destino postoperatorio como actividad propia a partir de codigo_destino (Planta y Despertar
se tratan como lo mismo; Reanimación es destino propio; Domicilio, UCSI, Urgencias y Otros se agrupan en OTROS).
    TRASLADO_OTRO_HOSPITAL      ftr_hosp.fecha_alta, motivo 3      "Realizar traslado a otro hospital"
    EXITUS                      ftr_hosp.fecha_alta, motivo 4      "Recibir alta hospitalaria por exitus"
    ALTA                        ftr_hosp.fecha_alta, resto         "Recibir alta hospitalaria"

Uso:  python build_event_log_v3.py --csv ..\BBDD\csv --out .
Salida: eventlog_pq.csv/.parquet (eventos), eventlog_pq_casos.csv/.parquet (1 fila por caso),
        eventlog_pq.xes.gz (pm4py) y resumen_v3.txt
"""
import argparse, os, sys
import numpy as np
import pandas as pd

PRE_DIAS = 1    # margen antes del ingreso (traslado/cama registrados la noche anterior)
POST_DIAS = 1   # margen después del alta
UCI = 1         # codigo_servicio de la Unidad de Cuidados Intensivos
DESTINO_GRUPO = {1: "PLANTA", 2: "PLANTA", 7: "UCI", 9: "REANIMACION", 3: "OTROS", 4: "OTROS", 5: "OTROS", 8: "OTROS"}
ACTIVIDADES = ["INGRESO", "INTERCONSULTA_UCI", "INTERCONSULTA_ESPECIALISTA", "QUIROFANO_INICIO", "QUIROFANO_FIN",
               "DESTINO_PLANTA", "DESTINO_UCI", "DESTINO_REANIMACION", "DESTINO_OTROS", "INGRESO_UCI", "TRASLADO_PLANTA", "TRASLADO_OTRO_HOSPITAL", "EXITUS", "ALTA"]


def read(csvdir, name, **kw):
    return pd.read_csv(os.path.join(csvdir, name + ".csv"), sep=";", low_memory=False, **kw)


def to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def build(csvdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    hosp = read(csvdir, "ftr_hosp")
    for c in ["fecha_ingreso", "fecha_alta", "fecha_exitus"]:
        hosp[c] = to_dt(hosp[c])
    n_total = len(hosp)
    # ---------------- filtro de población: ingreso programado (P) y proceso quirúrgico (Q)
    hosp = hosp[(hosp["tipo_ingreso"].str.strip() == "P") & (hosp["tipoproceso"].str.strip() == "Q")].copy()
    hosp["id_anonim_episodio"] = hosp["id_anonim_episodio"].astype("int64")
    episodios = set(hosp["id_anonim_episodio"])

    servicios = read(csvdir, "codigo_servicio").set_index("codigo_servicio")["servicio"].str.strip()
    motivos = read(csvdir, "motivo_alta_pk").set_index("motivo_alta_pk")["motivo_alta_desc"].str.strip()
    anestesia = read(csvdir, "tipo_anestesia_pk").set_index("tipo_anestesia_pk")["tipo_anesdesc"].str.strip()
    destinos = read(csvdir, "destinos_postoperatorio_quirofano").set_index("codigo_destino")["qrf_dest_desc"].str.strip()

    events = []

    def add(df, activity, ts_col, attrs=None):
        df = df[df["id_anonim_episodio"].isin(episodios)]
        if len(df) == 0:
            return
        d = pd.DataFrame({"case_id": df["id_anonim_episodio"].astype("int64").values,
                          "activity": activity if isinstance(activity, str) else activity.loc[df.index].values,
                          "timestamp": to_dt(df[ts_col]).values})
        if attrs:
            for k, v in attrs.items():
                d[k] = v.loc[df.index].values if hasattr(v, "loc") else v
        events.append(d.dropna(subset=["timestamp"]))

    # ---------------- 1. Ingreso
    add(hosp, "INGRESO", "fecha_ingreso",
        {"servicio": hosp["cod_serv_ingreso"].map(servicios), "cama": hosp["primera_cama_ingreso_his"].str.strip()})

    # ---------------- 2. Interconsultas: a UCI (preferente / cualquier prioridad) vs. a otro especialista
    ic = read(csvdir, "ftr_hosp_interconsultas")
    ic["id_anonim_episodio"] = ic["id_anonim_episodio"].astype("int64")
    act_ic = np.where(ic["cod_servicio_destino"] == UCI, "INTERCONSULTA_UCI", "INTERCONSULTA_ESPECIALISTA")
    add(ic, pd.Series(act_ic, index=ic.index), "fecha_interconsulta",
        {"prioridad": ic["prioridad"].str.strip(),
         "servicio_solicitante": ic["cod_servicio_solicitante"].map(servicios),
         "servicio_destino": ic["cod_servicio_destino"].map(servicios)})

    # ---------------- 3. Intervención quirúrgica
    iq = read(csvdir, "ftr_hosp_int_quirurgicas")
    iq["id_anonim_episodio"] = iq["id_anonim_episodio"].astype("int64")
    common = {"servicio": iq["codigo_servicio"].map(servicios), "tipo_intervencion": iq["tipo_intervencion"].str.strip(),
              "categoria": iq["categoria_intervencion"].str.strip(), "abordaje": iq["abordaje_quirurgico"].str.strip(),
              "anestesia": iq["codigo_tipo_anestesia"].map(anestesia)}
    add(iq, "QUIROFANO_INICIO", "fecha_inicio", common)
    add(iq, "QUIROFANO_FIN", "fecha_fin", {**common, "destino_postoperatorio": iq["codigo_destino"].map(destinos)})
    # destino postoperatorio previsto: evento propio al cierre de quirófano (sin dato de destino -> sin evento)
    grp = iq["codigo_destino"].map(DESTINO_GRUPO)
    iqd = iq[grp.notna()]
    add(iqd, ("DESTINO_" + grp[grp.notna()]), "fecha_fin",
        {"servicio": iqd["codigo_servicio"].map(servicios), "destino_grupo": grp[grp.notna()],
         "destino_postoperatorio": iqd["codigo_destino"].map(destinos)})

    # ---------------- 4. Traslados: ingreso en UCI / traslado a planta (num_traslado_asc = 1 es la cama inicial, no es traslado)
    tr = read(csvdir, "ftr_hosp_traslados")
    tr["id_anonim_episodio"] = tr["id_anonim_episodio"].astype("int64")
    tr = tr[tr["num_traslado_asc"] > 1]
    act_tr = np.where(tr["codigo_servicio"] == UCI, "INGRESO_UCI", "TRASLADO_PLANTA")
    add(tr, pd.Series(act_tr, index=tr.index), "fecha_desde",
        {"servicio": tr["codigo_servicio"].map(servicios), "cama": tr["cod_cama"].str.strip(), "num_traslado": tr["num_traslado_asc"]})

    # ---------------- 5. Fin del episodio
    fin_act = np.select([hosp["motivo_alta_pk"] == 4, hosp["motivo_alta_pk"] == 3], ["EXITUS", "TRASLADO_OTRO_HOSPITAL"], "ALTA")
    add(hosp, pd.Series(fin_act, index=hosp.index), "fecha_alta",
        {"servicio": hosp["cod_serv_alta"].map(servicios), "motivo_alta": hosp["motivo_alta_pk"].map(motivos),
         "cama": hosp["cama_al_alta"].str.strip()})

    # ---------------- ensamblado
    log = pd.concat(events, ignore_index=True, sort=False)
    w = hosp.set_index("id_anonim_episodio")[["fecha_ingreso", "fecha_alta"]]
    ini = log["case_id"].map(w["fecha_ingreso"]) - pd.Timedelta(days=PRE_DIAS)
    fin = log["case_id"].map(w["fecha_alta"]) + pd.Timedelta(days=POST_DIAS)
    fuera = ~log["timestamp"].between(ini, fin)
    descartados = log[fuera].groupby("activity").size()
    log = log[~fuera].copy()
    orden = {a: i for i, a in enumerate(ACTIVIDADES)}
    log["_o"] = log["activity"].map(orden)
    log = log.sort_values(["case_id", "timestamp", "_o"]).drop(columns="_o").reset_index(drop=True)
    log.insert(3, "event_id", np.arange(1, len(log) + 1))
    # fase respecto a la cirugía (útil para prefijos y para la regla de negocio del BPMN)
    t_qx = log[log["activity"] == "QUIROFANO_INICIO"].groupby("case_id")["timestamp"].min()
    tq = log["case_id"].map(t_qx)
    log["fase"] = np.select([tq.isna(), log["timestamp"] < tq], ["sin_cirugia", "preoperatorio"], "postoperatorio")
    log.loc[log["activity"].isin(["QUIROFANO_INICIO", "QUIROFANO_FIN"]), "fase"] = "cirugia"
    # el destino previsto se decide en quirófano: se marca como fase cirugia (el prefijo hasta QUIROFANO_FIN no lo incluye)
    log.loc[log["activity"].str.startswith("DESTINO_"), "fase"] = "cirugia"

    # ---------------- atributos de caso
    cs = hosp.rename(columns={"id_anonim_episodio": "case_id"}).copy()
    cs["servicio_ingreso"] = cs["cod_serv_ingreso"].map(servicios)
    cs["servicio_alta"] = cs["cod_serv_alta"].map(servicios)
    cs["motivo_alta"] = cs["motivo_alta_pk"].map(motivos)
    cs["estancia_dias"] = (cs["fecha_alta"] - cs["fecha_ingreso"]).dt.total_seconds() / 86400
    for c in ["proceso_paliativo", "cama_al_alta", "primera_cama_ingreso_his", "diag_ing"]:
        cs[c] = cs[c].astype(str).str.strip()
    cod = read(csvdir, "ftr_hosp_codificacion")
    dp = cod[cod["diagnostico_principal_sn"] == "S"].sort_values("orden_diagnostico").drop_duplicates("id_anonim_episodio")
    cs = cs.merge(dp[["id_anonim_episodio", "codigo_cie", "descripcion_cie"]]
                  .rename(columns={"id_anonim_episodio": "case_id", "codigo_cie": "cie_principal", "descripcion_cie": "cie_principal_desc"}),
                  on="case_id", how="left")
    cs = cs.merge(cod.groupby("id_anonim_episodio").size().rename("n_diagnosticos").reset_index()
                  .rename(columns={"id_anonim_episodio": "case_id"}), on="case_id", how="left")
    cs["cie_capitulo"] = cs["cie_principal"].str[0]

    # cirugía: primera intervención del episodio
    q1 = (log[log["activity"] == "QUIROFANO_FIN"].sort_values("timestamp").drop_duplicates("case_id")
          [["case_id", "tipo_intervencion", "categoria", "abordaje", "anestesia", "destino_postoperatorio", "servicio"]]
          .rename(columns={"servicio": "servicio_cirugia"}))
    cs = cs.merge(q1, on="case_id", how="left")
    cs["cirugia_realizada"] = cs["case_id"].isin(q1["case_id"]).astype(int)
    cs["n_intervenciones"] = cs["case_id"].map(log[log["activity"] == "QUIROFANO_INICIO"].groupby("case_id").size()).fillna(0).astype(int)
    dur = (log[log["activity"] == "QUIROFANO_FIN"].groupby("case_id")["timestamp"].min()
           - log[log["activity"] == "QUIROFANO_INICIO"].groupby("case_id")["timestamp"].min()).dt.total_seconds() / 60
    cs["duracion_cirugia_min"] = cs["case_id"].map(dur)
    cs["dias_ingreso_a_cirugia"] = cs["case_id"].map((t_qx - w["fecha_ingreso"]).dt.total_seconds() / 86400)

    # interconsultas y UCI por fase (para la regla de negocio: interconsulta preferente a UCI -> ingreso en UCI)
    def casos(act, fase=None):
        m = log["activity"] == act
        if fase:
            m &= log["fase"] == fase
        return log.loc[m, "case_id"].unique()
    cs["interconsulta_uci_preop"] = cs["case_id"].isin(casos("INTERCONSULTA_UCI", "preoperatorio")).astype(int)
    cs["interconsulta_uci_postop"] = cs["case_id"].isin(casos("INTERCONSULTA_UCI", "postoperatorio")).astype(int)
    cs["interconsulta_especialista_postop"] = cs["case_id"].isin(casos("INTERCONSULTA_ESPECIALISTA", "postoperatorio")).astype(int)
    cs["n_interconsultas"] = cs["case_id"].map(log[log["activity"].str.startswith("INTERCONSULTA")].groupby("case_id").size()).fillna(0).astype(int)
    cs["ingreso_uci"] = cs["case_id"].isin(casos("INGRESO_UCI")).astype(int)
    cs["ingreso_uci_postop"] = cs["case_id"].isin(casos("INGRESO_UCI", "postoperatorio")).astype(int)
    cs["destino_postop_grupo"] = cs["case_id"].map(log[log["activity"].str.startswith("DESTINO_")]
                                                   .sort_values("timestamp").drop_duplicates("case_id").set_index("case_id")["destino_grupo"])
    cs["destino_postop_uci"] = (cs["destino_postoperatorio"] == "UCI").astype(int)
    cs["destino_postop_reanimacion"] = (cs["destino_postop_grupo"] == "REANIMACION").astype(int)
    cs["n_traslados_planta"] = cs["case_id"].map(log[log["activity"] == "TRASLADO_PLANTA"].groupby("case_id").size()).fillna(0).astype(int)

    # ---------------- etiquetas candidatas
    dias_exitus = (cs["fecha_exitus"] - cs["fecha_alta"].dt.normalize()).dt.days
    cs["label_exitus_intrahosp"] = (cs["motivo_alta_pk"] == 4).astype(int)
    cs["label_exitus_30d"] = ((cs["motivo_alta_pk"] == 4) | dias_exitus.between(-1, 30)).astype(int)
    cs["label_exitus_90d"] = ((cs["motivo_alta_pk"] == 4) | dias_exitus.between(-1, 90)).astype(int)
    cs["label_ingreso_uci"] = cs["ingreso_uci"]
    # UCI no anticipada: ingreso en UCI sin "reserva de cama UCI" previa. En los datos la reserva se refleja en el
    # destino postoperatorio de quirófano (codigo_destino = 7 UCI), no en una interconsulta preoperatoria a UCI (0 casos).
    cs["label_uci_no_planificada"] = ((cs["ingreso_uci"] == 1) & (cs["destino_postop_uci"] == 0)
                                      & (cs["interconsulta_uci_preop"] == 0)).astype(int)
    cs["label_traslado_otro_hosp"] = (cs["motivo_alta_pk"] == 3).astype(int)
    cs["label_complicacion_postop"] = ((cs["ingreso_uci_postop"] == 1) | (cs["interconsulta_especialista_postop"] == 1)
                                       | (cs["label_exitus_intrahosp"] == 1) | (cs["label_traslado_otro_hosp"] == 1)).astype(int)
    cs["label_ftr_compuesto"] = ((cs["label_exitus_intrahosp"] == 1) | (cs["ingreso_uci"] == 1) | (cs["label_traslado_otro_hosp"] == 1)).astype(int)
    cs["label_reingreso_7d"] = (cs["reingreso7d_sn"].str.strip() == "S").astype(int)
    cs["label_estancia_prolongada"] = (cs["estancia_dias"] > cs["estancia_dias"].quantile(0.90)).astype(int)

    g = log.groupby("case_id")
    cs = cs.merge(pd.DataFrame({"n_eventos": g.size(), "n_actividades_distintas": g["activity"].nunique(),
                                "primer_evento": g["timestamp"].min(), "ultimo_evento": g["timestamp"].max()}).reset_index(),
                  on="case_id", how="left")
    cs = cs.drop(columns=["cod_serv_ingreso", "cod_serv_alta", "motivo_alta_pk"])

    # ---------------- salida
    log.to_csv(os.path.join(outdir, "eventlog_pq.csv"), index=False, sep=";", encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
    cs.to_csv(os.path.join(outdir, "eventlog_pq_casos.csv"), index=False, sep=";", encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
    log.to_parquet(os.path.join(outdir, "eventlog_pq.parquet"), index=False)
    cs.to_parquet(os.path.join(outdir, "eventlog_pq_casos.parquet"), index=False)
    try:
        import pm4py
        x = log[["case_id", "activity", "timestamp", "fase", "servicio", "prioridad", "servicio_destino", "tipo_intervencion",
                 "categoria", "anestesia", "destino_postoperatorio", "destino_grupo", "motivo_alta"]].copy()
        ca = cs[["case_id", "edad_ingreso", "sexo", "servicio_ingreso", "cie_principal", "cie_capitulo", "proceso_paliativo",
                 "cirugia_realizada", "interconsulta_uci_preop", "estancia_dias"] + [c for c in cs.columns if c.startswith("label_")]]
        x = x.merge(ca.rename(columns={c: "case:" + c for c in ca.columns if c != "case_id"}), on="case_id")
        x = x.rename(columns={"case_id": "case:concept:name", "activity": "concept:name", "timestamp": "time:timestamp"})
        x["case:concept:name"] = x["case:concept:name"].astype(str)
        # sin 'nan' de texto para que ProM/XESLite no rechace el fichero
        for c in x.columns:
            if x[c].dtype == object:
                x[c] = x[c].where(x[c].notna(), None)
        x = pm4py.format_dataframe(x, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
        # se eliminan los atributos vacíos evento a evento (pm4py los escribiría como float "nan")
        el = pm4py.convert_to_event_log(x)
        def vacio(v):
            return v is None or (isinstance(v, float) and np.isnan(v))
        for tr in el:
            for k in [k for k, v in tr.attributes.items() if vacio(v)]:
                del tr.attributes[k]
            for e in tr:
                for k in [k for k, v in e.items() if vacio(v)]:
                    del e[k]
        pm4py.write_xes(el, os.path.join(outdir, "eventlog_pq.xes.gz"))
    except ImportError:
        print("pm4py no disponible: se omite el XES", file=sys.stderr)

    # ---------------- resumen
    L = []
    p = L.append
    p("Event log v3 — Proceso de ingreso programado quirúrgico (tipo_ingreso = P, tipoproceso = Q)")
    p(f"Población: {len(cs):,} episodios de {n_total:,} ({len(cs)/n_total*100:.1f} %), {cs['id_anonim_paciente'].nunique():,} pacientes")
    p(f"Eventos: {len(log):,}; actividades: {log['activity'].nunique()}; periodo {log['timestamp'].min()} → {log['timestamp'].max()}")
    p(f"Eventos descartados fuera de [ingreso-{PRE_DIAS}d, alta+{POST_DIAS}d]: {int(descartados.sum()):,}")
    if len(descartados):
        p(descartados.to_string())
    p("\nEventos por actividad:")
    p(log["activity"].value_counts().reindex(ACTIVIDADES).fillna(0).astype(int).to_string())
    p("\nLongitud de traza (eventos por caso):")
    p(cs["n_eventos"].describe().round(1).to_string())
    var = log.groupby("case_id")["activity"].agg(lambda s: ">".join(s))
    p(f"\nVariantes: {var.nunique():,} distintas; top 10:")
    p(var.value_counts().head(10).to_string())
    p("\nCirugía: casos con intervención %d (%.1f %%); sin intervención registrada %d" %
      (cs["cirugia_realizada"].sum(), cs["cirugia_realizada"].mean()*100, (cs["cirugia_realizada"] == 0).sum()))
    p("Tipo de intervención (primera): " + cs["tipo_intervencion"].value_counts(dropna=False).to_dict().__repr__())
    p("Destino postoperatorio (detalle, 1a intervención): " + cs["destino_postoperatorio"].value_counts(dropna=False).to_dict().__repr__())
    p("Destino postoperatorio agrupado (1a intervención): " + cs["destino_postop_grupo"].value_counts(dropna=False).to_dict().__repr__())
    p("Destino previsto agrupado x ingreso real en UCI (casos):")
    p(pd.crosstab(cs["destino_postop_grupo"].fillna("SIN_CIRUGIA/SIN_DATO"), cs["ingreso_uci"], margins=True).to_string())
    p("Interconsulta a UCI preoperatoria: %d casos; postoperatoria: %d; interconsulta especialista postop: %d" %
      (cs["interconsulta_uci_preop"].sum(), cs["interconsulta_uci_postop"].sum(), cs["interconsulta_especialista_postop"].sum()))
    p("Ingreso en UCI: %d casos (postop %d); de ellos con interconsulta UCI preop: %d" %
      (cs["ingreso_uci"].sum(), cs["ingreso_uci_postop"].sum(), ((cs["ingreso_uci"] == 1) & (cs["interconsulta_uci_preop"] == 1)).sum()))
    p("Regla BPMN 'interconsulta preferente a UCI => ingreso en UCI': cumplida en %d de %d casos con interconsulta UCI preop" %
      (((cs["interconsulta_uci_preop"] == 1) & (cs["ingreso_uci"] == 1)).sum(), cs["interconsulta_uci_preop"].sum()))
    p("Reserva de cama UCI via destino postoperatorio = UCI: %d casos; de ellos ingresan en UCI %d; ingresos en UCI sin reserva: %d" %
      (cs["destino_postop_uci"].sum(), ((cs["destino_postop_uci"] == 1) & (cs["ingreso_uci"] == 1)).sum(),
       ((cs["destino_postop_uci"] == 0) & (cs["ingreso_uci"] == 1)).sum()))
    p("Exitus intrahospitalario x ingreso UCI: " + pd.crosstab(cs["label_exitus_intrahosp"], cs["ingreso_uci"]).to_string().replace("\n", " | "))
    p("\nEtiquetas candidatas (casos positivos):")
    for c in [c for c in cs.columns if c.startswith("label_")]:
        p(f"  {c:28s} {int(cs[c].sum()):6,d}  ({cs[c].mean()*100:.1f} %)")
    p("\nEstancia (días): " + cs["estancia_dias"].describe().round(2).to_dict().__repr__())
    txt = "\n".join(L)
    with open(os.path.join(outdir, "resumen_v3.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    build(a.csv, a.out)
