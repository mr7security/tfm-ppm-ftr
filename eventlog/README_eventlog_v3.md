# Event log v3 — Proceso de ingreso programado quirúrgico (destino postoperatorio)

*Alcance acordado en la reunión de dirección del 17/09/2026. Sustituye al event log v1 (proceso hospitalario completo) como base del TFM.*

**v3 (28/09/2026) — cambios respecto a v2: destino postoperatorio.** El destino al salir de quirófano pasa a ser una actividad propia, tomada de `ftr_hosp_int_quirurgicas.codigo_destino` (catálogo `destinos_postoperatorio_quirofano`): Planta y Despertar se consideran lo mismo (DESTINO_PLANTA); UCI (DESTINO_UCI) y Reanimación (DESTINO_REANIMACION) son destinos propios; Domicilio, UCSI, Urgencias y Otros se agrupan en DESTINO_OTROS. `ftr_hosp_traslados` se mantiene como el movimiento **real** dentro del hospital (INGRESO_UCI / TRASLADO_PLANTA), siempre con su hora real, incluidos los pocos traslados registrados antes o durante la cirugía. El log v2 (19/09) sigue en `EventLogV2/` sin cambios, junto con su discovery y la prueba PPM.

Generado con `build_event_log_v3.py` a partir de los CSV de `ftr_cige_v01` (versión de datos 23-24/03/2026, copia 12/09/2026):

```
python build_event_log_v3.py --csv ..\BBDD\csv --out .
```

Produce `eventlog_pq.csv` / `.parquet` (eventos), `eventlog_pq_casos.csv` / `.parquet` (una fila por caso con atributos y etiquetas), `eventlog_pq.xes.gz` (pm4py, válido también para Disco y ProM: sin valores `nan` de texto) y `resumen_v3.txt`. El modelo de referencia es `mapaHospitalizacionProgramadoQuirurgico.bpmn` (Camunda) con su imagen `.png`.

## Población y tablas

| Tabla | Uso |
|---|---|
| `ftr_hosp` (principal) | Filtro `tipo_ingreso = P` (programado, código 2 en la BBDD) y `tipoproceso = Q` (quirúrgico, código 2). Eventos INGRESO y ALTA / EXITUS / TRASLADO_OTRO_HOSPITAL; atributos de caso. |
| `ftr_hosp_interconsultas` | `cod_servicio_destino = 1` (UCI) → INTERCONSULTA_UCI; resto → INTERCONSULTA_ESPECIALISTA. |
| `ftr_hosp_int_quirurgicas` | QUIROFANO_INICIO / QUIROFANO_FIN con tipo, categoría, abordaje y anestesia. `codigo_destino` → DESTINO_PLANTA (1 Planta, 2 Despertar) / DESTINO_UCI (7) / DESTINO_REANIMACION (9) / DESTINO_OTROS (3 Domicilio, 4 UCSI, 5 Otros, 8 Urgencias), con timestamp = `fecha_fin` (la tabla no guarda hora propia del destino). Sin dato de destino → sin evento (6 intervenciones). |
| `ftr_hosp_traslados` | Movimiento real dentro del hospital. `num_traslado_asc > 1` (el 1 es la cama inicial): `codigo_servicio = 1` → INGRESO_UCI; resto → TRASLADO_PLANTA. Hora real de `fecha_desde`, aunque caiga antes o durante la cirugía. |
| `ftr_hosp_codificacion`, catálogos `*_pk`, `codigo_servicio` | Sólo para atributos de caso (CIE-10 principal) y descripciones. |

Quedan fuera del alcance: urgencias, constantes, laboratorio y valoración de enfermería.

**Resultado: 1.475 episodios (15,9 % de los 9.304), 1.308 pacientes, 7.780 eventos, 14 actividades, 97 variantes.** Mediana 5 eventos por caso (máx. 32). Periodo 01/03/2023 → 10/04/2024. Sólo 3 eventos descartados por caer fuera de `[ingreso − 1 d, alta + 1 d]` (una intervención completa con su destino).

## Actividades y correspondencia con el BPMN

| Actividad | Eventos | Elemento del BPMN |
|---|---|---|
| INGRESO | 1.475 | Ingresar paciente en hospital |
| INTERCONSULTA_UCI | 18 | Realizar interconsulta preferente a UCI |
| INTERCONSULTA_ESPECIALISTA | 452 | Realizar interconsulta normal especialista |
| QUIROFANO_INICIO / QUIROFANO_FIN | 1.396 / 1.396 | Realizar cirugía programada |
| DESTINO_PLANTA | 1.212 | Decisión "¿Riesgo post-quirúrgico?" → No (planta / despertar) |
| DESTINO_UCI | 124 | Decisión "¿Riesgo post-quirúrgico?" → Sí (reserva de cama UCI) |
| DESTINO_REANIMACION | 53 | Destino propio (no está en el BPMN actual) |
| DESTINO_OTROS | 7 | Destino propio (en este subconjunto sólo UCSI) |
| INGRESO_UCI | 148 | Ingresar paciente en UCI |
| TRASLADO_PLANTA | 24 | Trasladar post-quirúrgicamente paciente a planta |
| TRASLADO_OTRO_HOSPITAL | 1 | Realizar traslado a otro hospital |
| EXITUS | 9 | Recibir alta hospitalaria por exitus |
| ALTA | 1.465 | Recibir alta hospitalaria |

Cada evento lleva el atributo `fase` (`preoperatorio`, `cirugia`, `postoperatorio`, `sin_cirugia`) calculado respecto al primer QUIROFANO_INICIO del caso; sirve para cortar prefijos y para evaluar las decisiones del BPMN. Los eventos DESTINO_* se marcan como `cirugia` (se deciden en quirófano) y llevan `destino_grupo` y `destino_postoperatorio` (valor original del catálogo). A igualdad de minuto el orden es QUIROFANO_FIN → DESTINO_* → INGRESO_UCI.

Variantes principales: INGRESO → QUIROFANO → DESTINO_PLANTA → ALTA (916 casos, 62 %); con interconsulta a especialista antes, durante o después de la cirugía (100 + 50 + 36); INGRESO → ALTA sin cirugía (100); DESTINO_REANIMACION → ALTA (44); DESTINO_UCI → INGRESO_UCI → ALTA (36) y QUIROFANO_INICIO → INGRESO_UCI → QUIROFANO_FIN → DESTINO_UCI → ALTA (21, traslado a UCI registrado durante la intervención).

### Destino previsto frente a ingreso real en UCI (casos, primera intervención)

| Destino previsto | Casos | Ingresan en UCI |
|---|---|---|
| PLANTA (planta + despertar) | 1.165 | 11 |
| UCI | 110 | 110 |
| REANIMACION | 52 | 2 |
| OTROS (UCSI) | 6 | 1 |
| Sin cirugía / sin dato | 142 | 7 |
| **Total** | **1.475** | **131** |

Los 21 ingresos en UCI sin destino UCI previsto (11 + 2 + 1 + 7) son `label_uci_no_planificada`.

## Hallazgos que afectan al BPMN

1. **La "reserva de cama UCI" no está en las interconsultas sino en quirófano.** No hay ninguna interconsulta preoperatoria a UCI (0 casos); las 18 INTERCONSULTA_UCI son postoperatorias (12 casos, prioridades preferente 8 / urgente 6 / normal 4). En cambio `destino_postoperatorio = UCI` aparece en 110 casos y **los 110 ingresan en UCI**: esa columna es la evidencia de la decisión "¿Riesgo post-quirúrgico?" + reserva de cama. La regla de negocio del modelo se cumple si se lee como "destino postoperatorio UCI ⇒ ingreso en UCI" (110/110).
2. **Ingreso en UCI no planificado**: 21 casos entran en UCI sin reserva previa (destino postoperatorio distinto de UCI). Es la complicación postquirúrgica "sorpresa" que el modelo sitúa en la puerta "¿Complicación post-quirúrgica?".
3. **Exitus casi inexistente**: 9 exitus intrahospitalarios (0,6 %), 6 de ellos tras pasar por UCI; 26 a 90 días. Como target de clasificación es inviable en este subconjunto.
4. **Orden temporal UCI/quirófano**: 34 INGRESO_UCI tienen timestamp entre QUIROFANO_INICIO y QUIROFANO_FIN (el traslado se registra al cerrar quirófano), 13 son preoperatorios y 9 corresponden a casos sin intervención registrada. Se mantiene el orden real; en discovery aparecerá QUIROFANO_INICIO → INGRESO_UCI → QUIROFANO_FIN como variante.
5. **142 casos sin intervención registrada** (9,6 %) pese a ser ingresos programados quirúrgicos: coinciden con la rama "No cirugía" del BPMN (100 son INGRESO → ALTA directos).
6. El traslado a otro hospital es anecdótico (1 caso).

## Atributos de caso (`eventlog_pq_casos`)

Demografía y episodio: `edad_ingreso`, `sexo`, `diag_ing`, `servicio_ingreso`, `servicio_alta`, `proceso_paliativo`, `estancia_dias`, `cie_principal`, `cie_principal_desc`, `cie_capitulo`, `n_diagnosticos`, `motivo_alta`, `reingreso7d_sn`.

Cirugía: `cirugia_realizada`, `n_intervenciones`, `tipo_intervencion`, `categoria`, `abordaje`, `anestesia`, `destino_postoperatorio` (valor original), `destino_postop_grupo` (PLANTA / UCI / REANIMACION / OTROS), `servicio_cirugia`, `duracion_cirugia_min`, `dias_ingreso_a_cirugia`.

Interconsultas y UCI: `interconsulta_uci_preop`, `interconsulta_uci_postop`, `interconsulta_especialista_postop`, `n_interconsultas`, `ingreso_uci`, `ingreso_uci_postop`, `destino_postop_uci`, `destino_postop_reanimacion`, `n_traslados_planta`.

Traza: `n_eventos`, `n_actividades_distintas`, `primer_evento`, `ultimo_evento`.

### Etiquetas candidatas

| Etiqueta | Definición | Positivos |
|---|---|---|
| `label_exitus_intrahosp` | motivo_alta = EXITUS | 9 (0,6 %) |
| `label_exitus_30d` / `_90d` | exitus intrahospitalario o `fecha_exitus` ≤ alta + 30 / 90 d | 10 (0,7 %) / 26 (1,8 %) |
| `label_ingreso_uci` | algún INGRESO_UCI en el episodio | 131 (8,9 %) |
| `label_uci_no_planificada` | INGRESO_UCI sin destino postoperatorio UCI ni interconsulta UCI preoperatoria | 21 (1,4 %) |
| `label_traslado_otro_hosp` | motivo_alta = traslado a otro hospital | 1 (0,1 %) |
| `label_complicacion_postop` | ingreso UCI postoperatorio, interconsulta a especialista postoperatoria, exitus o traslado a otro hospital | 240 (16,3 %) |
| `label_ftr_compuesto` | exitus intrahospitalario o ingreso UCI o traslado a otro hospital | 135 (9,2 %) |
| `label_reingreso_7d` | reingreso7d_sn = S | 23 (1,6 %) |
| `label_estancia_prolongada` | estancia > percentil 90 (≈ 8,7 días) | 148 (10,0 %) |

## Avisos para PPM

- **Leakage**: `servicio_alta`, `motivo_alta`, `estancia_dias`, `cama_al_alta`, `n_eventos` y los eventos INGRESO_UCI / EXITUS / ALTA revelan el resultado; hay que truncar la traza antes del evento que define la etiqueta y excluir esos atributos.
- **DESTINO_UCI y `label_ingreso_uci`**: DESTINO_UCI implica ingreso en UCI en el 100 % de los casos. Si el prefijo incluye el evento DESTINO_* (o `destino_postop_grupo`), predecir `label_ingreso_uci` es trivial para la parte planificada; para medir capacidad predictiva real conviene cortar en QUIROFANO_FIN (sin destino) o usar `label_uci_no_planificada`. En `label_uci_no_planificada` el destino previsto no es fuga: es información legítima disponible al salir de quirófano.
- **Código PPM**: `EventLogV2/PPM/ppm_data.py` (o su copia en `EventLogV3/PPM/`) tiene la lista de actividades fija (`ACTS`); hay que añadir las cuatro DESTINO_* antes de volver a ejecutar los experimentos.
- Con trazas de 4 eventos de mediana, la señal predictiva estará sobre todo en los **atributos** (edad, CIE, servicio, tipo/duración de cirugía, anestesia, interconsultas previas) más que en la secuencia de control; conviene tenerlo en cuenta al comparar métodos model-based frente a ML.
- Split por paciente (`id_anonim_paciente`): 1.308 pacientes para 1.475 episodios.
- Formato: CSV con `;` y UTF-8, timestamps `YYYY-MM-DD HH:MM:SS`; parquet con tipos nativos; XES con `pm4py.read_xes('eventlog_pq.xes.gz')`.
