"""
Experimento FINAL del TFM sobre el event log v3: las tres tareas del alcance propuesto, en una sola corrida
reproducible, con los cinco algoritmos del esquema de Zare et al. (2025): Decision Tree, Random Forest, XGBoost,
LSTM y Transformer. Configuración básica, 3 semillas, prefijos k = 1..6, partición temporal 64/16/20 por caso,
sin n_diag (se codifica al alta).

  Tarea 1  outcome_A   ingreso en UCI (label_ingreso_uci) SIN información de planificación: se quitan los eventos
                       DESTINO_* y la anestesia Gral-UCI / Gral-REA se recodifica como General. El prefijo termina
                       antes del primer INGRESO_UCI y nunca en ALTA / EXITUS / TRASLADO_OTRO_HOSPITAL.
  Tarea 2  outcome_E   estancia prolongada (label_estancia_prolongada, estancia > p90 = 8,74 d) con el log completo.
                       El prefijo no termina en fin y se descarta cuando el tiempo transcurrido ya supera el umbral.
  Tarea 3  nextact (F) siguiente actividad sobre las 14 actividades del log v3. (Letras A-E = variantes de resultado; F = siguiente actividad.)

Reutiliza ppm_data_v3 (datos), run_experiments_v3 (modelos, métricas, figuras) y run_experiments_targets (regla de
corte de la estancia). No modifica resultados/, resultados_con_ndiag/ ni resultados_targets/.

Uso:  python run_final_v3.py          (salida en PPM/resultados_final/)
"""
import os, time, warnings
os.environ["PPM_OUT"] = "resultados_final"
import ppm_data_v3 as D
import run_experiments_v3 as R
import run_experiments_targets as T

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    R.LOG = open(os.path.join(R.OUT, "log_ejecucion_final.txt"), "w", encoding="utf-8")
    t_all = time.time()
    ev, cases = D.load_log()
    R.p(f"Event log v3: {len(cases)} casos, {len(ev)} eventos, {ev.activity.nunique()} actividades, "
        f"{ev.groupby('case_id').activity.apply(tuple).nunique()} variantes")
    for se in (True, False):
        h2, n = D.process_entropy(ev, se, 2); he, _ = D.process_entropy(ev, se, "e")
        R.p(f"Entropía del proceso ({'con' if se else 'sin'} START/END): {h2:.3f} bits / {he:.3f} nats, {n} transiciones distintas")
    R.p(f"Receta: 5 algoritmos, semillas {R.SEEDS}, k = 1..{D.MAXLEN}, partición temporal 64/16/20, "
        f"estáticos excluidos {sorted(D.EXCLUDE_STATIC)}")
    umbral = float(cases.estancia_dias.quantile(0.90))

    # Tarea 1: ingreso en UCI sin planificación (regla de corte original de ppm_data_v3)
    R.p("\n##### TAREA 1: ingreso en UCI sin planificación (variante A)")
    sA = R.run_outcome("outcome_A", "A", ev, cases, target="label_ingreso_uci")

    # Tarea 2: estancia prolongada (regla de corte por umbral de tiempo)
    R.p(f"\n##### TAREA 2: estancia prolongada (> {umbral:.2f} días)")
    sE = T.run_with_cut("outcome_E", "B", ev, cases, "label_estancia_prolongada", T.make_cut_estancia(umbral))

    # Tarea 3: siguiente actividad (restaura la función de prefijos original, que también genera la tarea nextact)
    import importlib; importlib.reload(D)
    R.p("\n##### TAREA 3: siguiente actividad")
    sN = R.run_nextact(ev, cases)

    R.p(f"\nTiempo total: {time.time() - t_all:.0f} s")
    R.LOG.close()
