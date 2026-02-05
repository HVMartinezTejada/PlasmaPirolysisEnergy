import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

# =========================================================
# CONSTANTES DE UNIDADES (blindaje)
# =========================================================
GJ_TO_MWH = 0.2777777778     # 1 GJ = 0.27778 MWh
MWH_TO_GJ = 3.6              # 1 MWh = 3.6 GJ
KWH_TO_MWH = 1.0 / 1000.0

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(page_title="Boson BEU — Calculadora de Impacto", layout="wide")
st.title("⚡ Boson BEU — Calculadora de Impacto: Residuos → Energía → CO₂e")
st.markdown("---")

CREATED_BY = "Created by: H. Vladimir Martínez-T <hader.martinez@upb.edu.co> NDA Boson Energy-UPB 2025"

# =========================================================
# PRESETS DE COMPOSICIÓN (PCI + proxy de H2 teórico)
# NOTA: h2_teorico_kg_ton es un proxy de aproximación, NO una medición.
# =========================================================
PRESET_LA_PRADERA_REAL = {
    "orgánicos (húmedos)": {"pct": 46.0, "pci_gj_ton": 4.5, "h2_teorico_kg_ton": 60,  "cenizas_pct": 18},
    "plásticos totales":   {"pct": 16.0, "pci_gj_ton": 32.0, "h2_teorico_kg_ton": 260, "cenizas_pct": 4},
    "papel/cartón extendido": {"pct": 18.0, "pci_gj_ton": 14.0, "h2_teorico_kg_ton": 140, "cenizas_pct": 10},
    "textiles":            {"pct": 3.9, "pci_gj_ton": 18.0, "h2_teorico_kg_ton": 160, "cenizas_pct": 8},
    "especiales / electrónicos / caucho / cuero": {"pct": 10.0, "pci_gj_ton": 20.0, "h2_teorico_kg_ton": 190, "cenizas_pct": 12},
    "metales / vidrio / finos": {"pct": 4.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 85},
    "otros":               {"pct": 2.1, "pci_gj_ton": 10.0, "h2_teorico_kg_ton": 100, "cenizas_pct": 15},
}

# Un preset “mejor separación en fuente” (para explorar mayor PCI)
PRESET_LA_PRADERA_MEJOR_SEPARACION = {
    "orgánicos (húmedos)": {"pct": 40.0, "pci_gj_ton": 4.5, "h2_teorico_kg_ton": 60,  "cenizas_pct": 18},
    "plásticos totales":   {"pct": 20.0, "pci_gj_ton": 32.0, "h2_teorico_kg_ton": 260, "cenizas_pct": 4},
    "papel/cartón extendido": {"pct": 20.0, "pci_gj_ton": 14.0, "h2_teorico_kg_ton": 140, "cenizas_pct": 10},
    "textiles":            {"pct": 4.0, "pci_gj_ton": 18.0, "h2_teorico_kg_ton": 160, "cenizas_pct": 8},
    "especiales / electrónicos / caucho / cuero": {"pct": 10.0, "pci_gj_ton": 20.0, "h2_teorico_kg_ton": 190, "cenizas_pct": 12},
    "metales / vidrio / finos": {"pct": 4.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 85},
    "otros":               {"pct": 2.0, "pci_gj_ton": 10.0, "h2_teorico_kg_ton": 100, "cenizas_pct": 15},
}

PRESET_RSU_GENERICO = {
    "plásticos": {"pct": 12.0, "pci_gj_ton": 35.0, "h2_teorico_kg_ton": 240, "cenizas_pct": 5},
    "orgánicos": {"pct": 45.0, "pci_gj_ton": 5.0,  "h2_teorico_kg_ton": 60,  "cenizas_pct": 15},
    "papel/cartón": {"pct": 18.0, "pci_gj_ton": 16.0, "h2_teorico_kg_ton": 140, "cenizas_pct": 8},
    "textiles": {"pct": 4.0, "pci_gj_ton": 20.0, "h2_teorico_kg_ton": 160, "cenizas_pct": 10},
    "madera": {"pct": 3.0, "pci_gj_ton": 18.0, "h2_teorico_kg_ton": 120, "cenizas_pct": 5},
    "otros combustibles": {"pct": 5.0, "pci_gj_ton": 25.0, "h2_teorico_kg_ton": 200, "cenizas_pct": 20},
    "inertes/metales": {"pct": 10.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 95},
    "otros inertes": {"pct": 3.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 90},
}

# =========================================================
# PARÁMETROS BASE (ajustables desde UI sin mutar el dict global)
# =========================================================
PARAMS_BASE = {
    # Capacidad BEU (FOAK) — 1 planta (no 1 reactor)
    "capacidad_beu_ton_ano": 36000.0,

    # Conversión energética (orden de magnitud)
    "eficiencia_conversion_plasma": 0.78,
    "eficiencia_generacion_electrica": 0.38,
    "autoconsumo_proceso_fraction": 0.28,
    "bop_kwh_por_ton": 100.0,  # consumo eléctrico fijo por tonelada

    # H2 — Método KPI (recomendado)
    "h2_neto_stationary_kg_ton": 70.0,   # objetivo Boson (estacionario)
    "h2_neto_mobility_kg_ton": 50.0,     # objetivo Boson (movilidad)

    # H2 — Método proxy por composición
    "eficiencia_h2_desde_teorico": 0.75,  # teórico -> bruto (orden de magnitud)
    "h2_neto_factor_stationary": 0.58,    # bruto -> neto (stationary)
    "h2_neto_factor_mobility": 0.42,      # bruto -> neto (mobility)

    # Consumo eléctrico auxiliar para upgrading (kWh/kg H2 neto)
    "kwh_por_kg_h2_upgrading": 10.0,

    # Fuel-Cell (kWh/kg H2 convertido) — “electricidad DC entregable”
    "kwh_e_por_kg_h2_fuelcell": 18.0,

    # Calor útil (aprox). En vez de “fracciones mágicas”, lo dejamos como “factor recuperable”
    # para visualizar órdenes de magnitud sin doble contabilizar.
    # Nota: esto NO es un modelo térmico de detalle.
    "calor_util_mwhth_A": 2.0,
    "calor_util_mwhth_B": 0.6,
    "calor_util_mwhth_C": 1.6,

    # Subproductos (por tonelada) — rango típico 0.10–0.12 t/ton (editable)
    "imbyrock_kg_ton": 110.0,
    "metales_kg_ton": 10.0,

    # CO2 capturable del proceso (por tonelada)
    "co2_capturable_ton_por_ton": 0.90,
    "ccs_captura_frac": 0.85,

    # Emisiones indirectas (proxy)
    "emis_indirectas_kgco2e_ton": 100.0,

    # Línea base / logística / red
    "factor_relleno_kgco2e_ton": 640.0,
    "factor_red_tco2e_mwh": 0.21742,
    "factor_transporte_kgco2_ton_km": 0.127,
    "dist_baseline_km": 55.0,
    "dist_cluster_km": 15.0,

    # Sanity checks
    "sanity_warn_factor": 1.10,  # si energía útil supera 110% del PCI, avisar
}

# Umbrales autosuficiencia térmica (en GJ/ton)
UMBRAL_AUTOSUF_GJ_TON = 9.0
UMBRAL_CASI_AUTOSUF_GJ_TON = 7.0

# =========================================================
# HELPERS
# =========================================================
def normalizar_composicion(comp: dict) -> dict:
    total = sum(v["pct"] for v in comp.values())
    if total <= 0:
        return comp
    if abs(total - 100.0) < 1e-9:
        return comp
    comp2 = {}
    for k, v in comp.items():
        vv = v.copy()
        vv["pct"] = vv["pct"] * 100.0 / total
        comp2[k] = vv
    return comp2

def fmt_es_num(x, dec=0, signo=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if signo:
        s = f"{x:+,.{dec}f}"
    else:
        s = f"{x:,.{dec}f}"
    # miles con punto, decimales con coma
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    if dec == 0:
        s = s.replace(",0", "")
    return s

def fmt_mcop(cop_anual):
    return fmt_es_num(cop_anual / 1e6, dec=1, signo=False)

def anotar_barras(ax, bars, valores, dec=2, signo=False):
    for b, v in zip(bars, valores):
        label = fmt_es_num(v, dec=dec, signo=signo)
        y = b.get_height()
        y_text = y / 2 if y != 0 else 0
        ax.text(
            b.get_x() + b.get_width() / 2,
            y_text,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white"
        )

@st.cache_data
def calcular_propiedades_mezcla(comp):
    pci_gj_ton = 0.0
    h2_teorico_kg_ton = 0.0
    fraccion_cenizas = 0.0

    for _, d in comp.items():
        fr = d["pct"] / 100.0
        pci_gj_ton += fr * d["pci_gj_ton"]
        h2_teorico_kg_ton += fr * d["h2_teorico_kg_ton"]
        fraccion_cenizas += fr * (d["cenizas_pct"] / 100.0)

    pci_mwh_ton = pci_gj_ton * GJ_TO_MWH
    pci_kcal_kg = pci_gj_ton * 239.0
    return {
        "pci_gj_ton": pci_gj_ton,
        "pci_mwh_ton": pci_mwh_ton,
        "pci_kcal_kg": pci_kcal_kg,
        "h2_teorico_kg_ton": h2_teorico_kg_ton,
        "fraccion_cenizas": fraccion_cenizas,
    }

def modo_defs():
    return {
        "A": {"nombre": "Modo A — Power/Heat-centric", "power_split": 1.0, "h2_split": 0.0, "h2_exporta": False, "h2_a_fuelcell": False},
        "B": {"nombre": "Modo B — H₂-centric",        "power_split": 0.0, "h2_split": 1.0, "h2_exporta": True,  "h2_a_fuelcell": False},
        "C": {"nombre": "Modo C — Mixed",            "power_split": 0.5, "h2_split": 0.5, "h2_exporta": False, "h2_a_fuelcell": True},
    }

def calcular_h2_neto_kg_ton(
    metodo_h2: str,
    modo_key: str,
    props: dict,
    p: dict,
    h2_grade_key: str
) -> tuple[float, float]:
    """
    Retorna (h2_bruto_kg_ton, h2_neto_kg_ton) para la fracción H2 del modo.
    - método KPI: neto fijo (50/70) * split del modo
    - método proxy: usa h2_teorico y eficiencias
    """
    md = modo_defs()[modo_key]
    split = md["h2_split"]
    h2_teorico = props["h2_teorico_kg_ton"]

    if metodo_h2 == "KPI Boson (recomendado)":
        if h2_grade_key == "mobility":
            h2_neto = p["h2_neto_mobility_kg_ton"] * split
        else:
            h2_neto = p["h2_neto_stationary_kg_ton"] * split
        # “bruto” aquí es informativo: asumimos neto ≈ 0.6 del bruto (orden de magnitud)
        h2_bruto = h2_neto / 0.60 if h2_neto > 0 else 0.0
        return h2_bruto, h2_neto

    # Proxy por composición
    h2_bruto = h2_teorico * p["eficiencia_h2_desde_teorico"] * split
    if h2_grade_key == "mobility":
        h2_neto = h2_bruto * p["h2_neto_factor_mobility"]
    else:
        h2_neto = h2_bruto * p["h2_neto_factor_stationary"]
    return h2_bruto, h2_neto

def calcular_modo_por_ton(
    modo_key: str,
    props: dict,
    p: dict,
    metodo_h2: str,
    h2_grade_key: str,
    frac_h2_a_fc_B: float
) -> dict:
    md = modo_defs()[modo_key]

    pci_gj_ton = props["pci_gj_ton"]
    pci_mwh_ton = props["pci_mwh_ton"]

    # Syngas (GJ/ton) disponible tras plasma
    syngas_gj_ton = pci_gj_ton * p["eficiencia_conversion_plasma"]

    # Autoconsumo del proceso (sobre syngas)
    syngas_neto_gj_ton = syngas_gj_ton * (1.0 - p["autoconsumo_proceso_fraction"])

    # ======= RUTA POWER (desde syngas) =======
    power_gj_ton = syngas_neto_gj_ton * md["power_split"]

    # Electricidad bruta desde syngas (MWh_e/ton)
    # (GJ → MWh con factor correcto)
    electricidad_syngas_mwh_ton = power_gj_ton * GJ_TO_MWH * p["eficiencia_generacion_electrica"]

    # ======= RUTA H2 =======
    h2_bruto_kg_ton, h2_neto_kg_ton = calcular_h2_neto_kg_ton(
        metodo_h2=metodo_h2,
        modo_key=modo_key,
        props=props,
        p=p,
        h2_grade_key=h2_grade_key
    )

    # Consumo eléctrico auxiliar para upgrading (MWh_e/ton)
    consumo_aux_mwh_ton = (h2_neto_kg_ton * p["kwh_por_kg_h2_upgrading"]) * KWH_TO_MWH

    # Fuel-cell (electricidad DC bruta) según fracción de H2 a FC
    electricidad_fuelcell_mwh_ton = 0.0
    frac_h2_a_fc = 0.0

    if modo_key == "B":
        frac_h2_a_fc = max(0.0, min(1.0, frac_h2_a_fc_B))
    elif md["h2_a_fuelcell"]:
        frac_h2_a_fc = 1.0

    if frac_h2_a_fc > 0:
        electricidad_fuelcell_mwh_ton = (h2_neto_kg_ton * frac_h2_a_fc * p["kwh_e_por_kg_h2_fuelcell"]) * KWH_TO_MWH

    # H2 exportable
    h2_exportable_kg_ton = 0.0
    if md["h2_exporta"]:
        h2_exportable_kg_ton = h2_neto_kg_ton * (1.0 - frac_h2_a_fc)

    # BOP (MWh_e/ton)
    bop_mwh_ton = p["bop_kwh_por_ton"] * KWH_TO_MWH

    # Electricidad neta (balance sistema): exporta (+) / importa (-)
    electricidad_neta_mwh_ton = (
        electricidad_syngas_mwh_ton
        + electricidad_fuelcell_mwh_ton
        - bop_mwh_ton
        - consumo_aux_mwh_ton
    )

    # ======= CALOR ÚTIL (MWh_th/ton) =======
    # Modelo de aproximación (no térmico de detalle): valores por modo (editables)
    if modo_key == "A":
        calor_util_mwhth_ton = p["calor_util_mwhth_A"]
    elif modo_key == "B":
        calor_util_mwhth_ton = p["calor_util_mwhth_B"]
    else:
        calor_util_mwhth_ton = p["calor_util_mwhth_C"]

    # Subproductos
    imbyrock_kg_ton = p["imbyrock_kg_ton"]
    metales_kg_ton = p["metales_kg_ton"]

    # CO2 capturable
    co2_capturable_ton_ton = p["co2_capturable_ton_por_ton"]

    # ======= EMISIONES (baseline vs Boson) =======
    # Baseline: relleno + transporte baseline (kgCO2e/ton)
    baseline_kg = p["factor_relleno_kgco2e_ton"] + p["factor_transporte_kgco2_ton_km"] * p["dist_baseline_km"]

    # Boson: transporte en clúster (kgCO2e/ton)
    transporte_cluster_kg = p["factor_transporte_kgco2_ton_km"] * p["dist_cluster_km"]

    # Electricidad: si exporta (positiva), evita emisiones => efecto negativo
    # si importa (negativa), suma emisiones => efecto positivo
    grid_kg_por_mwh = p["factor_red_tco2e_mwh"] * 1000.0
    efecto_electricidad_kg = -electricidad_neta_mwh_ton * grid_kg_por_mwh

    # Proceso: CO2 capturable (kgCO2/ton)
    proceso_sin_ccs_kg = co2_capturable_ton_ton * 1000.0
    proceso_con_ccs_kg = proceso_sin_ccs_kg * (1.0 - p["ccs_captura_frac"])

    indirectas_kg = p["emis_indirectas_kgco2e_ton"]

    boson_sin_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_sin_ccs_kg + indirectas_kg
    boson_con_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_con_ccs_kg + indirectas_kg

    delta_sin_ccs_kg = boson_sin_ccs_kg - baseline_kg
    delta_con_ccs_kg = boson_con_ccs_kg - baseline_kg

    # ======= SANITY CHECKS (energía) =======
    # Energía útil aproximada: electricidad neta + calor útil (no incluye “valor del H2 exportable” como energía final)
    energia_util_mwh_ton = max(electricidad_neta_mwh_ton, 0.0) + max(calor_util_mwhth_ton, 0.0)
    sanity_exceso = energia_util_mwh_ton > (p["sanity_warn_factor"] * pci_mwh_ton)

    return {
        "modo": md["nombre"],
        "residuos_desviados_ton_ton": 1.0,

        "pci_gj_ton": pci_gj_ton,
        "pci_mwh_ton": pci_mwh_ton,

        "electricidad_syngas_mwh_e_ton": electricidad_syngas_mwh_ton,
        "electricidad_neta_mwh_e_ton": electricidad_neta_mwh_ton,
        "consumo_aux_mwh_e_ton": consumo_aux_mwh_ton,
        "electricidad_fc_mwh_e_ton": electricidad_fuelcell_mwh_ton,

        "calor_util_mwh_th_ton": calor_util_mwhth_ton,

        "h2_bruto_kg_ton": h2_bruto_kg_ton,
        "h2_neto_kg_ton": h2_neto_kg_ton,
        "h2_exportable_kg_ton": h2_exportable_kg_ton,
        "frac_h2_a_fc": frac_h2_a_fc,

        "imbyrock_kg_ton": imbyrock_kg_ton,
        "metales_kg_ton": metales_kg_ton,

        "co2_capturable_tco2_ton": co2_capturable_ton_ton,

        "baseline_kgco2e_ton": baseline_kg,
        "boson_sin_ccs_kgco2e_ton": boson_sin_ccs_kg,
        "boson_con_ccs_kgco2e_ton": boson_con_ccs_kg,
        "delta_sin_ccs_kgco2e_ton": delta_sin_ccs_kg,
        "delta_con_ccs_kgco2e_ton": delta_con_ccs_kg,

        "sanity_energia_exceso": sanity_exceso,
        "energia_util_mwh_ton": energia_util_mwh_ton,
    }

def escalar_a_anual(kpi_por_ton, toneladas_ano):
    t = float(toneladas_ano)
    return {
        "residuos_desviados_t_ano": t,
        "electricidad_neta_mwh_e_ano": kpi_por_ton["electricidad_neta_mwh_e_ton"] * t,
        "electricidad_fc_mwh_e_ano": kpi_por_ton["electricidad_fc_mwh_e_ton"] * t,
        "consumo_aux_mwh_e_ano": kpi_por_ton["consumo_aux_mwh_e_ton"] * t,
        "calor_util_mwh_th_ano": kpi_por_ton["calor_util_mwh_th_ton"] * t,
        "h2_total_t_ano": (kpi_por_ton["h2_neto_kg_ton"] * t) / 1000.0,
        "h2_exportable_t_ano": (kpi_por_ton["h2_exportable_kg_ton"] * t) / 1000.0,
        "imbyrock_t_ano": (kpi_por_ton["imbyrock_kg_ton"] * t) / 1000.0,
        "metales_t_ano": (kpi_por_ton["metales_kg_ton"] * t) / 1000.0,
        "co2_capturable_tco2_ano": kpi_por_ton["co2_capturable_tco2_ton"] * t,
        "baseline_tco2e_ano": (kpi_por_ton["baseline_kgco2e_ton"] * t) / 1000.0,
        "boson_sin_ccs_tco2e_ano": (kpi_por_ton["boson_sin_ccs_kgco2e_ton"] * t) / 1000.0,
        "boson_con_ccs_tco2e_ano": (kpi_por_ton["boson_con_ccs_kgco2e_ton"] * t) / 1000.0,
        "delta_sin_ccs_tco2e_ano": (kpi_por_ton["delta_sin_ccs_kgco2e_ton"] * t) / 1000.0,
        "delta_con_ccs_tco2e_ano": (kpi_por_ton["delta_con_ccs_kgco2e_ton"] * t) / 1000.0,
    }

def construir_tabla_modo(kpi_ton, kpi_ano):
    filas = [
        ("Residuos desviados de relleno (disposición evitada)", "t/año", kpi_ano["residuos_desviados_t_ano"], "t/ton", kpi_ton["residuos_desviados_ton_ton"]),
        ("IMBYROCK® (escoria vitrificada)", "t/año", kpi_ano["imbyrock_t_ano"], "kg/ton", kpi_ton["imbyrock_kg_ton"]),
        ("CO₂ capturable del proceso (flujo concentrado)", "tCO₂/año", kpi_ano["co2_capturable_tco2_ano"], "tCO₂/ton", kpi_ton["co2_capturable_tco2_ton"]),
        ("Electricidad neta del sistema (exporta + / importa -)", "MWhₑ/año", kpi_ano["electricidad_neta_mwh_e_ano"], "MWhₑ/ton", kpi_ton["electricidad_neta_mwh_e_ton"]),
        ("Electricidad DC bruta vía Fuel-Cell (subcomponente; no sumar aparte)", "MWhₑ/año", kpi_ano["electricidad_fc_mwh_e_ano"], "MWhₑ/ton", kpi_ton["electricidad_fc_mwh_e_ton"]),
        ("Consumo eléctrico auxiliar (ruta H₂)", "MWhₑ/año", kpi_ano["consumo_aux_mwh_e_ano"], "MWhₑ/ton", kpi_ton["consumo_aux_mwh_e_ton"]),
        ("Calor útil recuperable", "MWhₜₕ/año", kpi_ano["calor_util_mwh_th_ano"], "MWhₜₕ/ton", kpi_ton["calor_util_mwh_th_ton"]),
        ("H₂ neto producido", "t H₂/año", kpi_ano["h2_total_t_ano"], "kg H₂/ton", kpi_ton["h2_neto_kg_ton"]),
        ("H₂ exportable", "t H₂/año", kpi_ano["h2_exportable_t_ano"], "kg H₂/ton", kpi_ton["h2_exportable_kg_ton"]),
        ("Línea base: relleno + transporte", "tCO₂e/año", kpi_ano["baseline_tco2e_ano"], "kgCO₂e/ton", kpi_ton["baseline_kgco2e_ton"]),
        ("Boson total SIN CCS", "tCO₂e/año", kpi_ano["boson_sin_ccs_tco2e_ano"], "kgCO₂e/ton", kpi_ton["boson_sin_ccs_kgco2e_ton"]),
        ("Boson total CON CCS", "tCO₂e/año", kpi_ano["boson_con_ccs_tco2e_ano"], "kgCO₂e/ton", kpi_ton["boson_con_ccs_kgco2e_ton"]),
        ("Δ vs línea base SIN CCS (Boson − línea base)", "tCO₂e/año", kpi_ano["delta_sin_ccs_tco2e_ano"], "kgCO₂e/ton", kpi_ton["delta_sin_ccs_kgco2e_ton"]),
        ("Δ vs línea base CON CCS (Boson − línea base)", "tCO₂e/año", kpi_ano["delta_con_ccs_tco2e_ano"], "kgCO₂e/ton", kpi_ton["delta_con_ccs_kgco2e_ton"]),
    ]
    return pd.DataFrame(filas, columns=["Indicador", "Unidad (anual)", "Total anual", "Unidad (por ton)", "Por tonelada"])

def graf_comparador_simple(titulo, ylabel, modos, vals, dec=2, signo=False):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(modos))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(modos)
    ax.axhline(0, linewidth=0.8)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    anotar_barras(ax, bars, vals, dec=dec, signo=signo)
    plt.tight_layout()
    return fig

# =========================================================
# MÓDULO ECONÓMICO ULTRA-COMPACTO (sin CAPEX)
# =========================================================
def calcular_economia_ultra_compacta(kpi_ano, econ):
    residuos_t = kpi_ano["residuos_desviados_t_ano"]
    elec_mwh = kpi_ano["electricidad_neta_mwh_e_ano"]
    h2_export_t = kpi_ano["h2_exportable_t_ano"]
    imby_t = kpi_ano["imbyrock_t_ano"]
    delta_con_ccs_tco2e = kpi_ano["delta_con_ccs_tco2e_ano"]

    ingreso_residuos = residuos_t * econ["tarifa_residuos_cop_ton"]

    kwh = elec_mwh * 1000.0
    if kwh >= 0:
        ingreso_elec = kwh * econ["precio_electricidad_cop_kwh"]
        costo_elec = 0.0
    else:
        ingreso_elec = 0.0
        costo_elec = abs(kwh) * econ["precio_electricidad_cop_kwh"]

    ingreso_h2 = (h2_export_t * 1000.0) * econ["precio_h2_cop_kg"]
    ingreso_imby = imby_t * econ["precio_imbyrock_cop_ton"]

    ingreso_carbono = 0.0
    if econ["incluir_carbono"]:
        ahorro_tco2e = max(-delta_con_ccs_tco2e, 0.0)  # solo si Δ<0
        ingreso_carbono = ahorro_tco2e * econ["carbono_usd_tco2e"] * econ["fx_cop_usd"]

    ingreso_total = ingreso_residuos + ingreso_elec + ingreso_h2 + ingreso_imby + ingreso_carbono
    costo_total = costo_elec
    neto = ingreso_total - costo_total

    return {
        "ingreso_residuos": ingreso_residuos,
        "ingreso_elec": ingreso_elec,
        "costo_elec": costo_elec,
        "ingreso_h2": ingreso_h2,
        "ingreso_imby": ingreso_imby,
        "ingreso_carbono": ingreso_carbono,
        "ingreso_total": ingreso_total,
        "costo_total": costo_total,
        "neto": neto,
    }

def tabla_economica_por_modo(econ_by_mode):
    filas = [
        ("Ingresos", "Ingreso por residuos (MCOP/año)", "ingreso_residuos"),
        ("Ingresos", "Ingreso por electricidad (MCOP/año)", "ingreso_elec"),
        ("Costos",   "Costo por electricidad importada (MCOP/año)", "costo_elec"),
        ("Ingresos", "Ingreso por H₂ exportable (MCOP/año)", "ingreso_h2"),
        ("Ingresos", "Ingreso por IMBYROCK® (MCOP/año)", "ingreso_imby"),
        ("Ingresos", "Ingreso por carbono (MCOP/año)", "ingreso_carbono"),
        ("Ingresos", "Total ingresos (MCOP/año)", "ingreso_total"),
        ("Costos",   "Total costos (MCOP/año)", "costo_total"),
        ("Neto",     "Resultado neto = ingresos − costos (MCOP/año)", "neto"),
    ]
    data = []
    for tipo, concepto, key in filas:
        data.append([
            tipo,
            concepto,
            fmt_mcop(econ_by_mode["A"][key]),
            fmt_mcop(econ_by_mode["B"][key]),
            fmt_mcop(econ_by_mode["C"][key]),
        ])
    return pd.DataFrame(data, columns=["Tipo", "Concepto", "Modo A", "Modo B", "Modo C"])

def formatear_tabla_anual(df):
    out = df.copy()
    for col in ["Modo A", "Modo B", "Modo C"]:
        vals = []
        for _, row in out.iterrows():
            u = row["Unidad"]
            v = row[col]
            if "t/año" in u and "CO₂" not in u:
                vals.append(fmt_es_num(v, dec=0))
            elif "MWh" in u:
                vals.append(fmt_es_num(v, dec=0, signo=True))
            elif "t H₂/año" in u:
                vals.append(fmt_es_num(v, dec=1))
            elif "tCO₂e/año" in u:
                vals.append(fmt_es_num(v, dec=0, signo=True))
            else:
                vals.append(fmt_es_num(v, dec=1, signo=True))
        out[col] = vals
    return out

def formatear_tabla_ton(df):
    out = df.copy()
    for col in ["Modo A", "Modo B", "Modo C"]:
        vals = []
        for _, row in out.iterrows():
            u = row["Unidad"]
            v = row[col]
            if "t/ton" in u:
                vals.append(fmt_es_num(v, dec=1))
            elif "kg/ton" in u:
                vals.append(fmt_es_num(v, dec=1, signo=False))
            elif "MWh" in u:
                vals.append(fmt_es_num(v, dec=2, signo=True))
            elif "tCO₂/ton" in u:
                vals.append(fmt_es_num(v, dec=2, signo=False))
            elif "kgCO₂e/ton" in u:
                vals.append(fmt_es_num(v, dec=1, signo=True))
            else:
                vals.append(fmt_es_num(v, dec=1, signo=True))
        out[col] = vals
    return out

# =========================================================
# SIDEBAR — CONFIGURACIÓN
# =========================================================
with st.sidebar:
    st.markdown(f"**{CREATED_BY}**")
    st.markdown("---")

    st.subheader("🏭 Definición BEU (planta)")
    st.info(
        "FOAK / 1 BEU ≈ **36.000 t/año**.\n\n"
        "En la app: **1 BEU (planta) ≈ 2 reactores ×(2 t/h)-operando + 1 reactor en standby**."
    )

    st.subheader("🧪 Residuos (preset + ajuste)")
    preset = st.selectbox(
        "Preset de composición:",
        [
            "La Pradera (AMVA) — caso más real",
            "La Pradera (AMVA) — mejor separación en fuente (PCI mayor)",
            "RSU municipal genérico"
        ],
        index=0
    )

    if preset.startswith("La Pradera") and "mejor separación" in preset:
        comp_base = {k: v.copy() for k, v in PRESET_LA_PRADERA_MEJOR_SEPARACION.items()}
    elif preset.startswith("La Pradera"):
        comp_base = {k: v.copy() for k, v in PRESET_LA_PRADERA_REAL.items()}
    else:
        comp_base = {k: v.copy() for k, v in PRESET_RSU_GENERICO.items()}

    modo_comp = st.radio("Modo de composición:", ["Usar preset", "Personalizar porcentajes"], index=0)

    if modo_comp == "Personalizar porcentajes":
        st.caption("Ajusta los %; la app normaliza automáticamente a 100% si no coincide.")
        comp_user = {}
        for comp, d in comp_base.items():
            comp_user[comp] = d.copy()
            comp_user[comp]["pct"] = st.slider(
                comp, min_value=0.0, max_value=100.0, value=float(d["pct"]), step=0.5, key=f"pct_{preset}_{comp}"
            )
        composicion = normalizar_composicion(comp_user)
    else:
        composicion = normalizar_composicion(comp_base)
        with st.expander("Ver composición (% normalizado)"):
            for comp, d in composicion.items():
                st.write(f"- **{comp}**: {d['pct']:.1f}%")

    st.subheader("⚙️ Modo de operación")
    modo_operacion = st.selectbox(
        "Selecciona el modo:",
        ["A — Power/Heat-centric", "B — H₂-centric", "C — Mixed"],
        index=2
    )
    modo_key = modo_operacion.split("—")[0].strip()

    st.subheader("🧴 Grado de H₂ (cuando aplica)")
    h2_grade = st.selectbox("Grado de H₂:", ["Estacionario (≈95% / fast-charging)", "Movilidad (≈99.999%)"], index=0)
    h2_grade_key = "stationary" if h2_grade.startswith("Estacionario") else "mobility"

    st.subheader("🧮 Método de cálculo de H₂")
    metodo_h2 = st.selectbox(
        "Selecciona método:",
        ["KPI Boson (recomendado)", "Proxy por composición (exploratorio)"],
        index=0,
        help=(
            "KPI Boson: fija el H₂ neto por tonelada (50–70 kg/ton) como orden de magnitud del caso Boson.\n"
            "Proxy por composición: usa el H₂ teórico del residuo y eficiencias (útil para sensibilidad)."
        )
    )

    frac_h2_a_fc_pct = 0.0
    if modo_key == "B":
        st.subheader("🔁 Ruta interna del H₂ (Modo B)")
        frac_h2_a_fc_pct = st.slider(
            "Fracción de H₂ a Fuel-Cell (0–100%)",
            min_value=0.0, max_value=100.0, value=0.0, step=5.0,
            help="0% = todo el H₂ neto exportable • 100% = todo el H₂ neto se convierte a electricidad DC (sin exportar H₂)."
        )
        st.caption("✅ **0% = todo exportable** • ✅ **100% = todo a DC** (Fuel-Cell).")

    st.subheader("📥 Capacidad")
    cap_total = st.number_input(
        "Residuos a tratar (toneladas/año):",
        min_value=1000.0, max_value=1_300_000.0, value=36000.0, step=1000.0
    )

    # Construir params locales (NO mutar PARAMS_BASE global)
    p = PARAMS_BASE.copy()

    st.subheader("🌍 Emisiones (supuestos editables)")
    with st.expander("Editar factores (relleno / red / transporte / CCS / indirectas)"):
        p["factor_relleno_kgco2e_ton"] = st.number_input(
            "Factor relleno (kgCO₂e/ton) — línea base", min_value=0.0, max_value=2000.0,
            value=float(p["factor_relleno_kgco2e_ton"]), step=10.0
        )
        p["factor_red_tco2e_mwh"] = st.number_input(
            "Factor de emisión red (tCO₂e/MWh)", min_value=0.0, max_value=1.5,
            value=float(p["factor_red_tco2e_mwh"]), step=0.01, format="%.5f"
        )
        p["factor_transporte_kgco2_ton_km"] = st.number_input(
            "Transporte (kgCO₂/(ton·km))", min_value=0.0, max_value=1.0,
            value=float(p["factor_transporte_kgco2_ton_km"]), step=0.005, format="%.3f"
        )
        p["dist_baseline_km"] = st.number_input(
            "Distancia baseline AMVA→La Pradera (km)", min_value=0.0, max_value=200.0,
            value=float(p["dist_baseline_km"]), step=1.0
        )
        p["dist_cluster_km"] = st.number_input(
            "Distancia clúster descentralizado (km)", min_value=0.0, max_value=100.0,
            value=float(p["dist_cluster_km"]), step=1.0
        )
        p["co2_capturable_ton_por_ton"] = st.number_input(
            "CO₂ capturable del proceso (tCO₂/ton)", min_value=0.2, max_value=2.0,
            value=float(p["co2_capturable_ton_por_ton"]), step=0.05
        )
        p["ccs_captura_frac"] = st.number_input(
            "Captura CCS (fracción 0–1)", min_value=0.0, max_value=1.0,
            value=float(p["ccs_captura_frac"]), step=0.01
        )
        p["emis_indirectas_kgco2e_ton"] = st.number_input(
            "Emisiones indirectas (kgCO₂e/ton) — proxy", min_value=0.0, max_value=500.0,
            value=float(p["emis_indirectas_kgco2e_ton"]), step=5.0
        )

    st.subheader("🔌 Ruta H₂: consumos y Fuel-Cell")
    with st.expander("Editar supuestos H₂ (consumo auxiliar / Fuel-Cell / KPI neto)"):
        p["kwh_por_kg_h2_upgrading"] = st.number_input(
            "Upgrading H₂ (kWhₑ/kg H₂ neto) — consumo eléctrico auxiliar",
            min_value=0.0, max_value=30.0, value=float(p["kwh_por_kg_h2_upgrading"]), step=0.5
        )
        p["kwh_e_por_kg_h2_fuelcell"] = st.number_input(
            "Fuel-Cell (kWhₑ/kg H₂ convertido) — electricidad DC entregable",
            min_value=0.0, max_value=30.0, value=float(p["kwh_e_por_kg_h2_fuelcell"]), step=0.5
        )
        p["h2_neto_stationary_kg_ton"] = st.number_input(
            "KPI H₂ neto estacionario (kg/ton)",
            min_value=0.0, max_value=150.0, value=float(p["h2_neto_stationary_kg_ton"]), step=5.0
        )
        p["h2_neto_mobility_kg_ton"] = st.number_input(
            "KPI H₂ neto movilidad (kg/ton)",
            min_value=0.0, max_value=150.0, value=float(p["h2_neto_mobility_kg_ton"]), step=5.0
        )

    st.subheader("🔥 Calor útil (MWhₜₕ/ton)")
    with st.expander("Editar calor útil por modo (modelo de aproximación)"):
        p["calor_util_mwhth_A"] = st.number_input("Modo A: calor útil (MWhₜₕ/ton)", min_value=0.0, max_value=5.0,
                                                 value=float(p["calor_util_mwhth_A"]), step=0.1)
        p["calor_util_mwhth_B"] = st.number_input("Modo B: calor útil (MWhₜₕ/ton)", min_value=0.0, max_value=5.0,
                                                 value=float(p["calor_util_mwhth_B"]), step=0.1)
        p["calor_util_mwhth_C"] = st.number_input("Modo C: calor útil (MWhₜₕ/ton)", min_value=0.0, max_value=5.0,
                                                 value=float(p["calor_util_mwhth_C"]), step=0.1)

    st.subheader("🧱 IMBYROCK® / metales")
    with st.expander("Editar subproductos por tonelada"):
        p["imbyrock_kg_ton"] = st.number_input("IMBYROCK® (kg/ton)", min_value=0.0, max_value=500.0,
                                              value=float(p["imbyrock_kg_ton"]), step=5.0)
        p["metales_kg_ton"] = st.number_input("Metales recuperables (kg/ton)", min_value=0.0, max_value=200.0,
                                             value=float(p["metales_kg_ton"]), step=1.0)

    st.subheader("💰 Módulo económico ultra-compacto (sin CAPEX)")
    with st.expander("Editar supuestos económicos (ingresos/costos directos)"):
        tarifa_residuos = st.number_input(
            "Tarifa por tratamiento/disposición evitada (COP/ton)",
            min_value=0.0, max_value=300000.0, value=109000.0, step=1000.0
        )
        precio_elec = st.number_input(
            "Precio energía (COP/kWh)",
            min_value=0.0, max_value=2000.0, value=300.0, step=10.0
        )
        precio_h2 = st.number_input(
            "Precio H₂ exportable (COP/kg) — opcional",
            min_value=0.0, max_value=100000.0, value=0.0, step=500.0
        )
        precio_imby = st.number_input(
            "Precio IMBYROCK® (COP/ton) — opcional",
            min_value=0.0, max_value=500000.0, value=0.0, step=5000.0
        )
        incluir_carbono = st.checkbox("Incluir ingreso por carbono (opcional)", value=False)
        carbono_usd = st.number_input("Precio carbono (USD/tCO₂e)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        fx_cop_usd = st.number_input("Tasa de cambio (COP/USD)", min_value=1000.0, max_value=10000.0, value=4200.0, step=50.0)

    econ_params = {
        "tarifa_residuos_cop_ton": float(tarifa_residuos),
        "precio_electricidad_cop_kwh": float(precio_elec),
        "precio_h2_cop_kg": float(precio_h2),
        "precio_imbyrock_cop_ton": float(precio_imby),
        "incluir_carbono": bool(incluir_carbono),
        "carbono_usd_tco2e": float(carbono_usd),
        "fx_cop_usd": float(fx_cop_usd),
    }

    st.markdown("---")
    st.caption("Δ vs línea base: **Δ < 0 = ahorro neto** (Boson mejor); **Δ > 0 = penalidad**.")

# =========================================================
# MAIN — KPI mezcla + autosuficiencia
# =========================================================
props = calcular_propiedades_mezcla(composicion)
pci_gj = props["pci_gj_ton"]
pci_mwh = props["pci_mwh_ton"]
pci_kcal = props["pci_kcal_kg"]

colA, colB, colC = st.columns(3)
with colA:
    st.metric("PCI de la mezcla (GJ/ton)", f"{pci_gj:.2f}")
with colB:
    st.metric("PCI equivalente (MWh/ton)", f"{pci_mwh:.2f}")
with colC:
    st.metric("PCI equivalente (kcal/kg)", f"{int(round(pci_kcal)):,}".replace(",", "."))

if pci_gj >= UMBRAL_AUTOSUF_GJ_TON:
    st.success(
        f"🔥 **Autosuficiencia térmica:** Autosuficiente. "
        f"El PCI ({pci_gj:.2f} GJ/ton) supera el umbral ≈{UMBRAL_AUTOSUF_GJ_TON:.1f} GJ/ton (≈2,5 MWh/ton)."
    )
elif pci_gj >= UMBRAL_CASI_AUTOSUF_GJ_TON:
    st.warning(
        f"🌡️ **Autosuficiencia térmica:** Casi autosuficiente. "
        f"El PCI ({pci_gj:.2f} GJ/ton) está en la franja 7–9 GJ/ton."
    )
else:
    st.error(
        f"❄️ **Autosuficiencia térmica:** Requiere apoyo energético. "
        f"El PCI ({pci_gj:.2f} GJ/ton) está por debajo de {UMBRAL_CASI_AUTOSUF_GJ_TON:.1f} GJ/ton."
    )

cap_beu = float(p["capacidad_beu_ton_ano"])
n_beu = int(math.ceil(float(cap_total) / cap_beu))
cap_por_beu = float(cap_total) / n_beu

st.info(
    f"🏗️ **Despliegue modular estimado:** **{n_beu} BEU(s)** para {fmt_es_num(cap_total,0)} t/año "
    f"(≈ {fmt_es_num(cap_por_beu,0)} t/año por BEU)."
)

st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    btn_calcular_modo = st.button("🚀 Calcular modo seleccionado", type="primary", use_container_width=True)
with c2:
    btn_comparar = st.button("🧭 Comparar Modos A vs B vs C (mismos supuestos)", use_container_width=True)

frac_B = float(frac_h2_a_fc_pct) / 100.0

# =========================================================
# RESULTADOS — MODO SELECCIONADO
# =========================================================
if btn_calcular_modo:
    st.header("📌 Resultados — modo seleccionado")

    kpi_ton = calcular_modo_por_ton(
        modo_key=modo_key,
        props=props,
        p=p,
        metodo_h2=metodo_h2,
        h2_grade_key=h2_grade_key,
        frac_h2_a_fc_B=frac_B
    )
    kpi_ano = escalar_a_anual(kpi_ton, cap_total)

    st.subheader(kpi_ton["modo"])
    df_modo = construir_tabla_modo(kpi_ton, kpi_ano)

    df_show = df_modo.copy()
    df_show["Total anual"] = df_show.apply(
        lambda r: fmt_es_num(r["Total anual"], dec=0, signo=("Δ" in r["Indicador"])), axis=1
    )
    df_show["Por tonelada"] = df_show.apply(
        lambda r: fmt_es_num(
            r["Por tonelada"],
            dec=2 if "MWh" in r["Unidad (por ton)"] else 1,
            signo=("Δ" in r["Indicador"])
        ),
        axis=1
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.info(
        "Claridad: **“Electricidad DC bruta vía Fuel-Cell”** es un **subcomponente** ya incluido en "
        "**“Electricidad neta del sistema”**. Se muestra para visualizar el “retorno” eléctrico del H₂, "
        "pero **no** debe sumarse como energía adicional."
    )

    if kpi_ton["sanity_energia_exceso"]:
        st.warning(
            f"⚠️ **Chequeo de sanidad energética:** (Electricidad neta positiva + Calor útil) ≈ "
            f"{fmt_es_num(kpi_ton['energia_util_mwh_ton'], dec=2)} MWh/ton "
            f"supera el {int(p['sanity_warn_factor']*100)}% del PCI ({fmt_es_num(pci_mwh, dec=2)} MWh/ton). "
            "Revisa supuestos de calor útil, eficiencias o el método de H₂."
        )

# =========================================================
# COMPARADOR — MODOS A vs B vs C
# =========================================================
if btn_comparar:
    st.header("🧭 Comparador de escenarios A vs B vs C")

    kpis_ton = {}
    kpis_ano = {}
    econ_res = {}

    for mk in ["A", "B", "C"]:
        kpi_t = calcular_modo_por_ton(
            modo_key=mk,
            props=props,
            p=p,
            metodo_h2=metodo_h2,
            h2_grade_key=h2_grade_key,
            frac_h2_a_fc_B=frac_B
        )
        kpi_a = escalar_a_anual(kpi_t, cap_total)
        kpis_ton[mk] = kpi_t
        kpis_ano[mk] = kpi_a
        econ_res[mk] = calcular_economia_ultra_compacta(kpi_a, econ_params)

    # --- Tabla ANUAL ---
    st.subheader("📌 Comparación anual (resultados directos con tus toneladas/año)")
    filas_anual = [
        ("Residuos desviados (disposición evitada)", "t/año", "residuos_desviados_t_ano"),
        ("IMBYROCK® (escoria vitrificada)", "t/año", "imbyrock_t_ano"),
        ("CO₂ capturable del proceso", "tCO₂/año", "co2_capturable_tco2_ano"),
        ("Electricidad neta del sistema (exporta + / importa -)", "MWhₑ/año", "electricidad_neta_mwh_e_ano"),
        ("Electricidad DC bruta vía Fuel-Cell (subcomponente)", "MWhₑ/año", "electricidad_fc_mwh_e_ano"),
        ("Consumo eléctrico auxiliar (ruta H₂)", "MWhₑ/año", "consumo_aux_mwh_e_ano"),
        ("Calor útil recuperable", "MWhₜₕ/año", "calor_util_mwh_th_ano"),
        ("H₂ neto producido", "t H₂/año", "h2_total_t_ano"),
        ("H₂ exportable", "t H₂/año", "h2_exportable_t_ano"),
        ("Línea base (relleno + transporte)", "tCO₂e/año", "baseline_tco2e_ano"),
        ("Boson SIN CCS", "tCO₂e/año", "boson_sin_ccs_tco2e_ano"),
        ("Boson CON CCS", "tCO₂e/año", "boson_con_ccs_tco2e_ano"),
        ("Δ vs línea base SIN CCS (Boson − línea base)", "tCO₂e/año", "delta_sin_ccs_tco2e_ano"),
        ("Δ vs línea base CON CCS (Boson − línea base)", "tCO₂e/año", "delta_con_ccs_tco2e_ano"),
    ]
    data_anual = []
    for nombre, unidad, key in filas_anual:
        data_anual.append([nombre, unidad, kpis_ano["A"][key], kpis_ano["B"][key], kpis_ano["C"][key]])
    df_comp_anual = pd.DataFrame(data_anual, columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"])
    st.dataframe(formatear_tabla_anual(df_comp_anual), use_container_width=True, hide_index=True)

    # --- Tabla POR TONELADA ---
    st.subheader("📊 Comparación por tonelada (normalizada; misma composición y supuestos)")
    filas_ton = [
        ("Residuos desviados (por definición)", "t/ton", "residuos_desviados_ton_ton"),
        ("IMBYROCK® (escoria vitrificada)", "kg/ton", "imbyrock_kg_ton"),
        ("CO₂ capturable del proceso", "tCO₂/ton", "co2_capturable_tco2_ton"),
        ("Electricidad neta del sistema", "MWhₑ/ton", "electricidad_neta_mwh_e_ton"),
        ("Electricidad DC bruta vía Fuel-Cell (subcomponente; no sumar aparte)", "MWhₑ/ton", "electricidad_fc_mwh_e_ton"),
        ("Consumo eléctrico auxiliar (ruta H₂)", "MWhₑ/ton", "consumo_aux_mwh_e_ton"),
        ("Calor útil recuperable", "MWhₜₕ/ton", "calor_util_mwh_th_ton"),
        ("H₂ neto", "kg/ton", "h2_neto_kg_ton"),
        ("H₂ exportable", "kg/ton", "h2_exportable_kg_ton"),
        ("Línea base (relleno + transporte)", "kgCO₂e/ton", "baseline_kgco2e_ton"),
        ("Boson SIN CCS", "kgCO₂e/ton", "boson_sin_ccs_kgco2e_ton"),
        ("Boson CON CCS", "kgCO₂e/ton", "boson_con_ccs_kgco2e_ton"),
        ("Δ vs línea base SIN CCS (Boson − línea base)", "kgCO₂e/ton", "delta_sin_ccs_kgco2e_ton"),
        ("Δ vs línea base CON CCS (Boson − línea base)", "kgCO₂e/ton", "delta_con_ccs_kgco2e_ton"),
    ]
    data_ton = []
    for nombre, unidad, key in filas_ton:
        data_ton.append([nombre, unidad, kpis_ton["A"][key], kpis_ton["B"][key], kpis_ton["C"][key]])
    df_comp_ton = pd.DataFrame(data_ton, columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"])
    st.dataframe(formatear_tabla_ton(df_comp_ton), use_container_width=True, hide_index=True)

    st.info(
        "Nota: **por tonelada** es normalizada (**Residuos desviados = 1,0 t/ton**). "
        "**Anual** muestra el valor directo para tu entrada (p.ej. 1.277.500 t/año). "
        "La fila **Electricidad DC bruta vía Fuel-Cell** es un subcomponente ya incluido en **Electricidad neta del sistema**."
    )

    # --- Gráficos separados ---
    st.subheader("📈 Comparadores gráficos (resumen)")
    modos = ["A", "B", "C"]

    st.pyplot(
        graf_comparador_simple(
            "Comparador — Electricidad neta del sistema (por tonelada)",
            "MWhₑ/ton",
            modos,
            [kpis_ton[m]["electricidad_neta_mwh_e_ton"] for m in modos],
            dec=2,
            signo=True
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Calor útil recuperable (por tonelada)",
            "MWhₜₕ/ton",
            modos,
            [kpis_ton[m]["calor_util_mwh_th_ton"] for m in modos],
            dec=2,
            signo=False
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Emisiones (por tonelada): Δ vs línea base SIN CCS",
            "kgCO₂e/ton  (Δ = Boson − línea base)",
            modos,
            [kpis_ton[m]["delta_sin_ccs_kgco2e_ton"] for m in modos],
            dec=1,
            signo=True
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Emisiones (por tonelada): Δ vs línea base CON CCS",
            "kgCO₂e/ton  (Δ = Boson − línea base)",
            modos,
            [kpis_ton[m]["delta_con_ccs_kgco2e_ton"] for m in modos],
            dec=1,
            signo=True
        )
    )

    # --- Económico ultra-compacto ---
    st.markdown("---")
    st.subheader("💰 Comparador económico ultra-compacto (sin CAPEX) — por modo")
    st.caption(
        "Nota de aproximación: el comparador económico **NO incluye CAPEX ni OPEX**. "
        "El objetivo es visualizar órdenes de magnitud e identificar el modo con mejor mezcla de ingresos bajo supuestos dados. "
        "Ingresos por IMBYROCK®, carbono u otros (p. ej., H₂ exportable) pueden ser **0** si no se conocen precios."
    )

    df_econ = tabla_economica_por_modo(econ_res)
    st.dataframe(df_econ, use_container_width=True, hide_index=True)
    st.caption("**MCOP = Millones de COP** (COP ÷ 1.000.000). Los cálculos se realizan en COP/año y se reportan en MCOP/año.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "⚠️ Nota: Esta calculadora es un **modelo de aproximación** para explorar órdenes de magnitud y trade-offs por modo. "
    "No contabiliza el beneficio adicional de que el H₂ desplace diésel/gasolina o H₂ gris, porque depende del end-use."
)
st.markdown(
    f"<p style='text-align:center; font-size:12px; color:gray;'>{CREATED_BY}</p>",
    unsafe_allow_html=True
)
