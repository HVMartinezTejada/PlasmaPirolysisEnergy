import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math

# =============================================
# CONFIGURACIÓN DE PÁGINA
# =============================================
st.set_page_config(page_title="Boson BEU — Calculadora de Impacto", layout="wide")
st.title("⚡ Boson BEU — Calculadora de Impacto: Residuos → Energía → CO₂e")
st.markdown("---")

CREATED_BY = "Created by: H. Vladimir Martínez-T <hader.martinez@upb.edu.co> NDA Boson Energy-UPB 2025"

# =============================================
# PRESETS DE COMPOSICIÓN (PCI y potencial H2 "teórico")
# Nota: h2_teorico_kg_ton es un proxy de aproximación (no es medición).
# =============================================
PRESET_LA_PRADERA = {
    "orgánicos (húmedos)": {"pct": 46.0, "pci_gj_ton": 4.5, "h2_teorico_kg_ton": 60, "cenizas_pct": 18},
    "plásticos totales": {"pct": 16.0, "pci_gj_ton": 32.0, "h2_teorico_kg_ton": 260, "cenizas_pct": 4},
    "papel/cartón extendido": {"pct": 18.0, "pci_gj_ton": 14.0, "h2_teorico_kg_ton": 140, "cenizas_pct": 10},
    "textiles": {"pct": 3.9, "pci_gj_ton": 18.0, "h2_teorico_kg_ton": 160, "cenizas_pct": 8},
    "especiales / electrónicos / caucho / cuero": {"pct": 10.0, "pci_gj_ton": 20.0, "h2_teorico_kg_ton": 190, "cenizas_pct": 12},
    "metales / vidrio / finos": {"pct": 4.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 85},
    "otros": {"pct": 2.1, "pci_gj_ton": 10.0, "h2_teorico_kg_ton": 100, "cenizas_pct": 15},
}

PRESET_RSU_GENERICO = {
    "plásticos": {"pct": 12.0, "pci_gj_ton": 35.0, "h2_teorico_kg_ton": 240, "cenizas_pct": 5},
    "orgánicos": {"pct": 45.0, "pci_gj_ton": 5.0, "h2_teorico_kg_ton": 60, "cenizas_pct": 15},
    "papel/cartón": {"pct": 18.0, "pci_gj_ton": 16.0, "h2_teorico_kg_ton": 140, "cenizas_pct": 8},
    "textiles": {"pct": 4.0, "pci_gj_ton": 20.0, "h2_teorico_kg_ton": 160, "cenizas_pct": 10},
    "madera": {"pct": 3.0, "pci_gj_ton": 18.0, "h2_teorico_kg_ton": 120, "cenizas_pct": 5},
    "otros combustibles": {"pct": 5.0, "pci_gj_ton": 25.0, "h2_teorico_kg_ton": 200, "cenizas_pct": 20},
    "inertes/metales": {"pct": 10.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 95},
    "otros inertes": {"pct": 3.0, "pci_gj_ton": 0.0, "h2_teorico_kg_ton": 0, "cenizas_pct": 90},
}

# =============================================
# PARÁMETROS BASE (comunes)
# =============================================
PARAMS_BASE = {
    # Capacidad BEU (FOAK) — 1 planta (no 1 reactor)
    "capacidad_beu_ton_ano": 36000.0,

    # Conversión energética
    "eficiencia_conversion_plasma": 0.78,
    "eficiencia_generacion_electrica": 0.38,
    "autoconsumo_proceso_fraction": 0.28,
    "bop_kwh_por_ton": 100.0,  # Balance of Plant (consumo eléctrico fijo por tonelada)

    # H2 (calibración de aproximación)
    "eficiencia_h2_desde_teorico": 0.75,  # teórico ~160 → bruto ~120 (orden de magnitud)
    "h2_neto_factor_stationary": 0.58,    # bruto ~120 → neto ~70
    "h2_neto_factor_mobility": 0.42,      # bruto ~120 → neto ~50
    "kwh_por_kg_h2_upgrading": 10.0,      # consumo eléctrico auxiliar: compresión, WGS, PSA, etc.
    "kwh_e_por_kg_h2_fuelcell": 18.0,     # electricidad DC aproximada en fuel cell (orden de magnitud)

    # Calor útil (fracción del contenido energético del residuo; MWh_th)
    "heat_fraction_A": 0.46,
    "heat_fraction_B": 0.14,
    "heat_fraction_C": 0.41,

    # Subproductos (por tonelada)
    "imbyrock_kg_ton": 110.0,             # 0.10–0.12 t/ton típico
    "metales_kg_ton": 10.0,

    # CO2 capturable del proceso (por tonelada)
    "co2_capturable_ton_por_ton": 0.90,   # 0.7–1.1 tCO2/ton
    "ccs_captura_frac": 0.85,

    # Emisiones indirectas (proxy)
    "emis_indirectas_kgco2e_ton": 100.0,

    # Línea base / logística / red (ajustable en UI)
    "factor_relleno_kgco2e_ton": 640.0,   # proxy estilo WARM (línea base)
    "factor_red_tco2e_mwh": 0.21742,      # proxy Colombia (tCO2e/MWh)
    "factor_transporte_kgco2_ton_km": 0.127,
    "dist_baseline_km": 55.0,             # AMVA → La Pradera
    "dist_cluster_km": 15.0,              # clúster descentralizado
}

# Umbrales autosuficiencia térmica (en GJ/ton)
UMBRAL_AUTOSUF_GJ_TON = 9.0
UMBRAL_CASI_AUTOSUF_GJ_TON = 7.0

# =============================================
# HELPERS
# =============================================
def normalizar_composicion(comp: dict) -> dict:
    """Normaliza porcentajes a 100% si el usuario deja una suma distinta."""
    total = sum(v["pct"] for v in comp.values())
    if total <= 0:
        return comp
    if abs(total - 100.0) < 1e-6:
        return comp
    comp2 = {}
    for k, v in comp.items():
        vv = v.copy()
        vv["pct"] = vv["pct"] * 100.0 / total
        comp2[k] = vv
    return comp2

def extraer_numero(txt):
    if isinstance(txt, (int, float, np.number)):
        return float(txt)
    s = str(txt).replace("\xa0", " ").replace(",", "")
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else 0.0

@st.cache_data
def calcular_propiedades_mezcla(comp):
    """Retorna PCI (GJ/ton), PCI (MWh/ton), H2 teórico (kg/ton), fracción cenizas."""
    pci_gj_ton = 0.0
    h2_teorico_kg_ton = 0.0
    fraccion_cenizas = 0.0

    for _, d in comp.items():
        fr = d["pct"] / 100.0
        pci_gj_ton += fr * d["pci_gj_ton"]
        h2_teorico_kg_ton += fr * d["h2_teorico_kg_ton"]
        fraccion_cenizas += fr * (d["cenizas_pct"] / 100.0)

    pci_mwh_ton = pci_gj_ton / 3.6
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
        "A": {
            "nombre": "Modo A — Power/Heat-centric",
            "power_split": 1.0,
            "h2_split": 0.0,
            "h2_exporta": False,
            "h2_a_fuelcell": False,
        },
        "B": {
            "nombre": "Modo B — H₂-centric",
            "power_split": 0.0,
            "h2_split": 1.0,
            "h2_exporta": True,
            "h2_a_fuelcell": False,
        },
        "C": {
            "nombre": "Modo C — Mixed",
            "power_split": 0.5,
            "h2_split": 0.5,
            "h2_exporta": False,   # en Mixed asumimos H2→fuel cell para “hub/isla”
            "h2_a_fuelcell": True,
        },
    }

def calcular_modo_por_ton(modo_key, props, p, h2_grade="stationary"):
    md = modo_defs()[modo_key]

    pci_gj_ton = props["pci_gj_ton"]
    pci_mwh_ton = props["pci_mwh_ton"]
    h2_teorico = props["h2_teorico_kg_ton"]

    # Syngas (GJ/ton) desde PCI
    syngas_gj_ton = pci_gj_ton * p["eficiencia_conversion_plasma"]

    # ======= RUTA POWER =======
    power_gj_ton = syngas_gj_ton * md["power_split"]
    autoconsumo_proceso_gj_ton = power_gj_ton * p["autoconsumo_proceso_fraction"]
    neto_power_gj_ton = power_gj_ton - autoconsumo_proceso_gj_ton
    electricidad_syngas_mwh_ton = (neto_power_gj_ton * 277.78) * p["eficiencia_generacion_electrica"]

    # ======= RUTA H2 =======
    h2_bruto_kg_ton = h2_teorico * p["eficiencia_h2_desde_teorico"] * md["h2_split"]

    if h2_grade == "mobility":
        h2_neto_kg_ton = h2_bruto_kg_ton * p["h2_neto_factor_mobility"]
    else:
        h2_neto_kg_ton = h2_bruto_kg_ton * p["h2_neto_factor_stationary"]

    # Consumo eléctrico auxiliar (MWh_e/ton) para upgrading / compresión / PSA
    consumo_aux_mwh_ton = (h2_neto_kg_ton * p["kwh_por_kg_h2_upgrading"]) / 1000.0

    # Si el modo manda H2 a fuel cell (Modo C), se convierte en electricidad DC aproximada
    electricidad_fuelcell_mwh_ton = 0.0
    if md["h2_a_fuelcell"]:
        electricidad_fuelcell_mwh_ton = (h2_neto_kg_ton * p["kwh_e_por_kg_h2_fuelcell"]) / 1000.0

    # Balance of Plant (MWh/ton)
    bop_mwh_ton = p["bop_kwh_por_ton"] / 1000.0

    # Electricidad neta (MWh_e/ton): exporta + / importa -
    electricidad_neta_mwh_ton = (
        electricidad_syngas_mwh_ton
        + electricidad_fuelcell_mwh_ton
        - bop_mwh_ton
        - consumo_aux_mwh_ton
    )

    # Calor útil (MWh_th/ton) — se reporta por separado
    if modo_key == "A":
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_A"]
    elif modo_key == "B":
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_B"]
    else:
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_C"]

    # Subproductos
    imbyrock_kg_ton = p["imbyrock_kg_ton"]
    metales_kg_ton = p["metales_kg_ton"]
    co2_capturable_ton_ton = p["co2_capturable_ton_por_ton"]

    # ======= EMISIONES (baseline vs Boson) =======
    # Línea base: relleno + transporte largo
    baseline_kg = p["factor_relleno_kgco2e_ton"] + p["factor_transporte_kgco2_ton_km"] * p["dist_baseline_km"]

    # Transporte Boson: clúster corto
    transporte_cluster_kg = p["factor_transporte_kgco2_ton_km"] * p["dist_cluster_km"]

    # Electricidad: si exportas, evitas; si importas, cargas huella.
    grid_kg_por_mwh = p["factor_red_tco2e_mwh"] * 1000.0
    efecto_electricidad_kg = -electricidad_neta_mwh_ton * grid_kg_por_mwh

    # Proceso: CO2 capturable (sin CCS vs con CCS)
    proceso_sin_ccs_kg = co2_capturable_ton_ton * 1000.0
    proceso_con_ccs_kg = proceso_sin_ccs_kg * (1.0 - p["ccs_captura_frac"])

    indirectas_kg = p["emis_indirectas_kgco2e_ton"]

    boson_sin_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_sin_ccs_kg + indirectas_kg
    boson_con_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_con_ccs_kg + indirectas_kg

    delta_sin_ccs_kg = boson_sin_ccs_kg - baseline_kg
    delta_con_ccs_kg = boson_con_ccs_kg - baseline_kg

    # H2 exportable (solo Modo B)
    h2_exportable_kg_ton = h2_neto_kg_ton if md["h2_exporta"] else 0.0

    return {
        "modo": md["nombre"],
        "residuos_desviados_ton_ton": 1.0,
        "pci_gj_ton": pci_gj_ton,
        "pci_mwh_ton": pci_mwh_ton,
        "electricidad_neta_mwh_e_ton": electricidad_neta_mwh_ton,
        "consumo_aux_mwh_e_ton": consumo_aux_mwh_ton,
        "calor_util_mwh_th_ton": calor_util_mwhth_ton,
        "h2_bruto_kg_ton": h2_bruto_kg_ton,
        "h2_neto_kg_ton": h2_neto_kg_ton,
        "h2_exportable_kg_ton": h2_exportable_kg_ton,
        "imbyrock_kg_ton": imbyrock_kg_ton,
        "metales_kg_ton": metales_kg_ton,
        "co2_capturable_tco2_ton": co2_capturable_ton_ton,
        "baseline_kgco2e_ton": baseline_kg,
        "boson_sin_ccs_kgco2e_ton": boson_sin_ccs_kg,
        "boson_con_ccs_kgco2e_ton": boson_con_ccs_kg,
        "delta_sin_ccs_kgco2e_ton": delta_sin_ccs_kg,
        "delta_con_ccs_kgco2e_ton": delta_con_ccs_kg,
    }

def escalar_a_anual(kpi_por_ton, toneladas_ano):
    t = toneladas_ano
    return {
        "residuos_desviados_t_ano": t,
        "electricidad_neta_mwh_e_ano": kpi_por_ton["electricidad_neta_mwh_e_ton"] * t,
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
        ("Electricidad neta (exporta + / importa -)", "MWhₑ/año", kpi_ano["electricidad_neta_mwh_e_ano"], "MWhₑ/ton", kpi_ton["electricidad_neta_mwh_e_ton"]),
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

# ---------- Gráficos comparador (separados) ----------
def graficar_comparador_energia(kpis_por_modo_ton: dict):
    modos = list(kpis_por_modo_ton.keys())
    elec = [kpis_por_modo_ton[m]["electricidad_neta_mwh_e_ton"] for m in modos]
    heat = [kpis_por_modo_ton[m]["calor_util_mwh_th_ton"] for m in modos]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(modos))
    width = 0.38

    ax.bar(x - width/2, elec, width, label="Electricidad neta (MWhₑ/ton)")
    ax.bar(x + width/2, heat, width, label="Calor útil (MWhₜₕ/ton)")

    ax.set_xticks(x)
    ax.set_xticklabels(modos)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Comparador Energía (por tonelada) — A/B/C")
    ax.set_ylabel("MWh por tonelada")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def graficar_comparador_emisiones(kpis_por_modo_ton: dict):
    modos = list(kpis_por_modo_ton.keys())
    delta_ccs = [kpis_por_modo_ton[m]["delta_con_ccs_kgco2e_ton"] for m in modos]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(modos))
    ax.bar(x, delta_ccs)

    ax.set_xticks(x)
    ax.set_xticklabels(modos)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Comparador Emisiones (por tonelada) — Δ vs línea base CON CCS")
    ax.set_ylabel("kgCO₂e/ton (Δ = Boson − línea base)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

# ---------- Módulo económico ultra-compacto (sin CAPEX) ----------
def calcular_economia_ultra_compacta(kpi_ano, kpi_ton, econ):
    """
    econ:
      - tarifa_residuos_cop_ton
      - precio_electricidad_cop_kwh
      - precio_h2_cop_kg
      - precio_imbyrock_cop_ton
      - incluir_carbono (bool)
      - carbono_usd_tco2e
      - fx_cop_usd
    """
    residuos_t = kpi_ano["residuos_desviados_t_ano"]
    elec_mwh = kpi_ano["electricidad_neta_mwh_e_ano"]
    h2_export_t = kpi_ano["h2_exportable_t_ano"]
    imby_t = kpi_ano["imbyrock_t_ano"]
    delta_con_ccs_tco2e = kpi_ano["delta_con_ccs_tco2e_ano"]

    # Ingreso por servicio de tratamiento / disposición evitada (contrato por tonelada)
    ingreso_residuos = residuos_t * econ["tarifa_residuos_cop_ton"]

    # Electricidad: si exporta → ingreso; si importa → costo
    kwh = elec_mwh * 1000.0
    if kwh >= 0:
        ingreso_elec = kwh * econ["precio_electricidad_cop_kwh"]
        costo_elec = 0.0
    else:
        ingreso_elec = 0.0
        costo_elec = abs(kwh) * econ["precio_electricidad_cop_kwh"]

    # H2 exportable (solo modo B por diseño de la app)
    ingreso_h2 = (h2_export_t * 1000.0) * econ["precio_h2_cop_kg"]

    # IMBYROCK (opcional; conservador por defecto = 0)
    ingreso_imby = imby_t * econ["precio_imbyrock_cop_ton"]

    # Carbono (opcional): solo si Δ es negativo (mejor que línea base)
    ingreso_carbono = 0.0
    if econ["incluir_carbono"]:
        ahorro_tco2e = max(-delta_con_ccs_tco2e, 0.0)
        ingreso_carbono = ahorro_tco2e * econ["carbono_usd_tco2e"] * econ["fx_cop_usd"]

    ingreso_total = ingreso_residuos + ingreso_elec + ingreso_h2 + ingreso_imby + ingreso_carbono
    costo_total = costo_elec  # sin OPEX por instrucción (ultra-compacto)

    # Mezcla de ingresos (si total > 0)
    def share(x, total):
        return 100.0 * x / total if total > 0 else 0.0

    mix = {
        "Residuos": share(ingreso_residuos, ingreso_total),
        "Electricidad": share(ingreso_elec, ingreso_total),
        "H₂ exportable": share(ingreso_h2, ingreso_total),
        "IMBYROCK®": share(ingreso_imby, ingreso_total),
        "Carbono": share(ingreso_carbono, ingreso_total),
    }

    return {
        "ingreso_residuos": ingreso_residuos,
        "ingreso_elec": ingreso_elec,
        "costo_elec": costo_elec,
        "ingreso_h2": ingreso_h2,
        "ingreso_imby": ingreso_imby,
        "ingreso_carbono": ingreso_carbono,
        "ingreso_total": ingreso_total,
        "costo_total": costo_total,
        "mix_pct": mix,
        "precio_electricidad_cop_kwh": econ["precio_electricidad_cop_kwh"],
        "tarifa_residuos_cop_ton": econ["tarifa_residuos_cop_ton"],
    }

def formatear_cop(x):
    return f"{x:,.0f}".replace(",", ".")

def formatear_pct(x):
    return f"{x:.1f}%"

# =============================================
# SIDEBAR — CONFIGURACIÓN
# =============================================
with st.sidebar:
    st.markdown(f"**{CREATED_BY}**")
    st.markdown("---")

    st.subheader("🏭 Definición BEU (planta)")
    st.info(
        "FOAK / 1 BEU ≈ **36.000 t/año**.\n\n"
        "En la app: **1 BEU (planta) ≈ 2×(2 t/h) operando + 1 standby**."
    )

    st.subheader("🧪 Residuos (preset + ajuste)")
    preset = st.selectbox(
        "Preset de composición:",
        ["La Pradera (AMVA) — caso más real", "RSU municipal genérico"],
        index=0
    )

    if preset.startswith("La Pradera"):
        comp_base = {k: v.copy() for k, v in PRESET_LA_PRADERA.items()}
    else:
        comp_base = {k: v.copy() for k, v in PRESET_RSU_GENERICO.items()}

    modo_comp = st.radio(
        "Modo de composición:",
        ["Usar preset", "Personalizar porcentajes"],
        index=0
    )

    if modo_comp == "Personalizar porcentajes":
        st.caption("Ajusta los %; la app normaliza automáticamente a 100% si no coincide.")
        comp_user = {}
        for comp, d in comp_base.items():
            comp_user[comp] = d.copy()
            comp_user[comp]["pct"] = st.slider(
                comp,
                min_value=0.0,
                max_value=100.0,
                value=float(d["pct"]),
                step=0.5,
                key=f"pct_{preset}_{comp}"
            )
        composicion = normalizar_composicion(comp_user)
    else:
        composicion = normalizar_composicion(comp_base)
        with st.expander("Ver composición (% ya normalizado)"):
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
    h2_grade = st.selectbox(
        "Grado de H₂ (afecta H₂ neto y consumo eléctrico auxiliar):",
        ["Estacionario (≈95% / fast-charging)", "Movilidad (≈99.999%)"],
        index=0
    )
    h2_grade_key = "stationary" if h2_grade.startswith("Estacionario") else "mobility"

    st.subheader("📥 Capacidad")
    cap_total = st.number_input(
        "Residuos a tratar (toneladas/año):",
        min_value=1000.0,
        max_value=1_300_000.0,
        value=36000.0,
        step=1000.0,
        help="Si superas 36.000 t/año, la app interpreta un despliegue modular (N BEUs)."
    )

    st.subheader("🌍 Emisiones (supuestos editables)")
    with st.expander("Editar factores (relleno / red / transporte / CCS)"):
        PARAMS_BASE["factor_relleno_kgco2e_ton"] = st.number_input(
            "Factor relleno (kgCO₂e/ton) — línea base",
            min_value=0.0, max_value=2000.0, value=float(PARAMS_BASE["factor_relleno_kgco2e_ton"]), step=10.0
        )
        PARAMS_BASE["factor_red_tco2e_mwh"] = st.number_input(
            "Factor de emisión red (tCO₂e/MWh)",
            min_value=0.0, max_value=1.5, value=float(PARAMS_BASE["factor_red_tco2e_mwh"]), step=0.01,
            format="%.5f"
        )
        PARAMS_BASE["factor_transporte_kgco2_ton_km"] = st.number_input(
            "Transporte (kgCO₂/(ton·km))",
            min_value=0.0, max_value=1.0, value=float(PARAMS_BASE["factor_transporte_kgco2_ton_km"]), step=0.005,
            format="%.3f"
        )
        PARAMS_BASE["dist_baseline_km"] = st.number_input(
            "Distancia baseline AMVA→La Pradera (km)",
            min_value=0.0, max_value=200.0, value=float(PARAMS_BASE["dist_baseline_km"]), step=1.0
        )
        PARAMS_BASE["dist_cluster_km"] = st.number_input(
            "Distancia clúster descentralizado (km)",
            min_value=0.0, max_value=100.0, value=float(PARAMS_BASE["dist_cluster_km"]), step=1.0
        )
        PARAMS_BASE["co2_capturable_ton_por_ton"] = st.number_input(
            "CO₂ capturable del proceso (tCO₂/ton)",
            min_value=0.2, max_value=2.0, value=float(PARAMS_BASE["co2_capturable_ton_por_ton"]), step=0.05
        )
        PARAMS_BASE["ccs_captura_frac"] = st.number_input(
            "Captura CCS (fracción 0–1)",
            min_value=0.0, max_value=1.0, value=float(PARAMS_BASE["ccs_captura_frac"]), step=0.01
        )
        PARAMS_BASE["emis_indirectas_kgco2e_ton"] = st.number_input(
            "Emisiones indirectas (kgCO₂e/ton) — proxy",
            min_value=0.0, max_value=500.0, value=float(PARAMS_BASE["emis_indirectas_kgco2e_ton"]), step=5.0
        )

    st.subheader("🔌 Ruta H₂: consumos y fuel cell")
    with st.expander("Editar supuestos H₂ (consumo auxiliar / fuel cell)"):
        PARAMS_BASE["kwh_por_kg_h2_upgrading"] = st.number_input(
            "Upgrading H₂ (kWhₑ/kg H₂) — consumo eléctrico auxiliar",
            min_value=0.0, max_value=30.0, value=float(PARAMS_BASE["kwh_por_kg_h2_upgrading"]), step=0.5
        )
        PARAMS_BASE["kwh_e_por_kg_h2_fuelcell"] = st.number_input(
            "Fuel cell (kWhₑ/kg H₂) — electricidad DC entregable",
            min_value=0.0, max_value=30.0, value=float(PARAMS_BASE["kwh_e_por_kg_h2_fuelcell"]), step=0.5
        )

    st.subheader("💰 Módulo económico ultra-compacto (sin CAPEX)")
    with st.expander("Editar supuestos económicos (ingresos/costos directos)"):
        tarifa_residuos = st.number_input(
            "Tarifa por tratamiento/disposición evitada (COP/ton)",
            min_value=0.0, max_value=300000.0, value=109000.0, step=1000.0,
            help="Rango orientativo conversado: ~76k COP/ton (tarifa) hasta ~109k COP/ton (con cargos asociados)."
        )
        precio_elec = st.number_input(
            "Precio energía (COP/kWh)",
            min_value=0.0, max_value=2000.0, value=300.0, step=10.0,
            help="Proxy conversado: ~300 COP/kWh (generación, excluye T&D y comercialización)."
        )
        precio_h2 = st.number_input(
            "Precio H₂ exportable (COP/kg) — opcional",
            min_value=0.0, max_value=100000.0, value=0.0, step=500.0,
            help="Dejar en 0 si no deseas monetizar H₂ (precio depende de end-use/contrato)."
        )
        precio_imby = st.number_input(
            "Precio IMBYROCK® (COP/ton) — opcional",
            min_value=0.0, max_value=500000.0, value=0.0, step=5000.0,
            help="Conservador: 0 hasta validar QA/regulación/mercado local."
        )

        incluir_carbono = st.checkbox("Incluir ingreso por carbono (opcional)", value=False)
        carbono_usd = st.number_input(
            "Precio carbono (USD/tCO₂e)",
            min_value=0.0, max_value=500.0, value=0.0, step=1.0,
            help="Mantener en 0 si no deseas monetizar carbono."
        )
        fx_cop_usd = st.number_input(
            "Tasa de cambio (COP/USD)",
            min_value=1000.0, max_value=10000.0, value=4200.0, step=50.0
        )

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

# =============================================
# MAIN — KPI mezcla + autosuficiencia
# =============================================
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

# Despliegue modular aproximado
cap_beu = PARAMS_BASE["capacidad_beu_ton_ano"]
n_beu = int(math.ceil(cap_total / cap_beu))
cap_por_beu = cap_total / n_beu
st.info(
    f"🏗️ **Despliegue modular estimado:** **{n_beu} BEU(s)** para {cap_total:,.0f} t/año "
    f"(≈ {cap_por_beu:,.0f} t/año por BEU).".replace(",", ".")
)

st.markdown("---")
c1, c2 = st.columns([1, 1])
with c1:
    btn_calcular_modo = st.button("🚀 Calcular modo seleccionado", type="primary", use_container_width=True)
with c2:
    btn_comparar = st.button("🧭 Comparar Modos A vs B vs C (mismos supuestos)", use_container_width=True)

# =============================================
# RESULTADOS — MODO SELECCIONADO
# =============================================
if btn_calcular_modo:
    st.header("📌 Resultados — modo seleccionado")

    kpi_ton = calcular_modo_por_ton(modo_key, props, PARAMS_BASE, h2_grade=h2_grade_key)
    kpi_ano = escalar_a_anual(kpi_ton, cap_total)

    st.subheader(kpi_ton["modo"])
    df_modo = construir_tabla_modo(kpi_ton, kpi_ano)
    st.dataframe(df_modo, use_container_width=True, hide_index=True)

    st.subheader("📈 Resumen visual (anual)")
    labels = [
        "Electricidad neta\n(MWhₑ/año)",
        "H₂ neto\n(t/año)",
        "IMBYROCK®\n(t/año)",
        "Calor útil\n(MWhₜₕ/año)",
    ]
    values = [
        kpi_ano["electricidad_neta_mwh_e_ano"],
        kpi_ano["h2_total_t_ano"],
        kpi_ano["imbyrock_t_ano"],
        kpi_ano["calor_util_mwh_th_ano"],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Productos/servicios energéticos y subproductos — totales anuales")
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() if v >= 0 else 0,
            f"{v:,.0f}".replace(",", "."),
            ha="center", va="bottom" if v >= 0 else "top",
            fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧾 Interpretación rápida")
    st.markdown(
        f"- **Electricidad neta**: **{kpi_ton['electricidad_neta_mwh_e_ton']:+.3f} MWhₑ/ton** "
        f"(positivo = exporta, negativo = importa).\n"
        f"- **Consumo eléctrico auxiliar (ruta H₂)**: **{kpi_ton['consumo_aux_mwh_e_ton']:.3f} MWhₑ/ton** "
        f"(energía que hay que pagar para upgrading/pureza/compresión).\n"
        f"- **Δ vs línea base CON CCS**: **{kpi_ton['delta_con_ccs_kgco2e_ton']:+.1f} kgCO₂e/ton** "
        f"(Δ < 0 = ahorro neto vs relleno+transporte)."
    )

    # ======= ECONOMÍA ULTRA-COMPACTA (sin CAPEX) =======
    st.markdown("---")
    st.subheader("💰 Economía ultra-compacta (sin CAPEX) — ingreso bruto aproximado")

    econ = calcular_economia_ultra_compacta(kpi_ano, kpi_ton, econ_params)

    colx, coly, colz = st.columns(3)
    with colx:
        st.metric("Ingreso por residuos (COP/año)", formatear_cop(econ["ingreso_residuos"]))
        st.caption(f"Tarifa usada: {formatear_cop(econ['tarifa_residuos_cop_ton'])} COP/ton")
    with coly:
        st.metric("Ingreso por electricidad (COP/año)", formatear_cop(econ["ingreso_elec"]))
        st.metric("Costo por electricidad importada (COP/año)", formatear_cop(econ["costo_elec"]))
        st.caption(f"Precio usado: {formatear_cop(econ['precio_electricidad_cop_kwh'])} COP/kWh")
    with colz:
        st.metric("Ingreso por H₂ exportable (COP/año)", formatear_cop(econ["ingreso_h2"]))
        st.metric("Ingreso por IMBYROCK® (COP/año)", formatear_cop(econ["ingreso_imby"]))
        st.metric("Ingreso por carbono (COP/año)", formatear_cop(econ["ingreso_carbono"]))

    st.markdown(
        f"**Ingreso bruto total (COP/año)**: **{formatear_cop(econ['ingreso_total'])}**  \n"
        f"**Costo directo por electricidad importada (COP/año)**: **{formatear_cop(econ['costo_total'])}**"
    )

    df_mix = pd.DataFrame(
        [(k, v) for k, v in econ["mix_pct"].items()],
        columns=["Componente", "Participación sobre ingresos (%)"]
    )
    df_mix["Participación sobre ingresos (%)"] = df_mix["Participación sobre ingresos (%)"].apply(formatear_pct)
    st.dataframe(df_mix, use_container_width=True, hide_index=True)

    st.caption(
        "Nota de aproximación: este módulo NO incluye CAPEX ni OPEX (salvo costo de electricidad importada, cuando aplica). "
        "El objetivo es visualizar la mezcla de ingresos por modo y órdenes de magnitud para discusión ejecutiva."
    )

# =============================================
# COMPARADOR — MODOS A vs B vs C
# =============================================
if btn_comparar:
    st.header("🧭 Comparador de escenarios A vs B vs C")

    kpis_ton = {}
    kpis_ano = {}

    for mk in ["A", "B", "C"]:
        kpi_t = calcular_modo_por_ton(mk, props, PARAMS_BASE, h2_grade=h2_grade_key)
        kpi_a = escalar_a_anual(kpi_t, cap_total)
        kpis_ton[mk] = kpi_t
        kpis_ano[mk] = kpi_a

    # Tabla por tonelada
    filas_ton = [
        ("Residuos desviados (t/ton)", "t/ton", "residuos_desviados_ton_ton"),
        ("IMBYROCK® (kg/ton)", "kg/ton", "imbyrock_kg_ton"),
        ("CO₂ capturable (tCO₂/ton)", "tCO₂/ton", "co2_capturable_tco2_ton"),
        ("Electricidad neta (MWhₑ/ton)", "MWhₑ/ton", "electricidad_neta_mwh_e_ton"),
        ("Consumo eléctrico auxiliar (MWhₑ/ton)", "MWhₑ/ton", "consumo_aux_mwh_e_ton"),
        ("Calor útil (MWhₜₕ/ton)", "MWhₜₕ/ton", "calor_util_mwh_th_ton"),
        ("H₂ neto (kg/ton)", "kg/ton", "h2_neto_kg_ton"),
        ("H₂ exportable (kg/ton)", "kg/ton", "h2_exportable_kg_ton"),
        ("Línea base (kgCO₂e/ton)", "kgCO₂e/ton", "baseline_kgco2e_ton"),
        ("Boson SIN CCS (kgCO₂e/ton)", "kgCO₂e/ton", "boson_sin_ccs_kgco2e_ton"),
        ("Boson CON CCS (kgCO₂e/ton)", "kgCO₂e/ton", "boson_con_ccs_kgco2e_ton"),
        ("Δ vs baseline SIN CCS (kgCO₂e/ton)", "kgCO₂e/ton", "delta_sin_ccs_kgco2e_ton"),
        ("Δ vs baseline CON CCS (kgCO₂e/ton)", "kgCO₂e/ton", "delta_con_ccs_kgco2e_ton"),
    ]
    data_ton = []
    for nombre, unidad, key in filas_ton:
        data_ton.append([nombre, unidad, kpis_ton["A"][key], kpis_ton["B"][key], kpis_ton["C"][key]])
    df_comp_ton = pd.DataFrame(data_ton, columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"])

    st.subheader("📊 Comparación por tonelada (misma composición y supuestos)")
    st.dataframe(df_comp_ton, use_container_width=True, hide_index=True)

    # Tabla anual
    filas_ano = [
        ("Residuos desviados (t/año)", "t/año", "residuos_desviados_t_ano"),
        ("IMBYROCK® (t/año)", "t/año", "imbyrock_t_ano"),
        ("CO₂ capturable (tCO₂/año)", "tCO₂/año", "co2_capturable_tco2_ano"),
        ("Electricidad neta (MWhₑ/año)", "MWhₑ/año", "electricidad_neta_mwh_e_ano"),
        ("Consumo eléctrico auxiliar (MWhₑ/año)", "MWhₑ/año", "consumo_aux_mwh_e_ano"),
        ("Calor útil (MWhₜₕ/año)", "MWhₜₕ/año", "calor_util_mwh_th_ano"),
        ("H₂ neto (t/año)", "t H₂/año", "h2_total_t_ano"),
        ("H₂ exportable (t/año)", "t H₂/año", "h2_exportable_t_ano"),
        ("Línea base (tCO₂e/año)", "tCO₂e/año", "baseline_tco2e_ano"),
        ("Boson SIN CCS (tCO₂e/año)", "tCO₂e/año", "boson_sin_ccs_tco2e_ano"),
        ("Boson CON CCS (tCO₂e/año)", "tCO₂e/año", "boson_con_ccs_tco2e_ano"),
        ("Δ vs baseline SIN CCS (tCO₂e/año)", "tCO₂e/año", "delta_sin_ccs_tco2e_ano"),
        ("Δ vs baseline CON CCS (tCO₂e/año)", "tCO₂e/año", "delta_con_ccs_tco2e_ano"),
    ]
    data_ano = []
    for nombre, unidad, key in filas_ano:
        data_ano.append([nombre, unidad, kpis_ano["A"][key], kpis_ano["B"][key], kpis_ano["C"][key]])
    df_comp_ano = pd.DataFrame(data_ano, columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"])

    st.subheader("📅 Comparación anual (para la capacidad seleccionada)")
    st.dataframe(df_comp_ano, use_container_width=True, hide_index=True)

    # Gráficos separados
    st.subheader("📈 Comparadores gráficos (resumen)")
    fig_energy = graficar_comparador_energia(kpis_ton)
    st.pyplot(fig_energy)
    fig_emis = graficar_comparador_emisiones(kpis_ton)
    st.pyplot(fig_emis)

    # ======= ECONOMÍA ULTRA-COMPACTA (comparador) =======
    st.markdown("---")
    st.subheader("💰 Comparador económico ultra-compacto (sin CAPEX) — por modo")

    filas_econ = []
    for mk in ["A", "B", "C"]:
        econ_m = calcular_economia_ultra_compacta(kpis_ano[mk], kpis_ton[mk], econ_params)
        filas_econ.append([
            mk,
            econ_m["ingreso_total"],
            econ_m["ingreso_residuos"],
            econ_m["ingreso_elec"],
            econ_m["costo_elec"],
            econ_m["ingreso_h2"],
            econ_m["ingreso_imby"],
            econ_m["ingreso_carbono"],
        ])

    df_econ = pd.DataFrame(
        filas_econ,
        columns=[
            "Modo",
            "Ingreso bruto total (COP/año)",
            "Residuos (COP/año)",
            "Electricidad (COP/año)",
            "Costo elec importada (COP/año)",
            "H₂ exportable (COP/año)",
            "IMBYROCK® (COP/año)",
            "Carbono (COP/año)",
        ]
    )

    # Formato COP
    for col in df_econ.columns[1:]:
        df_econ[col] = df_econ[col].apply(lambda x: formatear_cop(float(x)))

    st.dataframe(df_econ, use_container_width=True, hide_index=True)

    st.caption(
        "Nota de aproximación: el comparador económico NO incluye CAPEX ni OPEX. "
        "El objetivo es visualizar órdenes de magnitud e identificar el modo con mejor mezcla de ingresos bajo supuestos dados."
    )

    # Descarga CSV
    st.subheader("⬇️ Exportar (CSV)")
    csv_ton = df_comp_ton.to_csv(index=False).encode("utf-8")
    csv_ano = df_comp_ano.to_csv(index=False).encode("utf-8")
    csv_econ = df_econ.to_csv(index=False).encode("utf-8")

    colx, coly, colz = st.columns(3)
    with colx:
        st.download_button(
            "Descargar comparación por tonelada (CSV)",
            data=csv_ton,
            file_name="comparacion_modos_por_ton.csv",
            mime="text/csv",
            use_container_width=True
        )
    with coly:
        st.download_button(
            "Descargar comparación anual (CSV)",
            data=csv_ano,
            file_name="comparacion_modos_anual.csv",
            mime="text/csv",
            use_container_width=True
        )
    with colz:
        st.download_button(
            "Descargar comparación económica (CSV)",
            data=csv_econ,
            file_name="comparacion_modos_economia.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.caption(
        "Nota de aproximación: el comparador usa la misma composición, factores de red, relleno, transporte y CCS. "
        "Δ < 0 implica ahorro neto vs línea base (relleno + transporte)."
    )

# =============================================
# FOOTER
# =============================================
st.markdown("---")
st.caption(
    "⚠️ Nota: Esta calculadora es un **modelo de aproximación** para explorar órdenes de magnitud y trade-offs por modo. "
    "No contabiliza el beneficio adicional de que el H₂ desplace diésel/gasolina o H₂ gris, porque depende del end-use."
)
st.markdown(
    f"<p style='text-align:center; font-size:12px; color:gray;'>{CREATED_BY}</p>",
    unsafe_allow_html=True
)
