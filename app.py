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
# PRESETS DE COMPOSICIÓN (PCI y H2 "teórico")
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
# PARÁMETROS DEFAULT (comunes)
# =============================================
PARAMS_DEFAULT = {
    # Capacidad BEU (FOAK) — 1 planta (no 1 reactor)
    "capacidad_beu_ton_ano": 36000.0,

    # Conversión energética
    "eficiencia_conversion_plasma": 0.78,
    "eficiencia_generacion_electrica": 0.38,
    "autoconsumo_proceso_fraction": 0.28,
    "bop_kwh_por_ton": 100.0,  # Balance of Plant

    # H2 (calibración de aproximación)
    "eficiencia_h2_desde_teorico": 0.75,  # teórico ~160 → bruto ~120 (orden de magnitud)
    "h2_neto_factor_stationary": 0.58,    # bruto ~120 → neto ~70
    "h2_neto_factor_mobility": 0.42,      # bruto ~120 → neto ~50
    "kwh_por_kg_h2_upgrading": 10.0,      # consumo eléctrico auxiliar: compresión, WGS, PSA, etc.
    "kwh_e_por_kg_h2_fuelcell": 18.0,     # electricidad DC aproximada en fuel cell (orden de magnitud)

    # Calor útil (fracción del PCI, expresado en MWh_th)
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

def anotar_barras(ax, bars, valores, dec=1):
    for b, v in zip(bars, valores):
        label = fmt_es_num(v, dec=dec, signo=True)
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

def gj_to_mwh(gj):
    # 1 MWh = 3.6 GJ  →  MWh = GJ/3.6
    return gj / 3.6

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

    pci_mwh_ton = gj_to_mwh(pci_gj_ton)
    pci_kcal_kg = pci_gj_ton * 239.0
    return {
        "pci_gj_ton": pci_gj_ton,
        "pci_mwh_ton": pci_mwh_ton,
        "pci_kcal_kg": pci_kcal_kg,
        "h2_teorico_kg_ton": h2_teorico_kg_ton,
        "fraccion_cenizas": fraccion_cenizas,
    }

def modo_defs():
    # Nota clave: en Modo C ahora permitimos (por slider) exportar una fracción del H2 neto.
    return {
        "A": {"nombre": "Modo A — Power/Heat-centric", "power_split": 1.0, "h2_split": 0.0, "h2_exporta": False},
        "B": {"nombre": "Modo B — H₂-centric",        "power_split": 0.0, "h2_split": 1.0, "h2_exporta": True},
        "C": {"nombre": "Modo C — Mixed",            "power_split": 0.5, "h2_split": 0.5, "h2_exporta": True},
    }

def calcular_modo_por_ton(
    modo_key,
    props,
    p,
    h2_grade="stationary",
    frac_h2_a_fc_B=0.0,
    frac_h2_a_fc_C=1.0,
):
    md = modo_defs()[modo_key]

    pci_gj_ton = props["pci_gj_ton"]
    pci_mwh_ton = props["pci_mwh_ton"]
    h2_teorico = props["h2_teorico_kg_ton"]

    syngas_gj_ton = pci_gj_ton * p["eficiencia_conversion_plasma"]

    # ======= RUTA POWER =======
    power_gj_ton = syngas_gj_ton * md["power_split"]
    autoconsumo_proceso_gj_ton = power_gj_ton * p["autoconsumo_proceso_fraction"]
    neto_power_gj_ton = power_gj_ton - autoconsumo_proceso_gj_ton

    # ⚠️ Unidad correcta:
    # neto_power_gj_ton (GJ/ton) → MWh/ton = GJ/3.6
    electricidad_syngas_mwh_ton = gj_to_mwh(neto_power_gj_ton) * p["eficiencia_generacion_electrica"]

    # ======= RUTA H2 =======
    h2_bruto_kg_ton = h2_teorico * p["eficiencia_h2_desde_teorico"] * md["h2_split"]

    if h2_grade == "mobility":
        h2_neto_kg_ton = h2_bruto_kg_ton * p["h2_neto_factor_mobility"]
    else:
        h2_neto_kg_ton = h2_bruto_kg_ton * p["h2_neto_factor_stationary"]

    # Consumo eléctrico auxiliar asociado a upgrading del H2 neto (orden de magnitud).
    consumo_aux_mwh_ton = (h2_neto_kg_ton * p["kwh_por_kg_h2_upgrading"]) / 1000.0

    # ======= Split H2: Fuel-Cell vs Export =======
    if modo_key == "B":
        frac_h2_a_fc = max(0.0, min(1.0, frac_h2_a_fc_B))
    elif modo_key == "C":
        frac_h2_a_fc = max(0.0, min(1.0, frac_h2_a_fc_C))
    else:
        frac_h2_a_fc = 0.0

    # Electricidad DC desde Fuel-Cell (subcomponente, ya incluido en electricidad neta)
    electricidad_fuelcell_mwh_ton = (h2_neto_kg_ton * frac_h2_a_fc * p["kwh_e_por_kg_h2_fuelcell"]) / 1000.0

    # H2 exportable (solo para modos que lo habilitan)
    h2_exportable_kg_ton = 0.0
    if md["h2_exporta"]:
        h2_exportable_kg_ton = h2_neto_kg_ton * (1.0 - frac_h2_a_fc)

    # BOP (consumo fijo por tonelada)
    bop_mwh_ton = p["bop_kwh_por_ton"] / 1000.0

    # Electricidad neta del sistema (exporta + / importa -)
    electricidad_neta_mwh_ton = (
        electricidad_syngas_mwh_ton
        + electricidad_fuelcell_mwh_ton
        - bop_mwh_ton
        - consumo_aux_mwh_ton
    )

    # Calor útil (MWh_th/ton)
    if modo_key == "A":
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_A"]
    elif modo_key == "B":
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_B"]
    else:
        calor_util_mwhth_ton = pci_mwh_ton * p["heat_fraction_C"]

    imbyrock_kg_ton = p["imbyrock_kg_ton"]
    metales_kg_ton = p["metales_kg_ton"]
    co2_capturable_ton_ton = p["co2_capturable_ton_por_ton"]

    # ======= EMISIONES (baseline vs Boson) =======
    baseline_kg = p["factor_relleno_kgco2e_ton"] + p["factor_transporte_kgco2_ton_km"] * p["dist_baseline_km"]
    transporte_cluster_kg = p["factor_transporte_kgco2_ton_km"] * p["dist_cluster_km"]

    grid_kg_por_mwh = p["factor_red_tco2e_mwh"] * 1000.0
    efecto_electricidad_kg = -electricidad_neta_mwh_ton * grid_kg_por_mwh  # export (-) evita; import (+) carga

    proceso_sin_ccs_kg = co2_capturable_ton_ton * 1000.0
    proceso_con_ccs_kg = proceso_sin_ccs_kg * (1.0 - p["ccs_captura_frac"])

    indirectas_kg = p["emis_indirectas_kgco2e_ton"]

    boson_sin_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_sin_ccs_kg + indirectas_kg
    boson_con_ccs_kg = transporte_cluster_kg + efecto_electricidad_kg + proceso_con_ccs_kg + indirectas_kg

    delta_sin_ccs_kg = boson_sin_ccs_kg - baseline_kg
    delta_con_ccs_kg = boson_con_ccs_kg - baseline_kg

    return {
        "modo": md["nombre"],
        "residuos_desviados_ton_ton": 1.0,

        "pci_gj_ton": pci_gj_ton,
        "pci_mwh_ton": pci_mwh_ton,

        "electricidad_neta_mwh_e_ton": electricidad_neta_mwh_ton,
        "consumo_aux_mwh_e_ton": consumo_aux_mwh_ton,

        # Subcomponente (NO sumar aparte)
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
    }

def escalar_a_anual(kpi_por_ton, toneladas_ano):
    t = toneladas_ano
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

def graf_comparador_simple(titulo, ylabel, modos, vals, dec=1):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(modos))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(modos)
    ax.axhline(0, linewidth=0.8)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    anotar_barras(ax, bars, vals, dec=dec)
    plt.tight_layout()
    return fig

# ---------- Módulo económico ultra-compacto (sin CAPEX) ----------
def calcular_economia_ultra_compacta(kpi_ano, econ):
    residuos_t = kpi_ano["residuos_desviados_t_ano"]
    elec_mwh = kpi_ano["electricidad_neta_mwh_e_ano"]
    h2_export_t = kpi_ano["h2_exportable_t_ano"]
    imby_t = kpi_ano["imbyrock_t_ano"]
    delta_con_ccs_tco2e = kpi_ano["delta_con_ccs_tco2e_ano"]

    ingreso_residuos = residuos_t * econ["tarifa_residuos_cop_ton"]

    # Separación: precio export vs precio import
    kwh = elec_mwh * 1000.0
    if kwh >= 0:
        ingreso_elec = kwh * econ["precio_electricidad_export_cop_kwh"]
        costo_elec = 0.0
    else:
        ingreso_elec = 0.0
        costo_elec = abs(kwh) * econ["precio_electricidad_import_cop_kwh"]

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
        ("Ingresos", "Ingreso por electricidad exportada (MCOP/año)", "ingreso_elec"),
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

# =============================================
# SIDEBAR — CONFIGURACIÓN
# =============================================
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
        ["La Pradera (AMVA) — caso más real", "RSU municipal genérico"],
        index=0
    )

    if preset.startswith("La Pradera"):
        comp_base = {k: v.copy() for k, v in PRESET_LA_PRADERA.items()}
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
    h2_grade = st.selectbox("Grado de H₂:", ["Estacionario (≈95% / fast-charging)", "Movilidad (≈99.999%)"], index=0)
    h2_grade_key = "stationary" if h2_grade.startswith("Estacionario") else "mobility"

    # ---- Split H2 para modos B y C (lo pediste explícito) ----
    st.subheader("🔁 Ruta interna del H₂ (modos B y C)")
    with st.expander("Configurar split H₂ → Fuel-Cell (DC) vs H₂ exportable", expanded=True):
        frac_h2_a_fc_B_pct = st.slider(
            "Modo B — Fracción de H₂ a Fuel-Cell (0–100%)",
            min_value=0.0, max_value=100.0, value=0.0, step=5.0,
            help="0% = todo el H₂ neto exportable • 100% = todo el H₂ neto se convierte a electricidad DC (sin exportar H₂)."
        )
        st.caption("Modo B: ✅ 0% = todo exportable • ✅ 100% = todo a DC (Fuel-Cell).")

        frac_h2_a_fc_C_pct = st.slider(
            "Modo C — Fracción de H₂ a Fuel-Cell (0–100%)",
            min_value=0.0, max_value=100.0, value=100.0, step=5.0,
            help="En Modo C, el H₂ neto del 'split' puede ir a DC (Fuel-Cell) o exportarse. 100% replica el supuesto previo (todo a DC)."
        )
        st.caption("Modo C: ✅ 0% = H₂ neto exportable • ✅ 100% = H₂ neto a DC (Fuel-Cell).")

    st.subheader("📥 Capacidad")
    cap_total = st.number_input(
        "Residuos a tratar (toneladas/año):",
        min_value=1000.0, max_value=1_300_000.0, value=36000.0, step=1000.0
    )

    st.subheader("🌍 Emisiones (supuestos editables)")
    with st.expander("Editar factores (relleno / red / transporte / CCS)"):
        factor_relleno = st.number_input(
            "Factor relleno (kgCO₂e/ton) — línea base", min_value=0.0, max_value=2000.0,
            value=float(PARAMS_DEFAULT["factor_relleno_kgco2e_ton"]), step=10.0
        )
        factor_red = st.number_input(
            "Factor de emisión red (tCO₂e/MWh)", min_value=0.0, max_value=1.5,
            value=float(PARAMS_DEFAULT["factor_red_tco2e_mwh"]), step=0.01, format="%.5f"
        )
        factor_transporte = st.number_input(
            "Transporte (kgCO₂/(ton·km))", min_value=0.0, max_value=1.0,
            value=float(PARAMS_DEFAULT["factor_transporte_kgco2_ton_km"]), step=0.005, format="%.3f"
        )
        dist_baseline = st.number_input(
            "Distancia baseline AMVA→La Pradera (km)", min_value=0.0, max_value=200.0,
            value=float(PARAMS_DEFAULT["dist_baseline_km"]), step=1.0
        )
        dist_cluster = st.number_input(
            "Distancia clúster descentralizado (km)", min_value=0.0, max_value=100.0,
            value=float(PARAMS_DEFAULT["dist_cluster_km"]), step=1.0
        )
        co2_capturable = st.number_input(
            "CO₂ capturable del proceso (tCO₂/ton)", min_value=0.2, max_value=2.0,
            value=float(PARAMS_DEFAULT["co2_capturable_ton_por_ton"]), step=0.05
        )
        ccs_frac = st.number_input(
            "Captura CCS (fracción 0–1)", min_value=0.0, max_value=1.0,
            value=float(PARAMS_DEFAULT["ccs_captura_frac"]), step=0.01
        )
        emis_ind = st.number_input(
            "Emisiones indirectas (kgCO₂e/ton) — proxy", min_value=0.0, max_value=500.0,
            value=float(PARAMS_DEFAULT["emis_indirectas_kgco2e_ton"]), step=5.0
        )

    st.subheader("🔌 Ruta H₂: consumos y Fuel-Cell")
    with st.expander("Editar supuestos H₂ (consumo auxiliar / Fuel-Cell)"):
        kwh_kg_h2_up = st.number_input(
            "Upgrading H₂ (kWhₑ/kg H₂) — consumo eléctrico auxiliar",
            min_value=0.0, max_value=30.0, value=float(PARAMS_DEFAULT["kwh_por_kg_h2_upgrading"]), step=0.5
        )
        kwh_kg_h2_fc = st.number_input(
            "Fuel-Cell (kWhₑ/kg H₂) — electricidad DC entregable",
            min_value=0.0, max_value=30.0, value=float(PARAMS_DEFAULT["kwh_e_por_kg_h2_fuelcell"]), step=0.5
        )

    st.subheader("💰 Módulo económico ultra-compacto (sin CAPEX)")
    with st.expander("Editar supuestos económicos (ingresos/costos directos)"):
        tarifa_residuos = st.number_input(
            "Tarifa por tratamiento/disposición evitada (COP/ton)",
            min_value=0.0, max_value=300000.0, value=109000.0, step=1000.0
        )

        # ✅ Separación export/import (pedido)
        precio_elec_export = st.number_input(
            "Precio electricidad EXPORTADA (COP/kWh)",
            min_value=0.0, max_value=2000.0, value=300.0, step=10.0
        )
        precio_elec_import = st.number_input(
            "Precio electricidad IMPORTADA (COP/kWh)",
            min_value=0.0, max_value=4000.0, value=300.0, step=10.0
        )
        st.caption("Nota: si el sistema importa electricidad (electricidad neta < 0), se costea a **precio IMPORT**; si exporta (>0), se ingresa a **precio EXPORT**.")

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

    # Construir params efectivos (evita “mutar” defaults)
    PARAMS = PARAMS_DEFAULT.copy()
    PARAMS.update({
        "factor_relleno_kgco2e_ton": float(factor_relleno),
        "factor_red_tco2e_mwh": float(factor_red),
        "factor_transporte_kgco2_ton_km": float(factor_transporte),
        "dist_baseline_km": float(dist_baseline),
        "dist_cluster_km": float(dist_cluster),
        "co2_capturable_ton_por_ton": float(co2_capturable),
        "ccs_captura_frac": float(ccs_frac),
        "emis_indirectas_kgco2e_ton": float(emis_ind),
        "kwh_por_kg_h2_upgrading": float(kwh_kg_h2_up),
        "kwh_e_por_kg_h2_fuelcell": float(kwh_kg_h2_fc),
    })

    econ_params = {
        "tarifa_residuos_cop_ton": float(tarifa_residuos),
        "precio_electricidad_export_cop_kwh": float(precio_elec_export),
        "precio_electricidad_import_cop_kwh": float(precio_elec_import),
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

cap_beu = PARAMS["capacidad_beu_ton_ano"]
n_beu = int(math.ceil(cap_total / cap_beu))
cap_por_beu = cap_total / n_beu
st.info(
    f"🏗️ **Despliegue modular estimado:** **{n_beu} BEU(s)** para {fmt_es_num(cap_total,0)} t/año "
    f"(≈ {fmt_es_num(cap_por_beu,0)} t/año por BEU)."
)

st.markdown("---")

# ✅ Para evitar “congelamientos”: toggles que se mantienen al mover sliders
col1, col2 = st.columns([1, 1])
with col1:
    mostrar_modo = st.toggle("📌 Mostrar resultados del modo seleccionado", value=True)
with col2:
    mostrar_comparador = st.toggle("🧭 Mostrar comparador A vs B vs C (mismos supuestos)", value=True)

frac_B = float(frac_h2_a_fc_B_pct) / 100.0
frac_C = float(frac_h2_a_fc_C_pct) / 100.0

# =============================================
# RESULTADOS — MODO SELECCIONADO
# =============================================
if mostrar_modo:
    st.header("📌 Resultados — modo seleccionado")

    kpi_ton = calcular_modo_por_ton(
        modo_key,
        props,
        PARAMS,
        h2_grade=h2_grade_key,
        frac_h2_a_fc_B=frac_B,
        frac_h2_a_fc_C=frac_C,
    )
    kpi_ano = escalar_a_anual(kpi_ton, cap_total)

    st.subheader(kpi_ton["modo"])
    df_modo = construir_tabla_modo(kpi_ton, kpi_ano)

    df_show = df_modo.copy()
    df_show["Total anual"] = df_show.apply(
        lambda r: fmt_es_num(r["Total anual"], dec=0, signo=("Δ" in r["Indicador"])), axis=1
    )
    df_show["Por tonelada"] = df_show.apply(
        lambda r: fmt_es_num(r["Por tonelada"], dec=2 if "MWh" in r["Unidad (por ton)"] else 1, signo=("Δ" in r["Indicador"])),
        axis=1
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.info(
        "Guía: **“Electricidad DC bruta vía Fuel-Cell”** es un **subcomponente** ya incluido en "
        "**“Electricidad neta del sistema”**. Se muestra para visualizar el retorno eléctrico del H₂, "
        "pero **no** debe sumarse como energía adicional."
    )

    st.caption(
        "Nota técnica: el **consumo eléctrico auxiliar** de la ruta H₂ se aplica al H₂ neto (orden de magnitud). "
        "En futuras iteraciones puede diferenciarse por grado/pureza o por si el H₂ se exporta vs se usa localmente."
    )

# =============================================
# COMPARADOR — MODOS A vs B vs C
# =============================================
if mostrar_comparador:
    st.header("🧭 Comparador de escenarios A vs B vs C")

    kpis_ton = {}
    kpis_ano = {}
    econ_res = {}

    for mk in ["A", "B", "C"]:
        kpi_t = calcular_modo_por_ton(
            mk, props, PARAMS,
            h2_grade=h2_grade_key,
            frac_h2_a_fc_B=frac_B,
            frac_h2_a_fc_C=frac_C,
        )
        kpi_a = escalar_a_anual(kpi_t, cap_total)
        kpis_ton[mk] = kpi_t
        kpis_ano[mk] = kpi_a
        econ_res[mk] = calcular_economia_ultra_compacta(kpi_a, econ_params)

    st.subheader("📌 Comparación anual (resultados directos respecto a toneladas/año)")
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
    df_comp_anual = pd.DataFrame(
        [[n, u, kpis_ano["A"][k], kpis_ano["B"][k], kpis_ano["C"][k]] for n, u, k in filas_anual],
        columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"]
    )
    st.dataframe(formatear_tabla_anual(df_comp_anual), use_container_width=True, hide_index=True)

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
    df_comp_ton = pd.DataFrame(
        [[n, u, kpis_ton["A"][k], kpis_ton["B"][k], kpis_ton["C"][k]] for n, u, k in filas_ton],
        columns=["Indicador", "Unidad", "Modo A", "Modo B", "Modo C"]
    )
    st.dataframe(formatear_tabla_ton(df_comp_ton), use_container_width=True, hide_index=True)

    st.info(
        "Lectura rápida:\n"
        "- Tabla **por tonelada**: normalizada (por eso **Residuos desviados = 1,0 t/ton**).\n"
        "- Tabla **anual**: valor directo respecto a toneladas/año.\n"
        "- **Electricidad DC vía Fuel-Cell** es **subcomponente** (ya incluido en **Electricidad neta**)."
    )

    st.subheader("📈 Comparadores gráficos (resumen)")
    modos = ["A", "B", "C"]

    st.pyplot(
        graf_comparador_simple(
            "Comparador — Electricidad neta del sistema (por tonelada)",
            "MWhₑ/ton",
            modos,
            [kpis_ton[m]["electricidad_neta_mwh_e_ton"] for m in modos],
            dec=2
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Calor útil recuperable (por tonelada)",
            "MWhₜₕ/ton",
            modos,
            [kpis_ton[m]["calor_util_mwh_th_ton"] for m in modos],
            dec=2
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Emisiones (por tonelada): Δ vs línea base SIN CCS",
            "kgCO₂e/ton  (Δ = Boson − línea base)",
            modos,
            [kpis_ton[m]["delta_sin_ccs_kgco2e_ton"] for m in modos],
            dec=1
        )
    )
    st.pyplot(
        graf_comparador_simple(
            "Comparador — Emisiones (por tonelada): Δ vs línea base CON CCS",
            "kgCO₂e/ton  (Δ = Boson − línea base)",
            modos,
            [kpis_ton[m]["delta_con_ccs_kgco2e_ton"] for m in modos],
            dec=1
        )
    )

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
