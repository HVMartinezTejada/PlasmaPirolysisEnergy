import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# =============================================
# CONFIGURACIÓN DE PÁGINA
# =============================================
st.set_page_config(page_title="Calculadora Boson Energy", layout="wide")
st.title("⚡ Calculadora de Balance y Emisiones - Tecnología Boson")
st.markdown("---")

# =============================================
# CRÉDITOS (requerido)
# =============================================
CREATED_BY_HTML = (
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "Creado por: H. Vladimir Martínez-T &lt;hader.martinez@upb.edu.co&gt; — NDA Boson Energy-UPB 2025"
    "</p>"
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "Created by: H. Vladimir Martínez-T &lt;hader.martinez@upb.edu.co&gt; NDA Boson Energy-UPB 2025"
    "</p>"
)

# =============================================
# NOMBRES DE COLUMNAS (evita KeyError)
# =============================================
COL_INDICADOR = "Corriente de salida / Indicador"
COL_TOTAL = "Cantidad Anual Total"
COL_POR_TON = "Por Tonelada de Residuo"
COL_NOTA = "Condición Técnica / Nota"

def obtener_col(df: pd.DataFrame, candidatos: list[str]) -> str:
    """Devuelve el primer nombre de columna existente en df, o levanta error claro."""
    for c in candidatos:
        if c in df.columns:
            return c
    raise KeyError(f"No se encontró ninguna de estas columnas: {candidatos}. Columnas disponibles: {list(df.columns)}")

# =============================================
# CONSTANTES DIDÁCTICAS
# =============================================
LHV_H2_KWH_PER_KG = 33.33
LHV_H2_GJ_PER_KG = LHV_H2_KWH_PER_KG * 0.0036
GJ_PER_MWH = 3.6

UMBRAL_AUTOSUF_GJ_TON = 9.0
UMBRAL_CASI_AUTOSUF_GJ_TON = 7.0

EF_GRID_DEFAULT_TCO2E_MWH = 0.21742
EF_TRANSPORT_DEFAULT_KG_TKM = 0.127
EF_LANDFILL_WARM_KG_TON = 640.0
EF_LANDFILL_CO_PROXY_KG_TON = 880.0
GWP20_MULTIPLIER_DEFAULT = 2.84

# =============================================
# COMPOSICIÓN DE RESIDUOS (presets)
# =============================================
RSU_MUNICIPAL_BASE = {
    'plasticos': {'pct': 12.0, 'pci_gj_ton': 35.0, 'cenizas_pct': 5},
    'organicos': {'pct': 45.0, 'pci_gj_ton': 5.0,  'cenizas_pct': 15},
    'papel_carton': {'pct': 18.0, 'pci_gj_ton': 16.0, 'cenizas_pct': 8},
    'textiles': {'pct': 4.0,  'pci_gj_ton': 20.0, 'cenizas_pct': 10},
    'madera': {'pct': 3.0,  'pci_gj_ton': 18.0, 'cenizas_pct': 5},
    'otros_combustibles': {'pct': 5.0,  'pci_gj_ton': 25.0, 'cenizas_pct': 20},
    'inertes_metales': {'pct': 10.0, 'pci_gj_ton': 0.0,  'cenizas_pct': 95},
    'otros_inertes': {'pct': 3.0,  'pci_gj_ton': 0.0,  'cenizas_pct': 90}
}

LA_PRADERA_PRESET = {
    'plasticos': {'pct': 16.0, 'pci_gj_ton': 35.0, 'cenizas_pct': 5},
    'organicos': {'pct': 46.0, 'pci_gj_ton': 5.0,  'cenizas_pct': 15},
    'papel_carton': {'pct': 18.0, 'pci_gj_ton': 16.0, 'cenizas_pct': 8},
    'textiles': {'pct': 3.9,  'pci_gj_ton': 20.0, 'cenizas_pct': 10},
    'madera': {'pct': 2.0,  'pci_gj_ton': 18.0, 'cenizas_pct': 5},
    'otros_combustibles': {'pct': 10.0,  'pci_gj_ton': 25.0, 'cenizas_pct': 20},
    'inertes_metales': {'pct': 4.0, 'pci_gj_ton': 0.0,  'cenizas_pct': 95},
    'otros_inertes': {'pct': 0.1,  'pci_gj_ton': 0.0,  'cenizas_pct': 90}
}

# =============================================
# HELPERS
# =============================================
def extraer_numero(celda):
    if isinstance(celda, (int, float, np.number)):
        return float(celda)
    txt = str(celda).replace('\xa0', ' ').replace(',', '')
    m = re.search(r'([+-]?\d+(?:\.\d+)?)', txt)
    return float(m.group(1)) if m else 0.0

def clamp(x, a, b):
    return max(a, min(b, x))

# =============================================
# PARÁMETROS POR MODO
# =============================================
def defaults_por_modo(modo_operacion: str, h2_calidad: str):
    base = {
        "capacidad_foak_ton_año": 36000,
        "autoconsumo_bop_kwh_ton": 100,
        "factor_captura_ccs": 0.85,
        "emis_indirectas_kgco2e_ton": 100.0,
        "co2_capturable_kg_ton": 900.0,
        "metales_kg_ton": 10.0,
        "heat_recov_power": 0.80,
        "heat_recov_fc": 0.80,
        "heat_recov_process": 0.60,
        "heat_recov_plasma_loss": 0.40,
        "eta_fc_el": 0.55,
    }

    if modo_operacion == "Modo A — Power/Heat-centric":
        base.update({
            "eff_plasma": 0.90,
            "autoconsumo_syngas_frac": 0.18,
            "eta_el_direct": 0.45,
            "eta_syngas_to_h2_lhv": 0.00,
            "kwh_per_kg_h2_parasitic": 0.0,
        })
    elif modo_operacion == "Modo B — H₂-centric":
        if h2_calidad.startswith("Movilidad"):
            eta_h2 = 0.68
            kwh_per_kg = 12.0
        else:
            eta_h2 = 0.85
            kwh_per_kg = 10.0
        base.update({
            "eff_plasma": 0.90,
            "autoconsumo_syngas_frac": 0.18,
            "eta_el_direct": 0.00,
            "eta_syngas_to_h2_lhv": eta_h2,
            "kwh_per_kg_h2_parasitic": kwh_per_kg,
        })
    else:  # Modo C — Mixed
        if h2_calidad.startswith("Movilidad"):
            eta_h2 = 0.68
            kwh_per_kg = 12.0
        else:
            eta_h2 = 0.85
            kwh_per_kg = 10.0
        base.update({
            "eff_plasma": 0.90,
            "autoconsumo_syngas_frac": 0.18,
            "eta_el_direct": 0.45,
            "eta_syngas_to_h2_lhv": eta_h2,
            "kwh_per_kg_h2_parasitic": kwh_per_kg,
        })
    return base

# =============================================
# KPI POR TONELADA
# =============================================
def calcular_kpis_por_ton(
    pci_gj_ton: float,
    ash_frac: float,
    params_modo: dict,
    split_syngas_a_h2: float,
    usar_fuel_cell: bool,
    frac_h2_a_fc: float
):
    E_in_gj = pci_gj_ton
    E_syngas_gj = E_in_gj * params_modo["eff_plasma"]

    E_autoconsumo_gj = E_syngas_gj * params_modo["autoconsumo_syngas_frac"]
    E_disp_gj = max(E_syngas_gj - E_autoconsumo_gj, 0.0)

    f_h2 = clamp(split_syngas_a_h2, 0.0, 1.0)
    f_power = 1.0 - f_h2

    H2_lhv_gj = E_disp_gj * f_h2 * params_modo["eta_syngas_to_h2_lhv"]
    H2_kg = H2_lhv_gj / LHV_H2_GJ_PER_KG if LHV_H2_GJ_PER_KG > 0 else 0.0

    parasitic_mwh = (H2_kg * params_modo["kwh_per_kg_h2_parasitic"]) / 1000.0

    el_direct_gj = E_disp_gj * f_power * params_modo["eta_el_direct"]
    el_direct_mwh = el_direct_gj / GJ_PER_MWH

    frac_fc = clamp(frac_h2_a_fc, 0.0, 1.0) if usar_fuel_cell else 0.0
    el_fc_gj = H2_lhv_gj * frac_fc * params_modo["eta_fc_el"]
    el_fc_mwh = el_fc_gj / GJ_PER_MWH

    H2_export_kg = H2_kg * (1.0 - frac_fc)
    H2_fc_kg = H2_kg * frac_fc

    bop_mwh = params_modo["autoconsumo_bop_kwh_ton"] / 1000.0
    el_neta_mwh = el_direct_mwh + el_fc_mwh - parasitic_mwh - bop_mwh

    heat_from_power_gj = (E_disp_gj * f_power * (1.0 - params_modo["eta_el_direct"])
                          * params_modo["heat_recov_power"]) if params_modo["eta_el_direct"] > 0 else 0.0
    heat_from_fc_gj = (H2_lhv_gj * frac_fc * (1.0 - params_modo["eta_fc_el"])
                       * params_modo["heat_recov_fc"]) if usar_fuel_cell else 0.0
    heat_from_process_gj = E_autoconsumo_gj * params_modo["heat_recov_process"]
    heat_from_plasma_loss_gj = E_in_gj * (1.0 - params_modo["eff_plasma"]) * params_modo["heat_recov_plasma_loss"]

    heat_util_gj = heat_from_power_gj + heat_from_fc_gj + heat_from_process_gj + heat_from_plasma_loss_gj
    heat_util_mwhth = heat_util_gj / GJ_PER_MWH  # ✅ MWhₜₕ/ton

    escoria_kg = ash_frac * 1000.0 * 0.90
    escoria_kg = clamp(escoria_kg, 100.0, 120.0)

    return {
        "E_syngas_gj_ton": E_syngas_gj,
        "H2_kg_ton": H2_kg,
        "H2_export_kg_ton": H2_export_kg,
        "H2_fc_kg_ton": H2_fc_kg,
        "el_direct_mwh_ton": el_direct_mwh,
        "el_fc_mwh_ton": el_fc_mwh,
        "el_parasitic_mwh_ton": parasitic_mwh,
        "el_bop_mwh_ton": bop_mwh,
        "el_neta_mwh_ton": el_neta_mwh,
        "heat_util_mwhth_ton": heat_util_mwhth,
        "escoria_kg_ton": escoria_kg
    }

# =============================================
# EMISIONES POR TONELADA
# =============================================
def calcular_emisiones_por_ton(
    kpis_ton: dict,
    params_modo: dict,
    EF_landfill_kg_ton: float,
    EF_grid_tco2e_mwh: float,
    EF_transport_kg_tkm: float,
    dist_baseline_km: float,
    dist_cluster_km: float
):
    EF_grid_kg_mwh = EF_grid_tco2e_mwh * 1000.0

    baseline_landfill_kg = EF_landfill_kg_ton
    baseline_transport_kg = dist_baseline_km * EF_transport_kg_tkm
    baseline_total_kg = baseline_landfill_kg + baseline_transport_kg

    boson_transport_kg = dist_cluster_km * EF_transport_kg_tkm
    boson_grid_kg = -kpis_ton["el_neta_mwh_ton"] * EF_grid_kg_mwh

    direct_no_ccs_kg = params_modo["co2_capturable_kg_ton"]
    direct_with_ccs_kg = direct_no_ccs_kg * (1.0 - params_modo["factor_captura_ccs"])
    indirect_kg = params_modo["emis_indirectas_kgco2e_ton"]

    boson_total_no_ccs_kg = boson_transport_kg + boson_grid_kg + direct_no_ccs_kg + indirect_kg
    boson_total_with_ccs_kg = boson_transport_kg + boson_grid_kg + direct_with_ccs_kg + indirect_kg

    huella_neta_no_ccs_kg = boson_total_no_ccs_kg - baseline_total_kg
    huella_neta_with_ccs_kg = boson_total_with_ccs_kg - baseline_total_kg

    beneficio_landfill_kg = baseline_landfill_kg
    beneficio_transport_kg = baseline_transport_kg - boson_transport_kg
    efecto_electricidad_kg = kpis_ton["el_neta_mwh_ton"] * EF_grid_kg_mwh

    return {
        "baseline_total_kg": baseline_total_kg,
        "boson_total_no_ccs_kg": boson_total_no_ccs_kg,
        "boson_total_with_ccs_kg": boson_total_with_ccs_kg,
        "huella_neta_no_ccs_kg": huella_neta_no_ccs_kg,
        "huella_neta_with_ccs_kg": huella_neta_with_ccs_kg,
        "beneficio_landfill_kg": beneficio_landfill_kg,
        "beneficio_transport_kg": beneficio_transport_kg,
        "efecto_electricidad_kg": efecto_electricidad_kg,
        "direct_no_ccs_kg": direct_no_ccs_kg,
        "indirect_kg": indirect_kg,
    }

# =============================================
# VISUALIZACIÓN (corregida: usa nombres reales de columnas)
# =============================================
def visualizar_balance_unificado(df_unificado: pd.DataFrame, capacidad_total: float):
    col_ind = obtener_col(df_unificado, [COL_INDICADOR, "Corriente de Salida / Indicador"])
    col_total = obtener_col(df_unificado, [COL_TOTAL])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Balance integral — Capacidad total: {capacidad_total:,.0f} ton/año',
        fontsize=14,
        fontweight='bold'
    )

    # 1) Productos principales (unidades por barra)
    productos = [
        ("Hidrógeno (H₂) Exportable", "H₂ exportable\n(t/año)"),
        ("Electricidad Neta", "Electricidad neta\n(MWhₑ/año)"),
        ("Escoria vitrificada", "Escoria vitrificada\n(t/año)"),
        ("Calor Útil", "Calor útil\n(MWhₜₕ/año)")
    ]

    etiquetas, valores = [], []
    for search_term, display_label in productos:
        fila = df_unificado[df_unificado[col_ind].str.contains(search_term, regex=False)]
        etiquetas.append(display_label)
        valores.append(extraer_numero(fila.iloc[0][col_total]) if not fila.empty else 0)

    bars1 = axes[0, 0].bar(etiquetas, valores)
    axes[0, 0].set_title('1. Productos principales (unidades por barra)', fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad anual (no comparable entre barras por unidad)')
    for bar, val in zip(bars1, valores):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{val:,.0f}' if abs(val) >= 1000 else f'{val:,.2f}',
            ha='center', va='bottom', fontweight='bold'
        )

    # 2) Huella neta (t CO2e/año)
    fila_sin = df_unificado[df_unificado[col_ind].str.contains("Huella neta SIN CCS", regex=False)]
    fila_con = df_unificado[df_unificado[col_ind].str.contains("Huella neta CON CCS", regex=False)]
    huella_sin = extraer_numero(fila_sin.iloc[0][col_total]) if not fila_sin.empty else 0.0
    huella_con = extraer_numero(fila_con.iloc[0][col_total]) if not fila_con.empty else 0.0

    categorias_emis = ['Huella neta\n(SIN CCS)', 'Huella neta\n(CON CCS)']
    valores_emis = [huella_sin, huella_con]
    bars2 = axes[0, 1].bar(categorias_emis, valores_emis)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_title('2. Huella neta vs línea base (t CO2e/año)\n(negativo = ahorro)', fontweight='bold')
    axes[0, 1].set_ylabel('t CO2e por año')
    for bar, val in zip(bars2, valores_emis):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2 if val != 0 else 0,
            f'{val:+,.0f} t',
            ha='center', va='center', fontweight='bold', fontsize=10
        )

    # 3) Componentes (t CO2e/año)
    componentes = [
        "Beneficio: Relleno sanitario evitado",
        "Beneficio: Transporte evitado",
        "Efecto: Electricidad (exporta/importa)",
        "Emisiones: Proceso (SIN CCS)",
        "Emisiones: Indirectas"
    ]
    vals = []
    labels = []
    for comp in componentes:
        fila = df_unificado[df_unificado[col_ind].str.contains(comp, regex=False)]
        if not fila.empty:
            labels.append(comp.replace("Beneficio: ", "").replace("Efecto: ", "").replace("Emisiones: ", ""))
            vals.append(extraer_numero(fila.iloc[0][col_total]))

    if labels:
        axes[1, 0].barh(labels, vals)
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('3. Componentes (t CO2e/año)\n(+ beneficio, − penalidad)', fontweight='bold')
        axes[1, 0].set_xlabel('t CO2e por año')

    # 4) Tabla rápida KPIs por tonelada
    col_por_ton = obtener_col(df_unificado, [COL_POR_TON])
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    filas_resumen = df_unificado[df_unificado[col_ind].str.contains(
        "Residuos desviados|Hidrógeno|Electricidad Neta|Calor Útil|Escoria", regex=True
    )].head(6)

    tabla_data = filas_resumen[[col_ind, col_por_ton]].values
    if len(tabla_data) > 0:
        tabla = axes[1, 1].table(
            cellText=tabla_data,
            colLabels=['Indicador', 'Por tonelada'],
            loc='center',
            cellLoc='left'
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(8)
        tabla.scale(1.2, 1.8)
        axes[1, 1].set_title('4. KPIs por tonelada (vista rápida)', fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

# =============================================
# BARRA LATERAL
# =============================================
with st.sidebar:
    st.markdown(CREATED_BY_HTML, unsafe_allow_html=True)
    st.markdown("---")

    st.header("⚙️ Modo de operación (BEU)")
    modo_operacion = st.selectbox(
        "Selecciona el modo:",
        ["Modo A — Power/Heat-centric", "Modo B — H₂-centric", "Modo C — Mixed"],
        index=2
    )

    st.caption("Una planta BEU es un arreglo de 1–3 cámaras/reactores según la oferta de residuos.")
    # st.caption("A BEU plant is a setup of 1–3 reactor chambers depending on waste supply.")

    if modo_operacion == "Modo A — Power/Heat-centric":
        st.info("📌 Enfocado en **electricidad neta** y **calor útil**. H₂ por defecto = 0 (evita doble conteo).")
    elif modo_operacion == "Modo B — H₂-centric":
        st.info("📌 Enfocado en **producción de H₂**. Aparece el **consumo parásito (parasitic power)** por upgrading.")
    else:
        st.info("📌 Enfocado en **hub energético**: compensación explícita entre ruta potencia y ruta H₂.")

    if modo_operacion in ["Modo B — H₂-centric", "Modo C — Mixed"]:
        h2_calidad = st.radio(
            "Calidad/uso típico del H₂ (afecta rendimiento y consumo parásito):",
            ["Estacionario / carga rápida (~70 kg/ton, 95% pure)", "Movilidad (~50 kg/ton, 99.999% pure)"],
            index=0
        )
    else:
        h2_calidad = "N/A"

    if modo_operacion == "Modo A — Power/Heat-centric":
        split_h2 = 0.0
    elif modo_operacion == "Modo B — H₂-centric":
        split_h2 = 1.0
    else:
        split_h2 = st.slider("% de syngas asignado a la ruta H₂:", 0, 100, 50, 5) / 100.0

    usar_fuel_cell = False
    frac_h2_a_fc = 0.0
    if modo_operacion in ["Modo B — H₂-centric", "Modo C — Mixed"]:
        usar_fuel_cell = st.checkbox(
            "Convertir parte del H₂ a electricidad (celda de combustible)",
            value=True if modo_operacion == "Modo C — Mixed" else False
        )
        if usar_fuel_cell:
            frac_h2_a_fc = st.slider("% del H₂ que va a celda de combustible:", 0, 100, 100, 5) / 100.0

    params_modo = defaults_por_modo(modo_operacion, h2_calidad)

    st.markdown("---")
    st.header("🧪 Residuos de entrada")
    preset = st.radio("Preset de composición:", ["La Pradera (aprox.)", "RSU municipal genérico", "Personalizar"], index=0)

    if preset == "La Pradera (aprox.)":
        composicion_actual = {k: v.copy() for k, v in LA_PRADERA_PRESET.items()}
        st.success("✅ Usando preset aproximado La Pradera.")
    elif preset == "RSU municipal genérico":
        composicion_actual = {k: v.copy() for k, v in RSU_MUNICIPAL_BASE.items()}
        st.success("✅ Usando RSU municipal genérico.")
    else:
        st.warning("🛠️ Ajusta porcentajes. Deben sumar 100%.")
        composicion_actual = {k: v.copy() for k, v in LA_PRADERA_PRESET.items()}
        for componente in composicion_actual.keys():
            nombre = componente.replace("_", " ").title()
            composicion_actual[componente]["pct"] = st.slider(
                nombre, min_value=0.0, max_value=100.0,
                value=float(composicion_actual[componente]["pct"]),
                step=0.5, key=f"pct_{componente}"
            )
        suma = sum(v["pct"] for v in composicion_actual.values())
        if abs(suma - 100.0) > 0.1:
            st.error(f"❌ La suma actual es {suma:.1f}% (debe ser 100%).")
        else:
            st.success(f"✅ Suma total: {suma:.1f}%")

    st.markdown("---")
    st.header("🌍 Emisiones (supuestos editables)")
    horizonte = st.radio("Horizonte para metano en rellenos:", ["100 años (GWP100)", "20 años (GWP20)"], index=0)
    fuente_factor_relleno = st.selectbox(
        "Factor de emisiones del relleno (kgCO2e/ton):",
        ["EPA/WARM (~640)", "Proxy Colombia (~880)", "Personalizado"],
        index=0
    )
    if fuente_factor_relleno == "EPA/WARM (~640)":
        EF_landfill_base = EF_LANDFILL_WARM_KG_TON
    elif fuente_factor_relleno == "Proxy Colombia (~880)":
        EF_landfill_base = EF_LANDFILL_CO_PROXY_KG_TON
    else:
        EF_landfill_base = st.number_input("Factor de relleno (kgCO2e/ton):", min_value=0.0, value=640.0, step=10.0)

    if horizonte.startswith("20 años"):
        mult_20 = st.number_input("Multiplicador proxy GWP20 vs GWP100:", min_value=1.0, value=float(GWP20_MULTIPLIER_DEFAULT), step=0.05)
        EF_landfill_kg_ton = EF_landfill_base * mult_20
    else:
        EF_landfill_kg_ton = EF_landfill_base

    EF_grid_tco2e_mwh = st.number_input("Factor emisión red Colombia (tCO2e/MWh):", min_value=0.0, value=float(EF_GRID_DEFAULT_TCO2E_MWH), step=0.001)
    EF_transport_kg_tkm = st.number_input("Factor transporte (kgCO2/(ton·km)):", min_value=0.0, value=float(EF_TRANSPORT_DEFAULT_KG_TKM), step=0.001)
    dist_baseline_km = st.number_input("Distancia AMVA → La Pradera (km):", min_value=0.0, value=55.0, step=1.0)
    dist_cluster_km = st.number_input("Distancia clúster → planta Boson (km):", min_value=0.0, value=15.0, step=1.0)

# =============================================
# ÁREA PRINCIPAL
# =============================================
st.header("📥 Configuración de la Planta")

pci_mezcla_gj_ton = 0.0
fraccion_cenizas = 0.0
for componente, datos in composicion_actual.items():
    fr = datos["pct"] / 100.0
    pci_mezcla_gj_ton += fr * datos["pci_gj_ton"]
    fraccion_cenizas += fr * (datos["cenizas_pct"] / 100.0)

pci_mwh_ton = pci_mezcla_gj_ton / GJ_PER_MWH
pci_kcal_kg = pci_mezcla_gj_ton * 239.0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PCI de la mezcla", f"{pci_mezcla_gj_ton:.2f} GJ/ton")
with col2:
    st.metric("PCI equivalente", f"{pci_mwh_ton:.2f} MWh/ton")
with col3:
    st.metric("PCI equivalente", f"{int(round(pci_kcal_kg)):,d} kcal/kg")

if pci_mezcla_gj_ton >= UMBRAL_AUTOSUF_GJ_TON:
    st.success(f"🔥 **Autosuficiencia térmica:** Autosuficiente (PCI ≥ {UMBRAL_AUTOSUF_GJ_TON:.1f} GJ/ton).")
elif pci_mezcla_gj_ton >= UMBRAL_CASI_AUTOSUF_GJ_TON:
    st.warning("🌡️ **Autosuficiencia térmica:** Casi autosuficiente (7–9 GJ/ton).")
else:
    st.error("❄️ **Autosuficiencia térmica:** Requiere apoyo externo (PCI < 7 GJ/ton).")

st.subheader("Capacidad de Procesamiento")
capacidad_total = st.number_input(
    "Cantidad total de residuos a procesar (toneladas/año):",
    min_value=1000.0,
    max_value=1300000.0,
    value=float(params_modo["capacidad_foak_ton_año"]),
    step=1000.0,
    help=("FOAK/BEU ≈ 36,000 ton/año (≈100 t/d). "
          "Es una **planta/módulo**, no un reactor. Config típica: **2×(2 t/h) operando + 1 standby**.")
)

if capacidad_total > 200000:
    st.info("📌 Estás modelando un **escenario agregado** de varias plantas BEU.")

calcular = st.button("🚀 Calcular Balance Completo", type="primary", use_container_width=True)

if calcular:
    with st.spinner("Calculando balance..."):
        CAP_FOAK = params_modo["capacidad_foak_ton_año"]
        num_plantas = int(np.ceil(capacidad_total / CAP_FOAK))
        capacidad_por_planta = capacidad_total / num_plantas

        kpis_ton = calcular_kpis_por_ton(
            pci_gj_ton=pci_mezcla_gj_ton,
            ash_frac=fraccion_cenizas,
            params_modo=params_modo,
            split_syngas_a_h2=split_h2,
            usar_fuel_cell=usar_fuel_cell,
            frac_h2_a_fc=frac_h2_a_fc
        )

        emis_ton = calcular_emisiones_por_ton(
            kpis_ton=kpis_ton,
            params_modo=params_modo,
            EF_landfill_kg_ton=EF_landfill_kg_ton,
            EF_grid_tco2e_mwh=EF_grid_tco2e_mwh,
            EF_transport_kg_tkm=EF_transport_kg_tkm,
            dist_baseline_km=dist_baseline_km,
            dist_cluster_km=dist_cluster_km
        )

        toneladas = capacidad_por_planta

        residuos_desviados_t = toneladas
        H2_export_t = (kpis_ton["H2_export_kg_ton"] * toneladas) / 1000.0
        escoria_t = (kpis_ton["escoria_kg_ton"] * toneladas) / 1000.0
        el_neta_mwh = kpis_ton["el_neta_mwh_ton"] * toneladas
        heat_mwhth = kpis_ton["heat_util_mwhth_ton"] * toneladas

        huella_neta_no_ccs_t = (emis_ton["huella_neta_no_ccs_kg"] * toneladas) / 1000.0
        huella_neta_with_ccs_t = (emis_ton["huella_neta_with_ccs_kg"] * toneladas) / 1000.0

        beneficio_landfill_t = (emis_ton["beneficio_landfill_kg"] * toneladas) / 1000.0
        beneficio_transport_t = (emis_ton["beneficio_transport_kg"] * toneladas) / 1000.0
        efecto_electricidad_t = (emis_ton["efecto_electricidad_kg"] * toneladas) / 1000.0
        em_proceso_sin_ccs_t = (emis_ton["direct_no_ccs_kg"] * toneladas) / 1000.0
        em_indirectas_t = (emis_ton["indirect_kg"] * toneladas) / 1000.0

        def scale(x):
            return x * num_plantas

        rows = []

        def add_row(nombre, anual_val, anual_unit, per_ton_val, per_ton_unit, nota):
            rows.append([nombre, f"{anual_val:,.2f} {anual_unit}", f"{per_ton_val:,.3f} {per_ton_unit}", nota])

        add_row("Residuos desviados de relleno sanitario", scale(residuos_desviados_t), "t/año", 1.0, "ton/ton",
                "Cada ton tratada por BEU es 1 ton NO dispuesta en relleno sanitario.")
        add_row("Hidrógeno (H₂) Exportable", scale(H2_export_t), "t/año", kpis_ton["H2_export_kg_ton"], "kg/ton",
                "H₂ exportable (si no se consume en celda de combustible).")
        add_row("Electricidad Neta", scale(el_neta_mwh), "MWhₑ/año", kpis_ton["el_neta_mwh_ton"], "MWhₑ/ton",
                "Electricidad neta (puede ser negativa si el modo requiere importación).")
        add_row("Calor Útil", scale(heat_mwhth), "MWhₜₕ/año", kpis_ton["heat_util_mwhth_ton"], "MWhₜₕ/ton",
                "Energía térmica útil recuperable/exportable.")
        add_row("Escoria vitrificada (IMBYROCK®)", scale(escoria_t), "t/año", kpis_ton["escoria_kg_ton"], "kg/ton",
                "Sólido inerte/vítreo. Potencial valorización (requiere QA/regulación local).")

        add_row("Huella neta SIN CCS (Boson − línea base)", scale(huella_neta_no_ccs_t), "t CO2e/año",
                emis_ton["huella_neta_no_ccs_kg"]/1000.0, "t CO2e/ton",
                "Negativo = ahorro. Positivo = peor que la línea base.")
        add_row("Huella neta CON CCS (Boson − línea base)", scale(huella_neta_with_ccs_t), "t CO2e/año",
                emis_ton["huella_neta_with_ccs_kg"]/1000.0, "t CO2e/ton",
                "Ahorro adicional si se captura CO₂ (proxy).")

        add_row("Beneficio: Relleno sanitario evitado", scale(beneficio_landfill_t), "t CO2e/año",
                emis_ton["beneficio_landfill_kg"]/1000.0, "t CO2e/ton",
                "Beneficio por NO disposición en relleno sanitario.")
        add_row("Beneficio: Transporte evitado", scale(beneficio_transport_t), "t CO2e/año",
                emis_ton["beneficio_transport_kg"]/1000.0, "t CO2e/ton",
                "Diferencia por descentralización (clúster cercano).")
        add_row("Efecto: Electricidad (exporta/importa)", scale(efecto_electricidad_t), "t CO2e/año",
                emis_ton["efecto_electricidad_kg"]/1000.0, "t CO2e/ton",
                "Exportación evita red; importación carga huella.")
        add_row("Emisiones: Proceso (SIN CCS)", scale(em_proceso_sin_ccs_t), "t CO2e/año",
                emis_ton["direct_no_ccs_kg"]/1000.0, "t CO2e/ton",
                "Proxy: CO₂ capturable tratado como emitido si no hay captura.")
        add_row("Emisiones: Indirectas", scale(em_indirectas_t), "t CO2e/año",
                emis_ton["indirect_kg"]/1000.0, "t CO2e/ton",
                "Auxiliares no eléctricos (proxy).")

        df_unificado_total = pd.DataFrame(
            rows,
            columns=[COL_INDICADOR, COL_TOTAL, COL_POR_TON, COL_NOTA]
        )

        st.markdown("---")
        st.header("📊 Resultados del balance (tabla integral)")
        st.dataframe(df_unificado_total, use_container_width=True, hide_index=True)

        st.header("📈 Visualización de resultados")
        fig = visualizar_balance_unificado(df_unificado_total, capacidad_total)
        st.pyplot(fig)

# =============================================
# PIE DE PÁGINA
# =============================================
st.markdown("---")
st.markdown(CREATED_BY_HTML, unsafe_allow_html=True)
