import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PAGE CONFIGURATION & TITLE
# =============================================
st.set_page_config(page_title="Calculadora Boson Energy", layout="wide")
st.title("⚡ Calculadora de Balance y Emisiones - Tecnología Boson")
st.markdown("---")

# =============================================
# SESSION STATE INITIALIZATION
# =============================================
if 'params_ajustados' not in st.session_state:
    st.session_state.params_ajustados = None
if 'resultados_totales' not in st.session_state:
    st.session_state.resultados_totales = None
if 'df_unificado_total' not in st.session_state:
    st.session_state.df_unificado_total = None

# =============================================
# DATABASE OF RSU COMPONENTS (SIDEBAR)
# =============================================
COMPOSICION_RSU_BASE = {
    'plasticos': {'pct': 12.0, 'pci_gj_ton': 35.0, 'h2_potencial_kg_ton': 120, 'cenizas_pct': 5},
    'organicos': {'pct': 45.0, 'pci_gj_ton': 5.0,  'h2_potencial_kg_ton': 15,  'cenizas_pct': 15},
    'papel_carton': {'pct': 18.0, 'pci_gj_ton': 16.0, 'h2_potencial_kg_ton': 40,  'cenizas_pct': 8},
    'textiles': {'pct': 4.0,  'pci_gj_ton': 20.0, 'h2_potencial_kg_ton': 60,  'cenizas_pct': 10},
    'madera': {'pct': 3.0,  'pci_gj_ton': 18.0, 'h2_potencial_kg_ton': 35,  'cenizas_pct': 5},
    'otros_combustibles': {'pct': 5.0,  'pci_gj_ton': 25.0, 'h2_potencial_kg_ton': 80,  'cenizas_pct': 20},
    'inertes_metales': {'pct': 10.0, 'pci_gj_ton': 0.0,  'h2_potencial_kg_ton': 0,   'cenizas_pct': 95},
    'otros_inertes': {'pct': 3.0,  'pci_gj_ton': 0.0,  'h2_potencial_kg_ton': 0,   'cenizas_pct': 90}
}

# Fixed Process Parameters
PARAMS_PROCESO = {
    'capacidad_foak_ton_año': 36000,
    'eficiencia_conversion_plasma': 0.78,
    'eficiencia_generacion_electrica': 0.38,
    'autoconsumo_proceso_fraction': 0.28,
    'autoconsumo_bop_kwh_ton': 100,
    'factor_captura_ccs': 0.85,
    'emis_evitadas_vertedero_kgco2e_ton': -1000.0,
    'emis_evitadas_electricidad_kgco2e_ton': -750.0,
    'emis_indirectas_kgco2e_ton': 100.0,
}

# =============================================
# CORE CALCULATION FUNCTIONS
# =============================================
@st.cache_data
def calcular_parametros_desde_composicion(composicion_rsu):
    """Calculate key process parameters from RSU composition."""
    pci_mezcla_gj_ton = 0.0
    h2_potencial_kg_ton = 0.0
    fraccion_cenizas = 0.0

    for componente, datos in composicion_rsu.items():
        fraccion = datos['pct'] / 100.0
        pci_mezcla_gj_ton += fraccion * datos['pci_gj_ton']
        h2_potencial_kg_ton += fraccion * datos['h2_potencial_kg_ton']
        fraccion_cenizas += fraccion * (datos['cenizas_pct'] / 100.0)

    factor_eficiencia_h2 = 0.25
    h2_real_kg_ton = h2_potencial_kg_ton * factor_eficiencia_h2
    fraccion_vitrificacion = 0.90
    escoria_kg_ton = fraccion_cenizas * 1000 * fraccion_vitrificacion
    co2_proceso_kg_ton = pci_mezcla_gj_ton * 25  # Approximate factor

    params_ajustados = PARAMS_PROCESO.copy()
    params_ajustados.update({
        'pci_residuo_gj_ton': round(pci_mezcla_gj_ton, 2),
        'hidrogeno_kg_ton': round(h2_real_kg_ton, 1),
        'escoria_kg_ton': round(escoria_kg_ton, 1),
        'metales_kg_ton': 10.0,
        'co2_proceso_kg_ton': round(co2_proceso_kg_ton, 0),
        'calor_util_gj_ton': round(pci_mezcla_gj_ton * 0.20, 1),
    })
    return params_ajustados, pci_mezcla_gj_ton

def calcular_balance_boson(capacidad_ton_año, params):
    """Calculate complete mass, energy, and emissions balance for ONE plant."""
    resultados = {}
    toneladas = capacidad_ton_año

    # Energy Calculations
    energia_entrada_gj = toneladas * params['pci_residuo_gj_ton']
    energia_syngas_gj = energia_entrada_gj * params['eficiencia_conversion_plasma']
    autoconsumo_proceso_gj = energia_syngas_gj * params['autoconsumo_proceso_fraction']
    energia_neta_disponible_gj = energia_syngas_gj - autoconsumo_proceso_gj
    electricidad_bruta_mwh = (energia_neta_disponible_gj * 277.78) * params['eficiencia_generacion_electrica']
    autoconsumo_bop_mwh = toneladas * params['autoconsumo_bop_kwh_ton'] / 1000
    electricidad_neta_mwh = electricidad_bruta_mwh - autoconsumo_bop_mwh

    # Mass Calculations
    resultados['Hidrógeno (H₂)'] = toneladas * params['hidrogeno_kg_ton'] / 1000
    resultados['Escoria Vitrificada (IMBYROCK®)'] = toneladas * params['escoria_kg_ton'] / 1000
    resultados['Metales Recuperados'] = toneladas * params['metales_kg_ton'] / 1000
    resultados['CO₂ del Proceso (para captura)'] = toneladas * params['co2_proceso_kg_ton'] / 1000

    # Energy Results
    resultados['Energía en Syngas (GJ)'] = round(energia_syngas_gj, 1)
    resultados['Electriccción Neta Exportable (MWh)'] = round(electricidad_neta_mwh, 1)
    resultados['Calor Útil (GJ)'] = round(toneladas * params['calor_util_gj_ton'], 1)

    # Emissions Calculations (kg CO2e)
    emis_evitadas_total_kg = toneladas * (params['emis_evitadas_vertedero_kgco2e_ton'] + params['emis_evitadas_electricidad_kgco2e_ton'])
    emis_proceso_kg = toneladas * params['co2_proceso_kg_ton']
    emis_indirectas_kg = toneladas * params['emis_indirectas_kgco2e_ton']
    huella_proceso_sin_ccs_kg = emis_proceso_kg + emis_indirectas_kg
    huella_neta_sin_ccs_kg = huella_proceso_sin_ccs_kg + emis_evitadas_total_kg
    emis_proceso_con_ccs_kg = emis_proceso_kg * (1 - params['factor_captura_ccs'])
    huella_proceso_con_ccs_kg = emis_proceso_con_ccs_kg + emis_indirectas_kg
    huella_neta_con_ccs_kg
