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
# BASE DE DATOS DE COMPONENTES RSU (Sidebar)
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
# HELPER PARA EXTRAER NÚMEROS DE CADENAS
# =============================================
def extraer_numero(celda):
    """
    Extrae de forma robusta el primer número de una celda que puede venir
    como string con unidades, signo, comas, etc. Si falla, devuelve 0.0
    """
    if isinstance(celda, (int, float, np.number)):
        return float(celda)
    txt = str(celda).replace('\xa0', ' ').replace(',', '')
    m = re.search(r'([+-]?\d+(?:\.\d+)?)', txt)
    return float(m.group(1)) if m else 0.0

# =============================================
# FUNCIONES DE CÁLCULO (CORE)
# =============================================
@st.cache_data
def calcular_parametros_desde_composicion(composicion_rsu):
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
    co2_proceso_kg_ton = pci_mezcla_gj_ton * 25

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
    resultados = {}
    toneladas = capacidad_ton_año

    # Cálculos de Energía
    energia_entrada_gj = toneladas * params['pci_residuo_gj_ton']
    energia_syngas_gj = energia_entrada_gj * params['eficiencia_conversion_plasma']
    autoconsumo_proceso_gj = energia_syngas_gj * params['autoconsumo_proceso_fraction']
    energia_neta_disponible_gj = energia_syngas_gj - autoconsumo_proceso_gj
    electricidad_bruta_mwh = (energia_neta_disponible_gj * 277.78) * params['eficiencia_generacion_electrica']
    autoconsumo_bop_mwh = toneladas * params['autoconsumo_bop_kwh_ton'] / 1000
    electricidad_neta_mwh = electricidad_bruta_mwh - autoconsumo_bop_mwh

    # Cálculos de Masa
    resultados['Hidrógeno (H₂)'] = toneladas * params['hidrogeno_kg_ton'] / 1000
    resultados['Escoria Vitrificada (IMBYROCK®)'] = toneladas * params['escoria_kg_ton'] / 1000
    resultados['Metales Recuperados'] = toneladas * params['metales_kg_ton'] / 1000
    resultados['CO₂ del Proceso (para captura)'] = toneladas * params['co2_proceso_kg_ton'] / 1000

    # Resultados de Energía
    resultados['Energía en Syngas (GJ)'] = round(energia_syngas_gj, 1)
    resultados['Electricidad Neta Exportable (MWh)'] = round(electricidad_neta_mwh, 1)
    resultados['Calor Útil (GJ)'] = round(toneladas * params['calor_util_gj_ton'], 1)

    # Cálculos de Emisiones (kg CO2e)
    emis_evitadas_total_kg = toneladas * (params['emis_evitadas_vertedero_kgco2e_ton'] +
                                          params['emis_evitadas_electricidad_kgco2e_ton'])
    emis_proceso_kg = toneladas * params['co2_proceso_kg_ton']
    emis_indirectas_kg = toneladas * params['emis_indirectas_kgco2e_ton']
    huella_proceso_sin_ccs_kg = emis_proceso_kg + emis_indirectas_kg
    huella_neta_sin_ccs_kg = huella_proceso_sin_ccs_kg + emis_evitadas_total_kg
    emis_proceso_con_ccs_kg = emis_proceso_kg * (1 - params['factor_captura_ccs'])
    huella_proceso_con_ccs_kg = emis_proceso_con_ccs_kg + emis_indirectas_kg
    huella_neta_con_ccs_kg = huella_proceso_con_ccs_kg + emis_evitadas_total_kg

    # Almacenar resultados de emisiones en t CO2e
    resultados['Emisiones Evitadas (t CO2e)'] = emis_evitadas_total_kg / 1000
    resultados['Emisiones del Proceso (t CO2e)'] = emis_proceso_kg / 1000
    resultados['Emisiones Indirectas (t CO2e)'] = emis_indirectas_kg / 1000
    resultados['Huella Neta SIN CCS (t CO2e)'] = huella_neta_sin_ccs_kg / 1000
    resultados['Huella Neta CON CCS (t CO2e)'] = huella_neta_con_ccs_kg / 1000

    # Condiciones técnicas / justificaciones
    condiciones_tecnicas = {
        'Energía en Syngas (GJ)': 'Gasificación plasma >1100°C, atmósfera controlada, eficiencia ~78%',
        'Electricidad Neta Exportable (MWh)': (
            f"Motor/turbina syngas (η={params['eficiencia_generacion_electrica']*100:.0f}%), "
            f"autoconsumo proceso: {params['autoconsumo_proceso_fraction']*100:.0f}%"
        ),
        'Hidrógeno (H₂)': 'Craqueo térmico + Water-Gas Shift (WGS) + Purificación PSA (>99.97%)',
        'Calor Útil (GJ)': 'Recuperación de calor residual de motores y sistemas de gas',
        'Escoria Vitrificada (IMBYROCK®)': 'Temperatura plasma >1400°C + enfriamiento rápido (vitrificación)',
        'Metales Recuperados': 'Separación por densidad en baño fundido',
        'CO₂ del Proceso (para captura)': 'Flujo concentrado en syngas, listo para sistema CCUS',
        'Emisiones Evitadas (t CO2e)': 'Suma de emisiones evitadas por vertedero y electricidad desplazada',
        'Emisiones del Proceso (t CO2e)': 'Emisiones directas del proceso de conversión térmica',
        'Emisiones Indirectas (t CO2e)': 'Electricidad importada, insumos y auxiliares',
        'Huella Neta SIN CCS (t CO2e)': 'Emisiones netas (Proceso+Indirectas+Evitadas), sin captura de CO₂',
        'Huella Neta CON CCS (t CO2e)': 'Con captura de CO₂ del syngas (eficiencia ~85%)'
    }

    # Construir DataFrame Unificado (corrientes/indicadores clave)
    data = []
    claves_para_tabla = [
        'Hidrógeno (H₂)',
        'Escoria Vitrificada (IMBYROCK®)',
        'Metales Recuperados',
        'CO₂ del Proceso (para captura)',
        'Energía en Syngas (GJ)',
        'Electricidad Neta Exportable (MWh)',
        'Calor Útil (GJ)',
        'Emisiones Evitadas (t CO2e)',
        'Emisiones del Proceso (t CO2e)',
        'Emisiones Indirectas (t CO2e)',
        'Huella Neta SIN CCS (t CO2e)',
        'Huella Neta CON CCS (t CO2e)'
    ]

    for clave in claves_para_tabla:
        valor = resultados[clave]
        unidad = clave[clave.find('('):] if '(' in clave else ''
        nombre = clave.split(' (')[0] if ' (' in clave else clave

        # Formato por tipo de indicador
        if 't CO2e' in clave:
            cantidad_anual = f"{valor:+,.1f} t CO2e"
            por_tonelada = f"{valor*1000/toneladas:+,.1f} kg CO2e/ton" if toneladas > 0 else "N/A"
        elif 'GJ' in clave or 'MWh' in clave:
            cantidad_anual = f"{valor:,.1f} {unidad}"
            por_tonelada = (
                f"{valor/toneladas:,.1f} {unidad.replace(')', '/ton)')}"
                if toneladas > 0 else "N/A"
            )
        else:
            cantidad_anual = f"{valor:,.1f} {unidad}"
            por_tonelada = (
                f"{valor*1000/toneladas:,.1f} kg/ton"
                if toneladas > 0 else "N/A"
            )

        data.append([
            nombre,
            cantidad_anual,
            por_tonelada,
            condiciones_tecnicas.get(clave, 'N/A')
        ])

    df_unificado = pd.DataFrame(
        data,
        columns=[
            'Corriente de Salida / Indicador',
            'Cantidad Anual Total',
            'Por Tonelada de Residuo',
            'Condición Técnica / Justificación'
        ]
    )

    return resultados, df_unificado

def calcular_balance_boson_modular(capacidad_total_ton_año, params):
    CAPACIDAD_FOAK = params['capacidad_foak_ton_año']
    num_plantas = int(np.ceil(capacidad_total_ton_año / CAPACIDAD_FOAK))
    capacidad_por_planta = capacidad_total_ton_año / num_plantas

    if num_plantas == 1:
        config_texto = f"1 planta de {capacidad_por_planta:,.0f} ton/año"
    else:
        config_texto = f"{num_plantas} plantas modulares de {capacidad_por_planta:,.0f} ton/año c/u"

    resultados_individual, df_unificado_individual = calcular_balance_boson(
        capacidad_por_planta, params
    )

    def escalar_valor_celda(valor_celda, factor):
        txt = str(valor_celda)
        m = re.search(r'([+-]?[\d,]+\.?\d*)', txt.replace(',', ''))
        if m:
            numero = float(m.group(1))
            numero_escalado = numero * factor
            # conserva signo explícito si lo tenía
            if '+' in m.group(1) or '-' in m.group(1):
                formato = f"{numero_escalado:+,.1f}"
            else:
                formato = f"{numero_escalado:,.1f}"
            return re.sub(r'[+-]?[\d,]+\.?\d*', formato, txt)
        return valor_celda

    df_unificado_total = df_unificado_individual.copy()
    df_unificado_total['Cantidad Anual Total'] = df_unificado_total['Cantidad Anual Total'].apply(
        lambda x: escalar_valor_celda(x, num_plantas)
    )

    resultados_totales = {}
    for clave, valor in resultados_individual.items():
        if isinstance(valor, (int, float)):
            resultados_totales[clave] = valor * num_plantas
        else:
            resultados_totales[clave] = valor

    return resultados_totales, df_unificado_total, config_texto, num_plantas, capacidad_por_planta

def visualizar_balance_unificado(df_unificado, capacidad_total):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Balance Integral - Capacidad Total: {capacidad_total:,.0f} ton/año',
        fontsize=14,
        fontweight='bold'
    )

    # Gráfico 1: Productos Principales
    productos_principales = ['Hidrógeno', 'Electricidad Neta', 'Escoria Vitrificada', 'Calor Útil']
    valores_principales = []
    for producto in productos_principales:
        fila = df_unificado[df_unificado['Corriente de Salida / Indicador'].str.contains(producto)]
        if not fila.empty:
            valor_texto = fila.iloc[0]['Cantidad Anual Total']
            valores_principales.append(extraer_numero(valor_texto))
        else:
            valores_principales.append(0)

    colores_principales = ['#2ca02c', '#9c27b0', '#ff9800', '#ffc107']
    bars1 = axes[0, 0].bar(productos_principales, valores_principales, color=colores_principales)
    axes[0, 0].set_title('1. Productos Principales', fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad Anual')
    axes[0, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars1, valores_principales):
        etiqueta = f'{val:,.0f}' if abs(val) >= 1000 else f'{val:,.1f}'
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            etiqueta,
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Gráfico 2: Balance de Emisiones (Huella neta)
    huella_sin_ccs_fila = df_unificado[
        df_unificado['Corriente de Salida / Indicador'].str.contains('Huella Neta SIN CCS')
    ]
    huella_con_ccs_fila = df_unificado[
        df_unificado['Corriente de Salida / Indicador'].str.contains('Huella Neta CON CCS')
    ]

    huella_sin_ccs = extraer_numero(huella_sin_ccs_fila.iloc[0]['Cantidad Anual Total']) \
        if not huella_sin_ccs_fila.empty else 0
    huella_con_ccs = extraer_numero(huella_con_ccs_fila.iloc[0]['Cantidad Anual Total']) \
        if not huella_con_ccs_fila.empty else 0

    categorias_emis = ['Huella Neta\n(SIN CCS)', 'Huella Neta\n(CON CCS)']
    valores_emis = [huella_sin_ccs, huella_con_ccs]
    colores_emis = ['#ff5722', '#4caf50'] if huella_con_ccs < huella_sin_ccs else ['#ff5722', '#ff5722']

    bars2 = axes[0, 1].bar(categorias_emis, valores_emis, color=colores_emis)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].set_title('2. Balance Neto de Emisiones (t CO2e/año)', fontweight='bold')
    axes[0, 1].set_ylabel('t CO2e por año')
    for bar, val in zip(bars2, valores_emis):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f'{val:+,.0f} t',
            ha='center',
            va='center',
            color='white' if abs(val) > max(map(abs, valores_emis)) * 0.3 else 'black',
            fontweight='bold',
            fontsize=10
        )

    # Gráfico 3: Desglose de Emisiones
    componentes = []
    valores_componentes = []
    for comp in ['Emisiones Evitadas', 'Emisiones del Proceso', 'Emisiones Indirectas']:
        for _, fila in df_unificado.iterrows():
            if comp in fila['Corriente de Salida / Indicador']:
                valor_texto = fila['Cantidad Anual Total']
                componentes.append(comp)
                valores_componentes.append(extraer_numero(valor_texto))
                break

    if componentes:
        axes[1, 0].barh(componentes, valores_componentes,
                        color=['#4caf50', '#f44336', '#ff9800'])
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('3. Desglose de Emisiones (t CO2e/año)', fontweight='bold')
        axes[1, 0].set_xlabel('t CO2e por año')
        max_val = max(map(abs, valores_componentes)) if valores_componentes else 1
        for i, (comp, val) in enumerate(zip(componentes, valores_componentes)):
            axes[1, 0].text(
                val / 2 if val >= 0 else val * 1.05,
                i,
                f'{val:+,.0f}',
                va='center',
                ha='center' if val >= 0 else 'left',
                color='white' if abs(val) > max_val * 0.3 else 'black',
                fontweight='bold'
            )

    # Gráfico 4: Tabla de Indicadores
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    filas_resumen = df_unificado[
        ~df_unificado['Corriente de Salida / Indicador'].str.contains('Emisiones|Huella')
    ].head(6)
    tabla_data = filas_resumen[[
        'Corriente de Salida / Indicador',
        'Por Tonelada de Residuo'
    ]].values
    if len(tabla_data) > 0:
        tabla = axes[1, 1].table(
            cellText=tabla_data,
            colLabels=['Indicador', 'Rendimiento por Tonelada'],
            loc='center',
            cellLoc='left'
        )
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(8)
        tabla.scale(1.2, 1.8)
        axes[1, 1].set_title('4. Indicadores Clave de Eficiencia', fontweight='bold', pad=20)

    plt.tight_layout()
    return fig

# =============================================
# INTERFAZ PRINCIPAL DE STREAMLIT
# =============================================

# BARRA LATERAL: Entrada de Composición de RSU
with st.sidebar:
    st.header("🧪 Configuración de Residuos (RSU)")

    modo_composicion = st.radio(
        "Modo de composición:",
        ["Usar valores por defecto (RSU municipal)", "Personalizar porcentajes"],
        index=0
    )

    composicion_actual = {}

    if modo_composicion == "Usar valores por defecto (RSU municipal)":
        for componente, datos in COMPOSICION_RSU_BASE.items():
            composicion_actual[componente] = datos.copy()

        st.success("✅ Usando composición municipal por defecto.")
        with st.expander("Ver composición"):
            for comp, datos in composicion_actual.items():
                st.text(f"• {comp.replace('_', ' ').title()}: {datos['pct']}%")

    else:
        st.subheader("Ajuste los porcentajes de cada componente:")

        porcentajes = {}
        for componente, datos in COMPOSICION_RSU_BASE.items():
            nombre_bonito = componente.replace('_', ' ').title()
            porcentajes[componente] = st.slider(
                f"{nombre_bonito}",
                min_value=0.0,
                max_value=100.0,
                value=float(datos['pct']),
                step=0.5,
                key=f"slider_{componente}"
            )

        suma_total = sum(porcentajes.values())

        if abs(suma_total - 100.0) > 0.1:
            st.warning(f"⚠️ La suma total es {suma_total:.1f}% (debe ser 100%).")
        else:
            st.success(f"✅ Suma total: {suma_total:.1f}%")

        for componente, datos in COMPOSICION_RSU_BASE.items():
            composicion_actual[componente] = datos.copy()
            composicion_actual[componente]['pct'] = porcentajes[componente]

# ÁREA PRINCIPAL: Entrada de Capacidad y Resultados
st.header("📥 Configuración de la Planta")

# Calcular parámetros ajustados desde la composición
params_ajustados, pci_mezcla = calcular_parametros_desde_composicion(composicion_actual)

# Métricas clave de la composición
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PCI de la mezcla", f"{pci_mezcla:.2f} GJ/ton")
with col2:
    st.metric("H₂ estimado", f"{params_ajustados['hidrogeno_kg_ton']} kg/ton")
with col3:
    st.metric("Escoria estimada", f"{params_ajustados['escoria_kg_ton']} kg/ton")

# Entrada de capacidad de procesamiento
st.subheader("Capacidad de Procesamiento")
capacidad_total = st.number_input(
    "Cantidad total de residuos a procesar (toneladas/año):",
    min_value=1000.0,
    max_value=500000.0,
    value=36000.0,
    step=1000.0,
    help="La capacidad mínima para una planta FOAK es de 36,000 ton/año."
)

# Botón para calcular
calcular = st.button("🚀 Calcular Balance Completo", type="primary", use_container_width=True)

# =============================================
# PROCESAMIENTO Y VISUALIZACIÓN DE RESULTADOS
# =============================================
if calcular:
    with st.spinner("Calculando balance de masa, energía y emisiones..."):
        resultados_totales, df_unificado_total, config, num_plantas, cap_por_planta = \
            calcular_balance_boson_modular(capacidad_total, params_ajustados)

        st.markdown("---")
        st.header("🏭 Configuración Técnica")

        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            st.success(f"**{config}**")
        with col_conf2:
            st.info(f"Capacidad total del sistema: **{capacidad_total:,.0f} ton/año**")

        if num_plantas > 1:
            st.info(
                f"🔧 Nota: Despliegue modular basado en unidades estandarizadas "
                f"de {params_ajustados['capacidad_foak_ton_año']:,.0f} ton/año."
            )

        st.header("📊 Resultados del Balance")
        st.dataframe(df_unificado_total, use_container_width=True, hide_index=True)

        if num_plantas > 1:
            with st.expander(
                f"📈 Ver resultados por planta individual ({cap_por_planta:,.0f} ton/año c/u)"
            ):
                resultados_individual, df_individual = calcular_balance_boson(
                    cap_por_planta, params_ajustados
                )
                st.dataframe(df_individual, use_container_width=True, hide_index=True)

        st.header("📈 Visualización de Resultados")
        fig = visualizar_balance_unificado(df_unificado_total, capacidad_total)
        st.pyplot(fig)

        st.markdown("---")
        st.header("💡 Resumen Ejecutivo y Análisis de Emisiones")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            h2_total = resultados_totales.get('Hidrógeno (H₂)', 0)
            elec_total = resultados_totales.get('Electricidad Neta Exportable (MWh)', 0)
            st.metric("Producción total de H₂", f"{h2_total:,.1f} ton/año")
            st.metric("Electricidad neta exportable", f"{elec_total:,.0f} MWh/año")

        with col_res2:
            huella_sin_ccs = resultados_totales.get('Huella Neta SIN CCS (t CO2e)', 0)
            huella_con_ccs = resultados_totales.get('Huella Neta CON CCS (t CO2e)', 0)

            # Determinar el estado de carbono y color permitido por Streamlit
            if huella_sin_ccs < 0:
                estado_carbono = "CARBONO-NEGATIVO"
                icono = "✅"
                color_delta = "normal"   # verde (positivo) según convención de Streamlit
            elif huella_sin_ccs == 0:
                estado_carbono = "CARBONO-NEUTRAL"
                icono = "⚖️"
                color_delta = "off"      # sin énfasis de color
            else:
                estado_carbono = "HUELLA POSITIVA"
                icono = "⚠️"
                color_delta = "inverse"  # rojo cuando el delta es “malo”

            st.metric(
                f"{icono} Huella Neta (sin CCS)",
                f"{huella_sin_ccs:+,.0f} t CO2e/año",
                delta=estado_carbono,
                delta_color=color_delta
            )

            if huella_con_ccs < huella_sin_ccs:
                reduccion_ccs = (
                    (huella_sin_ccs - huella_con_ccs) / abs(huella_sin_ccs) * 100
                    if huella_sin_ccs != 0 else 0
                )
                st.metric(
                    "Huella Neta (con CCS)",
                    f"{huella_con_ccs:+,.0f} t CO2e/año",
                    delta=f"Reducción del {reduccion_ccs:.0f}%",
                    delta_color="normal"
                )

        if num_plantas > 1:
            st.subheader("💰 Implicaciones Estratégicas de la Modularidad")
            st.markdown(f"""
            Un sistema de **{num_plantas} plantas modulares** permite:
            - **Implementación por fases** (priorización de clústeres geográficos)
            - **Redundancia operativa** (mantenimiento sin interrupción total del sistema)
            - **Escalabilidad progresiva** según disponibilidad de residuos y financiamiento
            - **Distribución optimizada** de costos de logística de residuos
            """)

# Pie de página
st.markdown("---")
st.caption("""
⚠️ **Nota:** Los resultados dependen críticamente de:
1. Composición y poder calorífico de los residuos de entrada
2. Parámetros de eficiencia específicos del diseño final
3. Factor de emisión de la red eléctrica desplazada
""")
