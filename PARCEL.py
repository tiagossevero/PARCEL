"""
================================================================================
PROJETO PARCEL - Sistema de Análise e Monitoramento de Parcelamentos
Receita Estadual de Santa Catarina - SEF/SC
Dashboard Streamlit - Análise de Parcelamentos e Machine Learning
================================================================================
"""

import streamlit as st

# =============================================================================
# 1. IMPORTS E CONFIGURAÇÕES INICIAIS
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings
import ssl

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Hack SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="PARCEL - Monitoramento de Parcelamentos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. ESTILOS CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565C0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Estilo aprimorado para métricas/KPIs com tooltips visuais */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #1565C0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 6px 16px rgba(21, 101, 192, 0.3);
        transform: translateY(-3px);
        border-color: #0d47a1;
    }
    div[data-testid="stMetric"] > label {
        font-weight: 600;
        color: #2c3e50;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1565C0;
    }

    /* Estilo para ícone de ajuda nos KPIs */
    div[data-testid="stMetric"] [data-testid="stTooltipIcon"] {
        color: #1976d2;
        opacity: 0.7;
        transition: opacity 0.2s ease;
    }
    div[data-testid="stMetric"]:hover [data-testid="stTooltipIcon"] {
        opacity: 1;
    }

    /* Caixa de ajuda contextual aprimorada */
    .help-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #1976d2;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 0.95rem;
    }
    .help-box h4 {
        color: #1565C0;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }

    /* Dica de UX - indicador visual */
    .ux-tip {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 4px solid #ffa000;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .ux-tip::before {
        content: "💡";
        font-size: 1.2rem;
    }

    /* Legenda de indicadores */
    .indicator-legend {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.85rem;
    }
    .indicator-legend-title {
        font-weight: 600;
        color: #424242;
        margin-bottom: 8px;
    }
    .indicator-item {
        display: inline-flex;
        align-items: center;
        margin-right: 15px;
        margin-bottom: 5px;
    }
    .indicator-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }

    .risco-critico {
        background-color: #ffebee;
        border-left: 4px solid #c62828;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .risco-alto {
        background-color: #fff3e0;
        border-left: 4px solid #ef6c00;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .risco-medio {
        background-color: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .risco-baixo {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .alerta-urgente {
        background-color: #ffcdd2;
        border: 2px solid #c62828;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }

    /* Cards de resumo */
    .card-resumo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .card-resumo h3 {
        margin: 0;
        font-size: 2rem;
    }
    .card-resumo p {
        margin: 5px 0 0 0;
        opacity: 0.9;
    }

    /* KPI cards com cores por categoria */
    .kpi-total {
        border-left: 4px solid #1976d2 !important;
    }
    .kpi-success {
        border-left: 4px solid #2e7d32 !important;
    }
    .kpi-danger {
        border-left: 4px solid #c62828 !important;
    }
    .kpi-warning {
        border-left: 4px solid #f57c00 !important;
    }

    /* Seção de ajuda expandida */
    .help-section {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    .help-section-title {
        font-weight: 600;
        color: #1565C0;
        font-size: 1rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .help-section-content {
        color: #616161;
        font-size: 0.9rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 3. FUNÇÕES AUXILIARES
# =============================================================================

def formatar_cnpj(cnpj):
    """Formata CNPJ para garantir 14 dígitos com zeros à esquerda."""
    if pd.isna(cnpj):
        return None
    cnpj_str = str(cnpj).strip()
    cnpj_str = cnpj_str.replace('.', '').replace('/', '').replace('-', '')
    return cnpj_str.zfill(14)

def formatar_cnpj_visualizacao(cnpj):
    """Formata CNPJ para visualização (XX.XXX.XXX/XXXX-XX)."""
    cnpj_limpo = formatar_cnpj(cnpj)
    if not cnpj_limpo or len(cnpj_limpo) != 14:
        return cnpj
    return f"{cnpj_limpo[:2]}.{cnpj_limpo[2:5]}.{cnpj_limpo[5:8]}/{cnpj_limpo[8:12]}-{cnpj_limpo[12:14]}"

def formatar_valor_br(valor):
    """Formata valor para padrão brasileiro."""
    if valor is None or pd.isna(valor):
        return "R$ 0,00"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_valor_milhoes(valor):
    """Formata valor em milhões."""
    if valor is None or pd.isna(valor):
        return "R$ 0,00 MM"
    return f"R$ {valor/1e6:,.2f} MM".replace(",", "X").replace(".", ",").replace("X", ".")

def formatar_valor_bilhoes(valor):
    """Formata valor em bilhões."""
    if valor is None or pd.isna(valor):
        return "R$ 0,00 Bi"
    return f"R$ {valor/1e9:,.2f} Bi".replace(",", "X").replace(".", ",").replace("X", ".")

def get_cor_risco(classificacao):
    """Retorna cor baseada na classificação de risco."""
    cores = {
        'CRÍTICO': '#c62828',
        'ALTO': '#ef6c00',
        'MÉDIO': '#f9a825',
        'BAIXO': '#2e7d32'
    }
    return cores.get(classificacao, '#9e9e9e')

def get_cor_status(status):
    """Retorna cor baseada no status do parcelamento."""
    cores = {
        'ATIVO': '#1976d2',
        'QUITADO': '#2e7d32',
        'CANCELADO': '#c62828',
        'PENDENTE 1ª PARCELA': '#ff9800',
        'SUSPENSO': '#9e9e9e'
    }
    return cores.get(status, '#9e9e9e')

# =============================================================================
# 4. FUNÇÕES DE CONEXÃO E CARREGAMENTO DE DADOS
# =============================================================================

IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
DATABASE = 'gecob'

IMPALA_USER = st.secrets.get("impala_credentials", {}).get("user", "tsevero")
IMPALA_PASSWORD = st.secrets.get("impala_credentials", {}).get("password", "")

@st.cache_resource
def get_impala_engine():
    """Cria engine de conexão Impala."""
    try:
        engine = create_engine(
            f'impala://{IMPALA_HOST}:{IMPALA_PORT}/{DATABASE}',
            connect_args={
                'user': IMPALA_USER,
                'password': IMPALA_PASSWORD,
                'auth_mechanism': 'LDAP',
                'use_ssl': True
            }
        )
        return engine
    except Exception as e:
        st.sidebar.error(f"Erro na conexão: {str(e)[:100]}")
        return None


@st.cache_data(ttl=3600)
def carregar_resumo_executivo(_engine):
    """Carrega apenas o resumo executivo - rápido."""
    if _engine is None:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT * FROM gecob.parcel_resumo_executivo
            LIMIT 1
        """
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.error(f"Erro ao carregar resumo: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_dados_agregados(_engine):
    """Carrega dados agregados para visão geral - rápido."""
    dados = {}
    
    if _engine is None:
        return {}
    
    try:
        # 1. Resumo executivo
        dados['resumo'] = carregar_resumo_executivo(_engine)
        
        # 2. Métricas por GERFE
        query_gerfe = """
            SELECT * FROM gecob.parcel_metricas_gerfe
            ORDER BY valor_total DESC
        """
        dados['metricas_gerfe'] = pd.read_sql(query_gerfe, _engine)
        
        # 3. Métricas por GES
        query_ges = """
            SELECT * FROM gecob.parcel_metricas_ges
            ORDER BY valor_total DESC
        """
        dados['metricas_ges'] = pd.read_sql(query_ges, _engine)
        
        # 4. Série temporal
        query_serie = """
            SELECT * FROM gecob.parcel_serie_temporal
            ORDER BY periodo_pedido
        """
        dados['serie_temporal'] = pd.read_sql(query_serie, _engine)
        
        # 5. Análise por faixa de parcelas
        query_faixas = """
            SELECT * FROM gecob.parcel_analise_faixas
        """
        dados['analise_faixas'] = pd.read_sql(query_faixas, _engine)
        
        # 6. Análise por CNAE
        query_cnae = """
            SELECT * FROM gecob.parcel_analise_cnae
            ORDER BY valor_total DESC
            LIMIT 20
        """
        dados['analise_cnae'] = pd.read_sql(query_cnae, _engine)
        
        # 7. Programas especiais
        query_programas = """
            SELECT * FROM gecob.parcel_programas_especiais
            ORDER BY valor_total DESC
        """
        dados['programas_especiais'] = pd.read_sql(query_programas, _engine)
        
        # 8. Alertas (top 100)
        query_alertas = """
            SELECT * FROM gecob.parcel_alertas
            ORDER BY prioridade, valor_parcelado DESC
            LIMIT 100
        """
        dados['alertas'] = pd.read_sql(query_alertas, _engine)
        
        # 9. Empresas com score (top 500 por risco)
        query_empresas_score = """
            SELECT * FROM gecob.parcel_empresas_score
            ORDER BY score_risco_final DESC
            LIMIT 500
        """
        dados['empresas_score'] = pd.read_sql(query_empresas_score, _engine)
        
        # 10. RECUPERA+ resumo
        query_recuperamais = """
            SELECT 
                tipo_adesao,
                origem_debito,
                cd_status,
                desc_status,
                COUNT(*) as qtd,
                COUNT(DISTINCT cnpj_raiz) as empresas,
                CAST(SUM(valor_parcelado) AS DECIMAL(18,2)) as valor_total,
                ROUND(AVG(num_parcelas), 1) as media_parcelas
            FROM gecob.parcel_recuperamais
            GROUP BY tipo_adesao, origem_debito, cd_status, desc_status
            ORDER BY tipo_adesao, origem_debito, cd_status
        """
        dados['recuperamais_resumo'] = pd.read_sql(query_recuperamais, _engine)
        
        # 11. Expectativa de recebimento agregada
        query_expectativa = """
            SELECT 
                risco_empresa,
                COUNT(*) as qtd_parcelamentos,
                CAST(SUM(valor_esperado_3_meses) AS DECIMAL(18,2)) as total_3_meses,
                CAST(SUM(valor_esperado_12_meses) AS DECIMAL(18,2)) as total_12_meses,
                CAST(SUM(valor_ajustado_risco_3_meses) AS DECIMAL(18,2)) as ajustado_3_meses,
                CAST(SUM(valor_ajustado_risco_12_meses) AS DECIMAL(18,2)) as ajustado_12_meses
            FROM gecob.parcel_expectativa_recebimento
            GROUP BY risco_empresa
        """
        dados['expectativa_resumo'] = pd.read_sql(query_expectativa, _engine)
        
    except Exception as e:
        st.error(f"Erro ao carregar dados agregados: {e}")
        return {}
    
    return dados


def carregar_detalhes_empresa(_engine, cnpj_raiz):
    """Carrega detalhes completos de uma empresa - sob demanda."""
    dados = {}
    
    try:
        # Dados da empresa
        query_empresa = f"""
            SELECT * FROM gecob.parcel_empresas
            WHERE cnpj_raiz = '{cnpj_raiz}'
        """
        dados['empresa'] = pd.read_sql(query_empresa, _engine)
        
        # Score da empresa
        query_score = f"""
            SELECT * FROM gecob.parcel_empresas_score
            WHERE cnpj_raiz = '{cnpj_raiz}'
        """
        dados['score'] = pd.read_sql(query_score, _engine)
        
        # Parcelamentos da empresa
        query_parcelamentos = f"""
            SELECT * FROM gecob.parcel_base
            WHERE cnpj_raiz = '{cnpj_raiz}'
            ORDER BY dt_pedido DESC
        """
        dados['parcelamentos'] = pd.read_sql(query_parcelamentos, _engine)
        
        # Alertas da empresa
        query_alertas = f"""
            SELECT * FROM gecob.parcel_alertas
            WHERE cnpj_raiz = '{cnpj_raiz}'
            ORDER BY prioridade
        """
        dados['alertas'] = pd.read_sql(query_alertas, _engine)
        
        # Expectativa de recebimento
        query_expectativa = f"""
            SELECT * FROM gecob.parcel_expectativa_recebimento
            WHERE cnpj_raiz = '{cnpj_raiz}'
        """
        dados['expectativa'] = pd.read_sql(query_expectativa, _engine)
        
    except Exception as e:
        st.error(f"Erro ao carregar detalhes da empresa: {e}")
    
    return dados


def carregar_parcelamentos_gerfe(_engine, cd_gerfe):
    """Carrega parcelamentos de uma regional específica."""
    query = f"""
        SELECT 
            pb.num_parcelamento,
            pb.cnpj,
            pb.razao_social,
            pb.categoria_parcelamento,
            pb.desc_status,
            pb.valor_parcelado,
            pb.num_parcelas,
            pb.dt_pedido,
            es.score_risco_final,
            es.classificacao_risco
        FROM gecob.parcel_base pb
        LEFT JOIN gecob.parcel_empresas_score es ON pb.cnpj_raiz = es.cnpj_raiz
        WHERE pb.cd_gerfe = {cd_gerfe}
        ORDER BY pb.valor_parcelado DESC
        LIMIT 500
    """
    return pd.read_sql(query, _engine)


def carregar_recuperamais_detalhado(_engine):
    """Carrega dados detalhados do RECUPERA+."""
    query = """
        SELECT * FROM gecob.parcel_recuperamais
        ORDER BY valor_parcelado DESC
        LIMIT 1000
    """
    return pd.read_sql(query, _engine)


def carregar_dados_ml(_engine):
    """Carrega dados para treinamento do modelo de ML."""
    query = """
        SELECT 
            pe.cnpj_raiz,
            pe.total_parcelamentos,
            pe.qtd_ativos,
            pe.qtd_cancelados,
            pe.qtd_quitados,
            pe.qtd_pendentes_1a_parcela,
            pe.taxa_sucesso_pct,
            pe.valor_total_parcelado,
            pe.valor_medio_parcelamento,
            pe.media_parcelas,
            pe.media_dias_quitacao,
            pe.media_dias_ate_cancelamento,
            pe.dias_desde_ultimo_parcelamento,
            pe.flag_reincidente,
            pe.flag_nunca_quitou,
            pe.flag_simples_nacional,
            pe.classificacao_comportamento,
            -- Target: parcelamentos ativos que podem ser cancelados
            CASE 
                WHEN pe.qtd_ativos > 0 AND pe.classificacao_comportamento IN ('ALTO RISCO', 'RISCO MODERADO') 
                THEN 1 ELSE 0 
            END as risco_cancelamento
        FROM gecob.parcel_empresas pe
        WHERE pe.total_parcelamentos >= 2
    """
    return pd.read_sql(query, _engine)


# =============================================================================
# 5. FUNÇÕES DE MACHINE LEARNING
# =============================================================================

def treinar_modelo_risco(df):
    """Treina modelo de predição de risco de cancelamento."""
    
    # Preparar features
    features = [
        'total_parcelamentos', 'qtd_cancelados', 'qtd_quitados',
        'taxa_sucesso_pct', 'valor_medio_parcelamento', 'media_parcelas',
        'dias_desde_ultimo_parcelamento', 'flag_reincidente', 
        'flag_nunca_quitou', 'flag_simples_nacional'
    ]
    
    # Limpar dados
    df_ml = df.copy()
    df_ml = df_ml.dropna(subset=features + ['risco_cancelamento'])
    
    # Preencher NaN
    for col in features:
        if df_ml[col].dtype in ['float64', 'int64']:
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())
    
    X = df_ml[features]
    y = df_ml['risco_cancelamento']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo
    modelo = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    modelo.fit(X_train_scaled, y_train)
    
    # Avaliar
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    metricas = {
        'accuracy': (y_pred == y_test).mean(),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        'feature_importance': dict(zip(features, modelo.feature_importances_))
    }
    
    return modelo, scaler, features, metricas


def prever_risco_empresas(modelo, scaler, features, df):
    """Aplica modelo para prever risco das empresas."""
    df_pred = df.copy()
    
    # Preparar features
    for col in features:
        if col not in df_pred.columns:
            df_pred[col] = 0
        if df_pred[col].dtype in ['float64', 'int64']:
            df_pred[col] = df_pred[col].fillna(df_pred[col].median())
    
    X = df_pred[features].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Prever
    df_pred['prob_cancelamento'] = modelo.predict_proba(X_scaled)[:, 1]
    df_pred['risco_ml'] = modelo.predict(X_scaled)
    
    return df_pred


# =============================================================================
# 6. PÁGINAS DO DASHBOARD
# =============================================================================

def pagina_dashboard_executivo(dados, filtros_globais):
    """Dashboard executivo com KPIs principais."""
    st.markdown("<h1 class='main-header'>📊 Dashboard Executivo - Parcelamentos</h1>", unsafe_allow_html=True)

    resumo = dados.get('resumo', pd.DataFrame())

    if resumo.empty:
        st.warning("Dados de resumo não disponíveis.")
        return

    r = resumo.iloc[0]

    # Caixa de ajuda expandida
    with st.expander("ℹ️ Como interpretar este dashboard", expanded=False):
        st.markdown("""
        ### Objetivo
        Monitorar e analisar os parcelamentos de tributos estaduais em Santa Catarina, fornecendo
        uma visão consolidada para tomada de decisões.

        ### Como usar os indicadores

        | Indicador | O que significa | Ação sugerida |
        |-----------|-----------------|---------------|
        | **Total Parcelamentos** | Volume total de acordos no sistema | Acompanhar tendência mensal |
        | **Taxa de Sucesso** | Efetividade dos parcelamentos (Quitados ÷ Finalizados) | Meta: manter acima de 60% |
        | **Alertas Ativos** | Parcelamentos com risco de cancelamento | Priorizar ações de cobrança |
        | **Valor em Risco** | Montante com probabilidade de não recebimento | Focar em empresas críticas |

        ### Cores dos indicadores
        - 🟢 **Verde**: Indicador positivo (ex: quitados, taxa de sucesso alta)
        - 🔴 **Vermelho**: Indicador de atenção (ex: cancelados, alertas urgentes)
        - 🟡 **Amarelo**: Indicador de monitoramento (ex: pendentes)
        - 🔵 **Azul**: Indicador neutro/informativo (ex: totais)

        **Fonte de dados:** gecob.parcel_* | **Atualização:** Diária
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Passe o mouse sobre os cards para ver detalhes. Clique no ícone <b>(?)</b> para explicações completas.</span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs principais
    st.markdown("### 📈 Visão Geral")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Parcelamentos",
            f"{int(r['total_parcelamentos']):,}",
            help="📊 **Total de Parcelamentos**\n\nQuantidade total de parcelamentos registrados no sistema, incluindo todos os status (ativos, quitados, cancelados, pendentes).\n\n**Como usar:** Compare com meses anteriores para identificar tendências de adesão."
        )
    with col2:
        st.metric(
            "Empresas",
            f"{int(r['total_empresas']):,}",
            help="🏢 **Empresas Únicas**\n\nNúmero de empresas distintas (CNPJ raiz) que possuem pelo menos um parcelamento.\n\n**Observação:** Uma empresa pode ter múltiplos parcelamentos simultaneamente."
        )
    with col3:
        st.metric(
            "Valor Total",
            formatar_valor_bilhoes(r['valor_total_parcelado']),
            help="💰 **Valor Total Parcelado**\n\nSoma de todos os valores originais parcelados, em bilhões de reais.\n\n**Inclui:** Todos os parcelamentos independente do status atual."
        )
    with col4:
        st.metric(
            "Taxa de Sucesso",
            f"{r['taxa_sucesso_global_pct']:.1f}%",
            help="✅ **Taxa de Sucesso Global**\n\n**Fórmula:** (Quitados ÷ (Quitados + Cancelados)) × 100\n\n**Interpretação:**\n- Acima de 70%: Excelente\n- 50-70%: Adequado\n- Abaixo de 50%: Requer atenção\n\n**Meta institucional:** ≥ 65%"
        )
    with col5:
        st.metric(
            "Alertas Ativos",
            f"{int(r['total_alertas']):,}",
            delta=f"{int(r['alertas_urgentes'])} urgentes",
            delta_color="inverse",
            help="🚨 **Alertas de Risco**\n\nParcelamentos que apresentam sinais de possível inadimplência.\n\n**Urgentes:** Requerem ação imediata (empresas com alto risco de cancelamento).\n\n**Ação:** Acesse 'Central de Alertas' para detalhes."
        )

    st.markdown("---")

    # Legenda dos status
    st.markdown("""
    <div class='indicator-legend'>
        <div class='indicator-legend-title'>📋 Legenda dos Status</div>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #1976d2;'></span>Ativo: em andamento</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #2e7d32;'></span>Quitado: finalizado com sucesso</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #c62828;'></span>Cancelado: inadimplente</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #ff9800;'></span>Pendente: aguardando 1ª parcela</span>
    </div>
    """, unsafe_allow_html=True)

    # Segunda linha de KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🔵 Ativos",
            f"{int(r['qtd_ativos']):,}",
            delta=formatar_valor_bilhoes(r['valor_ativos']),
            help="🔵 **Parcelamentos Ativos**\n\nAcordos em andamento com pagamentos sendo realizados regularmente.\n\n**Valor mostrado abaixo:** Montante total em parcelamentos ativos (potencial de arrecadação).\n\n**Monitorar:** Aumento indica maior adesão; queda pode indicar quitações ou cancelamentos."
        )
    with col2:
        st.metric(
            "🟢 Quitados",
            f"{int(r['qtd_quitados']):,}",
            delta=formatar_valor_bilhoes(r['valor_quitados']),
            delta_color="off",
            help="🟢 **Parcelamentos Quitados**\n\nAcordos finalizados com sucesso - todas as parcelas foram pagas.\n\n**Valor mostrado abaixo:** Total já arrecadado através de quitações.\n\n**Indicador de sucesso:** Quanto maior, melhor a efetividade do programa."
        )
    with col3:
        st.metric(
            "🔴 Cancelados",
            f"{int(r['qtd_cancelados']):,}",
            delta=formatar_valor_bilhoes(r['valor_cancelados']),
            delta_color="off",
            help="🔴 **Parcelamentos Cancelados**\n\nAcordos cancelados, geralmente por inadimplência superior a 90 dias.\n\n**Valor mostrado abaixo:** Potencial de arrecadação perdido.\n\n**Atenção:** Valores retornam para cobrança regular ou dívida ativa."
        )
    with col4:
        st.metric(
            "🟡 Pendentes 1ª Parcela",
            f"{int(r['qtd_pendentes_1a_parcela']):,}",
            help="🟡 **Aguardando Primeira Parcela**\n\nParcelamentos contratados mas ainda sem pagamento inicial.\n\n**Risco:** Alta probabilidade de cancelamento se não regularizados.\n\n**Ação recomendada:** Contato proativo com contribuintes para regularização."
        )
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    serie = dados.get('serie_temporal', pd.DataFrame())
    
    with col1:
        st.markdown("### 📅 Evolução Mensal de Parcelamentos")
        if not serie.empty:
            # Últimos 24 meses
            serie_recente = serie.tail(24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=serie_recente['periodo_pedido'].astype(str),
                y=serie_recente['total_parcelamentos'],
                mode='lines+markers',
                name='Total',
                line=dict(color='#1976d2', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=serie_recente['periodo_pedido'].astype(str),
                y=serie_recente['qtd_quitados'],
                mode='lines+markers',
                name='Quitados',
                line=dict(color='#2e7d32', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=serie_recente['periodo_pedido'].astype(str),
                y=serie_recente['qtd_cancelados'],
                mode='lines+markers',
                name='Cancelados',
                line=dict(color='#c62828', width=2)
            ))
            
            fig.update_layout(
                height=400,
                xaxis_title="Período",
                yaxis_title="Quantidade",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Distribuição por Categoria")
        categorias = pd.DataFrame({
            'Categoria': ['ICMS', 'Dívida Ativa', 'Declarado', 'RECUPERA+'],
            'Quantidade': [
                int(r.get('qtd_parcel_icms', 0)),
                int(r.get('qtd_parcel_dva', 0)),
                int(r.get('qtd_parcel_declarado', 0)),
                int(r.get('total_recuperamais', 0))
            ],
            'Valor': [
                float(r.get('valor_parcel_icms', 0) or 0),
                float(r.get('valor_parcel_dva', 0) or 0),
                float(r.get('valor_parcel_declarado', 0) or 0),
                float(r.get('valor_recuperamais', 0) or 0)
            ]
        })
        
        fig = px.bar(
            categorias,
            x='Categoria',
            y='Quantidade',
            color='Categoria',
            color_discrete_sequence=['#1976d2', '#7b1fa2', '#388e3c', '#f57c00'],
            text='Quantidade'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Alertas e riscos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Alertas de Risco")
        
        st.markdown(f"""
        <div class='risco-critico'>
            <strong>🔴 RISCO CRÍTICO</strong><br>
            {int(r.get('empresas_risco_critico', 0)):,} empresas<br>
            Valor: {formatar_valor_bilhoes(r.get('valor_empresas_alto_risco', 0))}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='risco-alto'>
            <strong>🟠 RISCO ALTO</strong><br>
            {int(r.get('empresas_risco_alto', 0)):,} empresas<br>
            Alertas urgentes: {int(r.get('alertas_urgentes', 0)):,}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='warning-box'>
            <strong>⚠️ EMPRESAS REINCIDENTES</strong><br>
            {int(r.get('empresas_reincidentes', 0)):,} empresas com 3+ cancelamentos<br>
            {int(r.get('empresas_nunca_quitaram', 0)):,} nunca quitaram um parcelamento
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💰 Expectativa de Recebimento")
        
        st.markdown(f"""
        <div class='info-box'>
            <strong>📊 Próximos 3 meses</strong><br>
            Esperado: {formatar_valor_milhoes(r.get('expectativa_3_meses', 0))}<br>
            Ajustado por risco: {formatar_valor_milhoes(r.get('expectativa_ajustada_3_meses', 0))}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='info-box'>
            <strong>📊 Próximos 12 meses</strong><br>
            Esperado: {formatar_valor_milhoes(r.get('expectativa_12_meses', 0))}<br>
            Ajustado por risco: {formatar_valor_milhoes(r.get('expectativa_ajustada_12_meses', 0))}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='success-box'>
            <strong>✅ RECUPERA+</strong><br>
            Total: {int(r.get('total_recuperamais', 0)):,} parcelamentos<br>
            Ativos: {int(r.get('recuperamais_ativos', 0)):,} | 
            Quitados: {int(r.get('recuperamais_quitados', 0)):,}<br>
            Taxa sucesso: {r.get('taxa_sucesso_recuperamais_pct', 0):.1f}%
        </div>
        """, unsafe_allow_html=True)


def pagina_analise_temporal(dados, filtros_globais):
    """Página de análise temporal."""
    st.markdown("<h1 class='main-header'>📅 Análise Temporal</h1>", unsafe_allow_html=True)

    serie = dados.get('serie_temporal', pd.DataFrame())

    if serie.empty:
        st.warning("Dados de série temporal não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como usar a análise temporal", expanded=False):
        st.markdown("""
        ### Objetivo
        Acompanhar a evolução dos parcelamentos ao longo do tempo, identificando tendências e sazonalidades.

        ### Gráficos disponíveis
        | Gráfico | O que mostra | Uso |
        |---------|--------------|-----|
        | **Quantidade por Status** | Evolução de ativos, quitados e cancelados | Identificar tendências |
        | **Valor por Status** | Montantes quitados vs cancelados | Avaliar impacto financeiro |
        | **Taxa de Sucesso** | Evolução da efetividade | Monitorar performance |
        | **Ticket Médio** | Valor médio por parcelamento | Identificar mudanças de perfil |

        ### Como usar os filtros
        1. **Ano:** Selecione um ano específico ou "Todos" para visão completa
        2. **Categoria:** Filtre por tipo de parcelamento

        ### Dicas de análise
        - Picos em determinados meses podem indicar campanhas ou vencimentos
        - Queda na taxa de sucesso requer investigação
        - Aumento do ticket médio pode indicar parcelamentos de maior valor
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Use os filtros para segmentar a análise. Passe o mouse sobre os gráficos para ver valores detalhados.</span>
    </div>
    """, unsafe_allow_html=True)

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        anos = sorted(serie['ano_pedido'].unique())
        ano_selecionado = st.selectbox(
            "Ano",
            ["Todos"] + list(anos),
            index=0,
            help="Selecione um ano específico para análise focada ou 'Todos' para visão histórica completa."
        )
    with col2:
        categorias = ['Todas', 'ICMS', 'Dívida Ativa', 'Declarado', 'RECUPERA+']
        categoria_selecionada = st.selectbox(
            "Categoria",
            categorias,
            help="Filtre por categoria de parcelamento. Cada categoria tem características e taxas de sucesso diferentes."
        )
    
    # Aplicar filtros
    df = serie.copy()
    if ano_selecionado != "Todos":
        df = df[df['ano_pedido'] == ano_selecionado]
    
    st.markdown("---")
    
    # Gráfico de evolução
    st.markdown("### 📈 Evolução de Parcelamentos")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Quantidade por Status', 'Valor por Status', 
                       'Taxa de Sucesso', 'Ticket Médio'],
        vertical_spacing=0.12
    )
    
    # Quantidade por status
    fig.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['total_parcelamentos'],
        name='Total', line=dict(color='#1976d2')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_quitados'],
        name='Quitados', line=dict(color='#2e7d32')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_cancelados'],
        name='Cancelados', line=dict(color='#c62828')
    ), row=1, col=1)
    
    # Valor por status
    fig.add_trace(go.Bar(
        x=df['periodo_pedido'].astype(str), y=df['valor_quitado']/1e6,
        name='Valor Quitado (MM)', marker_color='#2e7d32'
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=df['periodo_pedido'].astype(str), y=df['valor_cancelado']/1e6,
        name='Valor Cancelado (MM)', marker_color='#c62828'
    ), row=1, col=2)
    
    # Taxa de sucesso
    fig.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['taxa_sucesso_pct'],
        name='Taxa Sucesso %', line=dict(color='#7b1fa2'), fill='tozeroy'
    ), row=2, col=1)
    
    # Ticket médio
    fig.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['ticket_medio']/1000,
        name='Ticket Médio (mil)', line=dict(color='#f57c00')
    ), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Por categoria
    st.markdown("### 🏷️ Evolução por Categoria")
    
    fig_cat = go.Figure()
    fig_cat.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_icms'],
        name='ICMS', stackgroup='one'
    ))
    fig_cat.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_dva'],
        name='Dívida Ativa', stackgroup='one'
    ))
    fig_cat.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_declarado'],
        name='Declarado', stackgroup='one'
    ))
    fig_cat.add_trace(go.Scatter(
        x=df['periodo_pedido'].astype(str), y=df['qtd_recuperamais'],
        name='RECUPERA+', stackgroup='one'
    ))
    
    fig_cat.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_cat, use_container_width=True)
    
    # Tabela resumo
    st.markdown("### 📋 Dados Detalhados")
    
    df_display = df[['periodo_pedido', 'total_parcelamentos', 'qtd_ativos', 
                     'qtd_quitados', 'qtd_cancelados', 'taxa_sucesso_pct',
                     'valor_total', 'ticket_medio']].copy()
    df_display.columns = ['Período', 'Total', 'Ativos', 'Quitados', 'Cancelados',
                          'Taxa Sucesso %', 'Valor Total', 'Ticket Médio']
    df_display['Valor Total'] = df_display['Valor Total'].apply(lambda x: formatar_valor_milhoes(x))
    df_display['Ticket Médio'] = df_display['Ticket Médio'].apply(lambda x: formatar_valor_br(x))
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def pagina_analise_regional(dados, filtros_globais):
    """Página de análise por regional (GERFE)."""
    st.markdown("<h1 class='main-header'>🗺️ Análise por Regional (GERFE)</h1>", unsafe_allow_html=True)

    df_gerfe = dados.get('metricas_gerfe', pd.DataFrame())

    if df_gerfe.empty:
        st.warning("Dados de regionais não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como interpretar a análise regional", expanded=False):
        st.markdown("""
        ### O que é GERFE?
        **Gerência Regional da Fazenda Estadual** - unidades descentralizadas da SEF/SC responsáveis
        pela fiscalização e atendimento em cada região do estado.

        ### Indicadores desta página
        | Indicador | Significado | Uso |
        |-----------|-------------|-----|
        | **Taxa de Sucesso** | % de parcelamentos quitados vs finalizados | Comparar efetividade entre regionais |
        | **Empresas Alto Risco** | Empresas com score ≥ 60 | Priorizar ações de cobrança |
        | **Valor por Regional** | Montante parcelado por região | Identificar concentração de débitos |

        ### Como usar
        1. Compare as taxas de sucesso entre regionais
        2. Identifique regionais com baixa performance para ações focadas
        3. Use o treemap para visualizar a distribuição de valores
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Use o seletor ao final da página para ver detalhes de cada regional. O treemap mostra proporcionalmente o valor por região.</span>
    </div>
    """, unsafe_allow_html=True)

    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Regionais",
            len(df_gerfe),
            help="🏛️ **Regionais Ativas**\n\nNúmero de GERFEs (Gerências Regionais da Fazenda Estadual) com parcelamentos registrados.\n\n**Cobertura:** Abrange todo o estado de Santa Catarina."
        )
    with col2:
        melhor_gerfe = df_gerfe.loc[df_gerfe['taxa_sucesso_pct'].idxmax(), 'gerfe']
        st.metric(
            "Maior Taxa Sucesso",
            f"{df_gerfe['taxa_sucesso_pct'].max():.1f}%",
            help=f"🏆 **Melhor Performance**\n\nRegional com maior taxa de sucesso em parcelamentos.\n\n**Regional:** {melhor_gerfe}\n\n**Benchmark:** Esta regional pode servir de referência para boas práticas."
        )
    with col3:
        pior_gerfe = df_gerfe.loc[df_gerfe['taxa_sucesso_pct'].idxmin(), 'gerfe']
        st.metric(
            "Menor Taxa Sucesso",
            f"{df_gerfe['taxa_sucesso_pct'].min():.1f}%",
            help=f"⚠️ **Menor Performance**\n\nRegional com menor taxa de sucesso em parcelamentos.\n\n**Regional:** {pior_gerfe}\n\n**Ação:** Avaliar causas e implementar melhorias."
        )
    with col4:
        st.metric(
            "Empresas Alto Risco",
            f"{df_gerfe['empresas_alto_risco'].sum():,}",
            help="🎯 **Empresas de Alto Risco**\n\nTotal de empresas com score de risco ≥ 60 em todas as regionais.\n\n**Classificação:**\n- Score 60-79: Risco Alto\n- Score ≥ 80: Risco Crítico\n\n**Ação:** Priorizar para ações preventivas de cobrança."
        )
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Taxa de Sucesso por Regional")
        
        df_sorted = df_gerfe.sort_values('taxa_sucesso_pct', ascending=True)
        
        fig = px.bar(
            df_sorted,
            x='taxa_sucesso_pct',
            y='gerfe',
            orientation='h',
            color='taxa_sucesso_pct',
            color_continuous_scale='RdYlGn',
            text='taxa_sucesso_pct'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Valor por Regional")
        
        fig = px.treemap(
            df_gerfe,
            path=['gerfe'],
            values='valor_total',
            color='taxa_sucesso_pct',
            color_continuous_scale='RdYlGn',
            hover_data=['qtd_ativos', 'empresas_distintas']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Ranking
    st.markdown("### 🏆 Ranking de Regionais")
    
    df_rank = df_gerfe[['gerfe', 'total_parcelamentos', 'empresas_distintas',
                        'qtd_ativos', 'qtd_quitados', 'qtd_cancelados',
                        'taxa_sucesso_pct', 'valor_total', 'empresas_alto_risco']].copy()
    df_rank.columns = ['Regional', 'Total', 'Empresas', 'Ativos', 'Quitados',
                       'Cancelados', 'Taxa Sucesso %', 'Valor Total', 'Alto Risco']
    df_rank['Valor Total'] = df_rank['Valor Total'].apply(lambda x: formatar_valor_milhoes(x))
    df_rank = df_rank.sort_values('Taxa Sucesso %', ascending=False)
    
    st.dataframe(df_rank, use_container_width=True, hide_index=True)
    
    # Drilldown
    st.markdown("---")
    st.markdown("### 🔍 Detalhamento por Regional")
    
    gerfe_selecionada = st.selectbox(
        "Selecione uma regional para ver detalhes",
        df_gerfe['gerfe'].tolist()
    )
    
    if gerfe_selecionada:
        cd_gerfe = df_gerfe[df_gerfe['gerfe'] == gerfe_selecionada]['cd_gerfe'].iloc[0]
        gerfe_info = df_gerfe[df_gerfe['gerfe'] == gerfe_selecionada].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Parcelamentos",
                f"{int(gerfe_info['total_parcelamentos']):,}",
                help="📊 **Total na Regional**\n\nQuantidade de parcelamentos registrados nesta GERFE, incluindo todos os status."
            )
        with col2:
            st.metric(
                "Taxa de Sucesso",
                f"{gerfe_info['taxa_sucesso_pct']:.1f}%",
                help="✅ **Taxa de Sucesso Regional**\n\nPercentual de parcelamentos quitados em relação aos finalizados (quitados + cancelados).\n\n**Compare:** Com a média estadual para avaliar performance."
            )
        with col3:
            st.metric(
                "Valor Ativo",
                formatar_valor_milhoes(gerfe_info['valor_ativos']),
                help="💰 **Valor em Parcelamentos Ativos**\n\nMontante total em parcelamentos em andamento nesta regional.\n\n**Representa:** Potencial de arrecadação futura."
            )
        
        # Carregar parcelamentos da regional
        if st.button("📋 Carregar Parcelamentos"):
            with st.spinner("Carregando..."):
                engine = get_impala_engine()
                df_parcel = carregar_parcelamentos_gerfe(engine, cd_gerfe)
                
                if not df_parcel.empty:
                    st.dataframe(df_parcel, use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhum parcelamento encontrado.")


def pagina_analise_setorial(dados, filtros_globais):
    """Página de análise por setor (GES)."""
    st.markdown("<h1 class='main-header'>🏭 Análise por Setor (GES)</h1>", unsafe_allow_html=True)

    df_ges = dados.get('metricas_ges', pd.DataFrame())

    if df_ges.empty:
        st.warning("Dados setoriais não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como interpretar a análise setorial", expanded=False):
        st.markdown("""
        ### O que é GES?
        **Gerência de Gestão Setorial** - unidades especializadas por segmento econômico na SEF/SC,
        responsáveis pelo acompanhamento de setores específicos da economia.

        ### Análise CNAE
        O **CNAE** (Classificação Nacional de Atividades Econômicas) permite identificar
        quais setores da economia têm maior volume de parcelamentos e sua taxa de sucesso.

        ### Como usar esta análise
        1. **Identificar setores problemáticos:** Baixa taxa de sucesso indica necessidade de atenção
        2. **Priorizar ações:** Focar em setores com alto valor parcelado e baixa taxa de sucesso
        3. **Benchmarking:** Comparar desempenho entre setores similares
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>O gráfico de pizza mostra a distribuição de valores. Passe o mouse para ver detalhes de cada setor.</span>
    </div>
    """, unsafe_allow_html=True)

    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Taxa de Sucesso por Setor")
        
        df_sorted = df_ges.sort_values('taxa_sucesso_pct', ascending=True)
        
        fig = px.bar(
            df_sorted,
            x='taxa_sucesso_pct',
            y='ges',
            orientation='h',
            color='taxa_sucesso_pct',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Valor por Setor")
        
        fig = px.pie(
            df_ges,
            values='valor_total',
            names='ges',
            hole=0.4
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Por CNAE
    st.markdown("### 🏢 Análise por Setor Econômico (CNAE)")
    
    df_cnae = dados.get('analise_cnae', pd.DataFrame())
    
    if not df_cnae.empty:
        fig = px.bar(
            df_cnae.head(15),
            x='valor_total',
            y='descricao_secao',
            orientation='h',
            color='taxa_sucesso_pct',
            color_continuous_scale='RdYlGn',
            hover_data=['total_parcelamentos', 'empresas_distintas']
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def pagina_empresas_risco(dados, filtros_globais):
    """Página de empresas por risco."""
    st.markdown("<h1 class='main-header'>🎯 Análise de Risco - Empresas</h1>", unsafe_allow_html=True)

    df_empresas = dados.get('empresas_score', pd.DataFrame())

    if df_empresas.empty:
        st.warning("Dados de empresas não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como funciona o Score de Risco", expanded=False):
        st.markdown("""
        ### Metodologia do Score
        O score de risco varia de **0 a 100** (maior = mais arriscado) e é calculado com base em 5 componentes:

        | Componente | Peso | O que mede |
        |------------|------|------------|
        | **Histórico de Cancelamentos** | 30% | Taxa de cancelamento passada |
        | **Valor em Risco** | 25% | Montante em parcelamentos ativos |
        | **Recência** | 20% | Tempo desde último parcelamento |
        | **Reincidência** | 15% | Se tem 3+ cancelamentos |
        | **Comportamento** | 10% | Padrão de pagamento histórico |

        ### Classificações de Risco
        - 🔴 **CRÍTICO (≥80):** Requer ação imediata - alta probabilidade de cancelamento
        - 🟠 **ALTO (60-79):** Requer atenção - monitoramento frequente
        - 🟡 **MÉDIO (40-59):** Monitoramento regular
        - 🟢 **BAIXO (<40):** Baixo risco - acompanhamento padrão

        ### Como usar os filtros
        1. Selecione as classificações de risco desejadas
        2. Defina um score mínimo para focar em casos mais críticos
        3. Use o filtro de valor para priorizar por impacto financeiro
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Use os filtros abaixo para segmentar empresas. Clique nas colunas da tabela para ordenar.</span>
    </div>
    """, unsafe_allow_html=True)

    # Filtros com tooltips
    col1, col2, col3 = st.columns(3)

    with col1:
        riscos = df_empresas['classificacao_risco'].unique().tolist()
        risco_filtro = st.multiselect(
            "Classificação de Risco",
            riscos,
            default=riscos,
            help="Filtre por nível de risco. Selecione múltiplas classificações para comparar."
        )
    with col2:
        score_min = st.slider(
            "Score Mínimo",
            0, 100, 0,
            help="Defina o score mínimo para exibição. Valores mais altos mostram apenas empresas de maior risco."
        )
    with col3:
        valor_min = st.number_input(
            "Valor Ativo Mínimo (R$)",
            0, 100000000, 0,
            help="Filtre empresas com valor mínimo em parcelamentos ativos. Útil para priorizar por impacto financeiro."
        )

    # Aplicar filtros
    df = df_empresas[
        (df_empresas['classificacao_risco'].isin(risco_filtro)) &
        (df_empresas['score_risco_final'] >= score_min) &
        (df_empresas['valor_ativos'] >= valor_min)
    ].copy()

    st.markdown("---")

    # Legenda de classificações
    st.markdown("""
    <div class='indicator-legend'>
        <div class='indicator-legend-title'>🎯 Classificações de Risco</div>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #c62828;'></span>Crítico (≥80): Ação imediata</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #ef6c00;'></span>Alto (60-79): Requer atenção</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #f9a825;'></span>Médio (40-59): Monitorar</span>
        <span class='indicator-item'><span class='indicator-dot' style='background-color: #2e7d32;'></span>Baixo (<40): Baixo risco</span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Empresas Filtradas",
            f"{len(df):,}",
            help="📊 **Resultado do Filtro**\n\nQuantidade de empresas que atendem aos critérios selecionados.\n\n**Dica:** Ajuste os filtros para refinar sua análise."
        )
    with col2:
        st.metric(
            "Score Médio",
            f"{df['score_risco_final'].mean():.1f}" if not df.empty else "N/A",
            help="📈 **Score Médio do Grupo**\n\nMédia do score de risco das empresas filtradas.\n\n**Interpretação:**\n- < 40: Grupo de baixo risco\n- 40-60: Grupo moderado\n- > 60: Grupo de alto risco"
        )
    with col3:
        st.metric(
            "Valor em Risco",
            formatar_valor_milhoes(df['valor_ativos'].sum()),
            help="💰 **Valor Total em Risco**\n\nSoma dos valores em parcelamentos ativos das empresas filtradas.\n\n**Representa:** Potencial de arrecadação que pode ser perdido se houver cancelamentos."
        )
    with col4:
        reincidentes = df['flag_reincidente'].sum() if not df.empty else 0
        st.metric(
            "Reincidentes",
            f"{int(reincidentes):,}",
            help="🔄 **Empresas Reincidentes**\n\nEmpresas com 3 ou mais parcelamentos cancelados no histórico.\n\n**Risco elevado:** Padrão de comportamento indica maior probabilidade de novos cancelamentos."
        )
    
    st.markdown("---")
    
    # Distribuição por risco
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribuição por Risco")
        
        dist_risco = df['classificacao_risco'].value_counts().reset_index()
        dist_risco.columns = ['Risco', 'Quantidade']
        
        fig = px.pie(
            dist_risco,
            values='Quantidade',
            names='Risco',
            color='Risco',
            color_discrete_map={
                'CRÍTICO': '#c62828',
                'ALTO': '#ef6c00',
                'MÉDIO': '#f9a825',
                'BAIXO': '#2e7d32'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Componentes do Score")
        
        componentes = df[['score_historico_cancelamento', 'score_recencia',
                          'score_valor_risco', 'score_reincidencia', 
                          'score_comportamento']].mean()
        
        fig = go.Figure(go.Bar(
            x=componentes.values,
            y=['Histórico Cancel.', 'Recência', 'Valor em Risco', 
               'Reincidência', 'Comportamento'],
            orientation='h',
            marker_color=['#1976d2', '#7b1fa2', '#388e3c', '#f57c00', '#c62828']
        ))
        fig.update_layout(height=400, xaxis_title="Score Médio")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Lista de empresas
    st.markdown("### 📋 Empresas por Score de Risco")
    
    df_display = df[['cnpj', 'razao_social', 'classificacao_risco', 
                     'score_risco_final', 'total_parcelamentos', 
                     'qtd_ativos', 'qtd_cancelados', 'valor_ativos',
                     'taxa_sucesso_pct', 'gerfe']].copy()
    df_display.columns = ['CNPJ', 'Razão Social', 'Risco', 'Score',
                          'Total Parcel.', 'Ativos', 'Cancelados', 
                          'Valor Ativo', 'Taxa Sucesso %', 'GERFE']
    df_display['CNPJ'] = df_display['CNPJ'].apply(formatar_cnpj_visualizacao)
    df_display['Valor Ativo'] = df_display['Valor Ativo'].apply(lambda x: formatar_valor_br(x))
    df_display = df_display.sort_values('Score', ascending=False)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)


def pagina_drilldown_empresa(dados, filtros_globais):
    """Página de drilldown por empresa."""
    st.markdown("<h1 class='main-header'>🔍 Análise Detalhada - Empresa</h1>", unsafe_allow_html=True)

    df_empresas = dados.get('empresas_score', pd.DataFrame())

    if df_empresas.empty:
        st.warning("Dados de empresas não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como usar esta análise", expanded=False):
        st.markdown("""
        ### Visão 360° da Empresa
        Esta página oferece uma visão completa de uma empresa específica, incluindo:

        - **Dados cadastrais:** CNPJ, razão social, regional e setor
        - **Score de risco:** Pontuação e classificação atual
        - **Histórico completo:** Todos os parcelamentos da empresa
        - **Alertas ativos:** Situações que requerem atenção
        - **Estatísticas:** Taxa de sucesso, valores e comportamento

        ### Como usar
        1. Selecione uma empresa no campo de busca
        2. Analise o score e a classificação de risco
        3. Verifique os alertas ativos
        4. Consulte o histórico de parcelamentos

        ### Dicas
        - O score mostrado é dinâmico e atualizado diariamente
        - Empresas ordenadas por score (maior risco primeiro)
        - Timeline mostra a evolução dos parcelamentos
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Digite parte do nome ou CNPJ para filtrar. Empresas estão ordenadas por score de risco (maior primeiro).</span>
    </div>
    """, unsafe_allow_html=True)

    # Seletor de empresa
    opcoes = df_empresas.apply(
        lambda x: f"{x['razao_social']} - {formatar_cnpj_visualizacao(x['cnpj'])} (Score: {x['score_risco_final']:.0f})",
        axis=1
    ).tolist()

    empresa_sel = st.selectbox(
        "🔎 Buscar empresa:",
        opcoes[:200],
        help="Selecione uma empresa para ver análise detalhada. Lista ordenada por score de risco (maior primeiro)."
    )
    
    if empresa_sel:
        cnpj_raiz = df_empresas.iloc[opcoes.index(empresa_sel)]['cnpj_raiz']
        
        # Carregar detalhes
        with st.spinner("Carregando detalhes..."):
            engine = get_impala_engine()
            detalhes = carregar_detalhes_empresa(engine, cnpj_raiz)
        
        if detalhes:
            empresa = detalhes.get('empresa', pd.DataFrame())
            score = detalhes.get('score', pd.DataFrame())
            parcelamentos = detalhes.get('parcelamentos', pd.DataFrame())
            alertas = detalhes.get('alertas', pd.DataFrame())
            
            st.markdown("---")
            
            if not empresa.empty:
                emp = empresa.iloc[0]
                
                # Cabeçalho
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 🏢 {emp['razao_social']}")
                    st.markdown(f"**CNPJ:** {formatar_cnpj_visualizacao(emp['cnpj'])}")
                    st.markdown(f"**GERFE:** {emp['gerfe']} | **GES:** {emp['ges']}")
                    st.markdown(f"**Situação:** {emp['situacao_cadastral']}")
                
                with col2:
                    if not score.empty:
                        sc = score.iloc[0]
                        cor = get_cor_risco(sc['classificacao_risco'])
                        st.markdown(f"""
                        <div style='background-color: {cor}20; border: 3px solid {cor}; 
                                    padding: 20px; border-radius: 10px; text-align: center;'>
                            <h2 style='color: {cor}; margin: 0;'>{sc['score_risco_final']:.0f}</h2>
                            <p style='margin: 5px 0 0 0;'><strong>{sc['classificacao_risco']}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # KPIs
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(
                        "Total Parcelamentos",
                        f"{int(emp['total_parcelamentos']):,}",
                        help="📊 **Histórico Total**\n\nQuantidade total de parcelamentos já realizados por esta empresa."
                    )
                with col2:
                    st.metric(
                        "🔵 Ativos",
                        f"{int(emp['qtd_ativos']):,}",
                        help="🔵 **Em Andamento**\n\nParcelamentos atualmente em andamento com pagamentos regulares."
                    )
                with col3:
                    st.metric(
                        "🟢 Quitados",
                        f"{int(emp['qtd_quitados']):,}",
                        help="🟢 **Finalizados com Sucesso**\n\nParcelamentos que foram completamente pagos pela empresa."
                    )
                with col4:
                    st.metric(
                        "🔴 Cancelados",
                        f"{int(emp['qtd_cancelados']):,}",
                        help="🔴 **Cancelados**\n\nParcelamentos cancelados por inadimplência.\n\n**Atenção:** Histórico de cancelamentos afeta o score de risco."
                    )
                with col5:
                    st.metric(
                        "Taxa Sucesso",
                        f"{emp['taxa_sucesso_pct']:.1f}%",
                        help="✅ **Taxa de Sucesso Individual**\n\n**Fórmula:** Quitados ÷ (Quitados + Cancelados)\n\n**Interpretação:**\n- ≥70%: Bom histórico\n- 50-70%: Moderado\n- <50%: Atenção"
                    )

                st.markdown("---")

                # Valores
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Valor Total Parcelado",
                        formatar_valor_milhoes(emp['valor_total_parcelado']),
                        help="💰 **Valor Histórico**\n\nSoma de todos os valores já parcelados pela empresa ao longo do tempo."
                    )
                with col2:
                    st.metric(
                        "Valor Ativo",
                        formatar_valor_milhoes(emp['valor_ativos']),
                        help="💵 **Valor em Aberto**\n\nMontante em parcelamentos atualmente ativos.\n\n**Representa:** Potencial de arrecadação futura."
                    )
                with col3:
                    st.metric(
                        "Ticket Médio",
                        formatar_valor_br(emp['valor_medio_parcelamento']),
                        help="📈 **Valor Médio por Parcelamento**\n\nMédia dos valores parcelados.\n\n**Uso:** Comparar com média geral para entender perfil da empresa."
                    )
                
                st.markdown("---")
                
                # Alertas
                if not alertas.empty:
                    st.markdown("### ⚠️ Alertas Ativos")
                    for _, alerta in alertas.iterrows():
                        cor_alerta = '#c62828' if alerta['prioridade'] == 1 else '#ef6c00' if alerta['prioridade'] == 2 else '#f9a825'
                        st.markdown(f"""
                        <div style='background-color: {cor_alerta}20; border-left: 4px solid {cor_alerta}; 
                                    padding: 10px; margin: 5px 0; border-radius: 5px;'>
                            <strong>{alerta['tipo_alerta']}</strong> - {alerta['desc_prioridade']}<br>
                            Parcelamento: {alerta['num_parcelamento']} | Valor: {formatar_valor_br(alerta['valor_parcelado'])}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Histórico de parcelamentos
                st.markdown("### 📋 Histórico de Parcelamentos")
                
                if not parcelamentos.empty:
                    df_hist = parcelamentos[['num_parcelamento', 'categoria_parcelamento',
                                            'desc_status', 'valor_parcelado', 'num_parcelas',
                                            'dt_pedido', 'faixa_parcelas']].copy()
                    df_hist.columns = ['Nº Parcelamento', 'Categoria', 'Status', 
                                      'Valor', 'Parcelas', 'Data Pedido', 'Faixa']
                    df_hist['Valor'] = df_hist['Valor'].apply(lambda x: formatar_valor_br(x))
                    
                    st.dataframe(df_hist, use_container_width=True, hide_index=True)
                    
                    # Gráfico de evolução
                    fig = px.timeline(
                        parcelamentos,
                        x_start='dt_pedido',
                        x_end=parcelamentos['dt_pedido'] + pd.Timedelta(days=30),
                        y='num_parcelamento',
                        color='desc_status',
                        hover_data=['valor_parcelado', 'num_parcelas']
                    )
                    st.plotly_chart(fig, use_container_width=True)


def pagina_alertas(dados, filtros_globais):
    """Página de alertas."""
    st.markdown("<h1 class='main-header'>🚨 Central de Alertas</h1>", unsafe_allow_html=True)

    alertas = dados.get('alertas', pd.DataFrame())

    if alertas.empty:
        st.success("✅ Nenhum alerta ativo no momento.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como funcionam os alertas", expanded=False):
        st.markdown("""
        ### Sistema de Alertas
        Os alertas são gerados automaticamente para identificar parcelamentos que requerem atenção especial.

        ### Níveis de Prioridade
        | Prioridade | Cor | Significado | Ação |
        |------------|-----|-------------|------|
        | **URGENTE** | 🔴 Vermelho | Risco iminente de cancelamento | Ação imediata |
        | **ALTA** | 🟠 Laranja | Alto risco identificado | Ação em 48h |
        | **MÉDIA** | 🟡 Amarelo | Monitoramento necessário | Acompanhar |
        | **BAIXA** | 🟢 Verde | Atenção preventiva | Verificar |

        ### Tipos de Alertas
        - **Risco de Cancelamento:** Empresa com histórico de inadimplência
        - **Valor Elevado:** Parcelamento de grande valor em risco
        - **Reincidência:** Empresa com múltiplos cancelamentos
        - **Atraso de Pagamento:** Parcelas em atraso

        ### Como usar
        1. Priorize alertas URGENTES
        2. Use os filtros para segmentar
        3. Acesse o drill-down para detalhes da empresa
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Priorize os alertas URGENTES (vermelho). Use os filtros para focar em tipos específicos de alertas.</span>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Alertas",
            len(alertas),
            help="📊 **Alertas Ativos**\n\nQuantidade total de alertas que requerem atenção.\n\n**Meta:** Reduzir progressivamente através de ações proativas."
        )
    with col2:
        urgentes = len(alertas[alertas['prioridade'] == 1])
        st.metric(
            "🔴 Urgentes",
            urgentes,
            delta="Prioridade 1",
            delta_color="inverse",
            help="🔴 **Alertas Urgentes**\n\nParcelamentos com risco iminente de cancelamento.\n\n**SLA:** Ação em até 24 horas.\n\n**Critérios:** Score ≥80, valor elevado, ou histórico crítico."
        )
    with col3:
        alta = len(alertas[alertas['prioridade'] == 2])
        st.metric(
            "🟠 Alta Prioridade",
            alta,
            help="🟠 **Alta Prioridade**\n\nParcelamentos com alto risco identificado.\n\n**SLA:** Ação em até 48 horas.\n\n**Critérios:** Score 60-79, valor significativo."
        )
    with col4:
        valor = alertas['valor_parcelado'].sum()
        st.metric(
            "Valor em Risco",
            formatar_valor_milhoes(valor),
            help="💰 **Valor Total em Alerta**\n\nSoma dos valores dos parcelamentos com alertas ativos.\n\n**Representa:** Potencial de perda se não houver ação."
        )
    
    st.markdown("---")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        prioridades = alertas['desc_prioridade'].unique().tolist()
        prioridade_sel = st.multiselect(
            "Prioridade",
            prioridades,
            default=prioridades,
            help="Filtre por nível de prioridade. Comece pelos URGENTES para ações imediatas."
        )
    with col2:
        tipos = alertas['tipo_alerta'].unique().tolist()
        tipo_sel = st.multiselect(
            "Tipo de Alerta",
            tipos,
            default=tipos,
            help="Filtre por tipo de alerta. Útil para análises específicas ou ações em lote."
        )
    
    # Aplicar filtros
    df = alertas[
        (alertas['desc_prioridade'].isin(prioridade_sel)) &
        (alertas['tipo_alerta'].isin(tipo_sel))
    ]
    
    st.markdown("---")
    
    # Distribuição
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Por Prioridade")
        dist_prioridade = df['desc_prioridade'].value_counts().reset_index()
        dist_prioridade.columns = ['Prioridade', 'Quantidade']
        
        fig = px.pie(
            dist_prioridade,
            values='Quantidade',
            names='Prioridade',
            color='Prioridade',
            color_discrete_map={
                'URGENTE': '#c62828',
                'ALTA': '#ef6c00',
                'MÉDIA': '#f9a825',
                'BAIXA': '#2e7d32'
            }
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Por Tipo")
        dist_tipo = df['tipo_alerta'].value_counts().reset_index()
        dist_tipo.columns = ['Tipo', 'Quantidade']
        
        fig = px.bar(
            dist_tipo,
            x='Quantidade',
            y='Tipo',
            orientation='h',
            color='Quantidade',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Lista de alertas
    st.markdown("### 📋 Lista de Alertas")
    
    df_display = df[['desc_prioridade', 'tipo_alerta', 'razao_social',
                     'cnpj', 'valor_parcelado', 'dias_vida_parcelamento',
                     'classificacao_risco', 'gerfe']].copy()
    df_display.columns = ['Prioridade', 'Tipo', 'Empresa', 'CNPJ',
                          'Valor', 'Dias Ativo', 'Risco', 'GERFE']
    df_display['CNPJ'] = df_display['CNPJ'].apply(formatar_cnpj_visualizacao)
    df_display['Valor'] = df_display['Valor'].apply(lambda x: formatar_valor_br(x))
    
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)


def pagina_recuperamais(dados, filtros_globais):
    """Página de análise RECUPERA+."""
    st.markdown("<h1 class='main-header'>🔄 RECUPERA+ - Programa de Recuperação</h1>", unsafe_allow_html=True)

    # Info do programa
    with st.expander("ℹ️ Sobre o RECUPERA+", expanded=False):
        st.markdown("""
        ### Lei nº 18.819/2024 - Programa de Recuperação de Créditos Ampliado

        **Base legal:** Convênio ICMS nº 113/2023 (CONFAZ)

        ### Abrangência
        - Débitos de ICMS com fatos geradores até 31/12/2022
        - Período de adesão: Janeiro/2024 a Maio/2024

        ### Benefícios Concedidos
        | Modalidade | Redução Juros/Multas | Parcelas |
        |------------|---------------------|----------|
        | **Cota Única** | Até 95% | À vista |
        | **Parcelado** | Até 80% | Até 120x |

        ### Origens do Débito
        - **DECLARADO:** ICMS declarado pelo contribuinte
        - **DEFESA PRÉVIA:** Termo de intimação defesa prévia
        - **NOTIFICAÇÃO FISCAL:** Auto de infração
        - **DÍVIDA ATIVA:** Débitos inscritos em DVA

        ### Objetivo
        Oportunizar a regularização de pendências fiscais com benefícios significativos,
        promovendo a recuperação de créditos tributários do Estado.
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>O RECUPERA+ é um programa especial com período de adesão encerrado. Esta análise mostra o desempenho dos parcelamentos contratados.</span>
    </div>
    """, unsafe_allow_html=True)

    resumo = dados.get('recuperamais_resumo', pd.DataFrame())

    if resumo.empty:
        st.warning("Dados do RECUPERA+ não disponíveis.")
        return

    # KPIs
    total = resumo.groupby('tipo_adesao').agg({
        'qtd': 'sum',
        'empresas': 'sum',
        'valor_total': 'sum'
    }).reset_index()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Adesões",
            f"{resumo['qtd'].sum():,}",
            help="📊 **Adesões ao Programa**\n\nQuantidade total de parcelamentos contratados no RECUPERA+.\n\n**Inclui:** Cota única e parcelamentos."
        )
    with col2:
        st.metric(
            "Empresas Beneficiadas",
            f"{resumo['empresas'].sum():,}",
            help="🏢 **Empresas Participantes**\n\nNúmero de empresas que aderiram ao programa.\n\n**Benefício:** Regularização fiscal com descontos significativos."
        )
    with col3:
        st.metric(
            "Valor Total",
            formatar_valor_bilhoes(resumo['valor_total'].sum()),
            help="💰 **Valor Total do Programa**\n\nSoma dos valores parcelados/pagos através do RECUPERA+.\n\n**Representa:** Volume de créditos tributários em recuperação."
        )
    with col4:
        quitados = resumo[resumo['desc_status'] == 'QUITADO']['qtd'].sum()
        cancelados = resumo[resumo['desc_status'] == 'CANCELADO']['qtd'].sum()
        taxa = (quitados / (quitados + cancelados) * 100) if (quitados + cancelados) > 0 else 0
        st.metric(
            "Taxa Sucesso",
            f"{taxa:.1f}%",
            help="✅ **Taxa de Sucesso do Programa**\n\n**Fórmula:** Quitados ÷ (Quitados + Cancelados)\n\n**Objetivo:** Manter acima de 70%.\n\n**Nota:** Cotas únicas tendem a ter taxa de 100%."
        )
    
    st.markdown("---")
    
    # Por tipo de adesão
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Por Tipo de Adesão")
        
        fig = px.pie(
            total,
            values='qtd',
            names='tipo_adesao',
            color='tipo_adesao',
            color_discrete_map={
                'COTA ÚNICA': '#2e7d32',
                'PARCELADO': '#1976d2'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Por Status")
        
        status_total = resumo.groupby('desc_status')['qtd'].sum().reset_index()
        
        fig = px.pie(
            status_total,
            values='qtd',
            names='desc_status',
            color='desc_status',
            color_discrete_map={
                'ATIVO': '#1976d2',
                'QUITADO': '#2e7d32',
                'CANCELADO': '#c62828'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detalhamento por origem
    st.markdown("### 📋 Detalhamento por Origem do Débito")
    
    df_origem = resumo.groupby(['origem_debito', 'desc_status']).agg({
        'qtd': 'sum',
        'valor_total': 'sum'
    }).reset_index()
    
    fig = px.bar(
        df_origem,
        x='origem_debito',
        y='qtd',
        color='desc_status',
        barmode='group',
        color_discrete_map={
            'ATIVO': '#1976d2',
            'QUITADO': '#2e7d32',
            'CANCELADO': '#c62828',
            'PENDENTE 1ª PARCELA': '#ff9800'
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabela resumo
    st.markdown("### 📋 Resumo Detalhado")
    
    df_display = resumo.copy()
    df_display['valor_total'] = df_display['valor_total'].apply(lambda x: formatar_valor_milhoes(x))
    df_display.columns = ['Tipo Adesão', 'Origem', 'Cód Status', 'Status', 
                          'Quantidade', 'Empresas', 'Valor Total', 'Média Parcelas']
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def pagina_machine_learning(dados, filtros_globais):
    """Página de Machine Learning."""
    st.markdown("<h1 class='main-header'>🤖 Machine Learning - Predição de Risco</h1>", unsafe_allow_html=True)

    # Explicação
    with st.expander("ℹ️ Como funciona o Modelo de ML", expanded=False):
        st.markdown("""
        ### Objetivo
        Prever a **probabilidade de cancelamento** de parcelamentos ativos, permitindo ações preventivas.

        ### Modelo Utilizado
        **Gradient Boosting Classifier** - algoritmo de ensemble que combina múltiplas árvores de decisão
        para criar um modelo robusto de classificação.

        ### Features (Variáveis de Entrada)
        | Feature | Descrição | Peso Típico |
        |---------|-----------|-------------|
        | `taxa_sucesso_pct` | Taxa de sucesso histórica | Alto |
        | `qtd_cancelados` | Quantidade de cancelamentos | Alto |
        | `flag_reincidente` | Se tem 3+ cancelamentos | Médio |
        | `valor_medio_parcelamento` | Ticket médio | Médio |
        | `dias_desde_ultimo` | Recência | Médio |
        | `flag_nunca_quitou` | Nunca finalizou com sucesso | Médio |

        ### Métricas de Avaliação
        - **Acurácia:** Percentual de predições corretas
        - **ROC-AUC:** Capacidade de distinguir classes (0.5 = aleatório, 1.0 = perfeito)

        ### Como Interpretar
        - **Probabilidade > 70%:** Risco muito alto de cancelamento
        - **Probabilidade 50-70%:** Risco elevado
        - **Probabilidade < 50%:** Risco moderado a baixo

        ### Uso Recomendado
        1. Execute o modelo para gerar predições atualizadas
        2. Priorize empresas com maior probabilidade
        3. Implemente ações preventivas antes do cancelamento
        """)
    
    engine = get_impala_engine()

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>Clique no botão abaixo para treinar o modelo e gerar predições de risco para todas as empresas.</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Treinar Modelo e Gerar Predições", help="Executa o treinamento do modelo e gera predições atualizadas. Pode levar alguns minutos."):
        with st.spinner("Carregando dados..."):
            df_ml = carregar_dados_ml(engine)

        if df_ml.empty:
            st.error("Dados insuficientes para treinamento.")
            return

        with st.spinner("Treinando modelo..."):
            modelo, scaler, features, metricas = treinar_modelo_risco(df_ml)

        st.success("✅ Modelo treinado com sucesso!")

        # Métricas do modelo
        st.markdown("### 📊 Métricas do Modelo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Acurácia",
                f"{metricas['accuracy']*100:.1f}%",
                help="📊 **Acurácia do Modelo**\n\nPercentual de predições corretas no conjunto de teste.\n\n**Interpretação:**\n- >80%: Excelente\n- 70-80%: Bom\n- <70%: Precisa melhorar"
            )
        with col2:
            st.metric(
                "ROC-AUC",
                f"{metricas['roc_auc']:.3f}",
                help="📈 **ROC-AUC Score**\n\nCapacidade do modelo de distinguir entre classes.\n\n**Interpretação:**\n- 1.0: Perfeito\n- 0.8-0.9: Muito bom\n- 0.7-0.8: Bom\n- 0.5: Aleatório"
            )
        with col3:
            st.metric(
                "Features",
                len(features),
                help="🔢 **Variáveis do Modelo**\n\nQuantidade de features utilizadas para treinar o modelo.\n\n**Nota:** Mais features nem sempre significam melhor modelo."
            )
        
        # Importância das features
        st.markdown("### 📈 Importância das Features")
        
        importance = pd.DataFrame({
            'Feature': list(metricas['feature_importance'].keys()),
            'Importância': list(metricas['feature_importance'].values())
        }).sort_values('Importância', ascending=True)
        
        fig = px.bar(
            importance,
            x='Importância',
            y='Feature',
            orientation='h',
            color='Importância',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Aplicar predições
        st.markdown("### 🎯 Empresas com Maior Risco de Cancelamento")
        
        df_empresas = dados.get('empresas_score', pd.DataFrame())
        
        if not df_empresas.empty:
            with st.spinner("Gerando predições..."):
                df_pred = prever_risco_empresas(modelo, scaler, features, df_empresas)
            
            # Top empresas com maior probabilidade
            df_top = df_pred.nlargest(50, 'prob_cancelamento')
            
            df_display = df_top[['razao_social', 'cnpj', 'prob_cancelamento',
                                 'classificacao_risco', 'valor_ativos',
                                 'qtd_ativos', 'total_parcelamentos', 'gerfe']].copy()
            df_display.columns = ['Razão Social', 'CNPJ', 'Prob. Cancelamento',
                                  'Risco Atual', 'Valor Ativo', 'Ativos',
                                  'Total Parcel.', 'GERFE']
            df_display['CNPJ'] = df_display['CNPJ'].apply(formatar_cnpj_visualizacao)
            df_display['Prob. Cancelamento'] = df_display['Prob. Cancelamento'].apply(lambda x: f"{x*100:.1f}%")
            df_display['Valor Ativo'] = df_display['Valor Ativo'].apply(lambda x: formatar_valor_br(x))
            
            st.dataframe(df_display, use_container_width=True, hide_index=True, height=500)
            
            # Gráfico de distribuição
            st.markdown("### 📊 Distribuição de Probabilidades")
            
            fig = px.histogram(
                df_pred,
                x='prob_cancelamento',
                nbins=50,
                color_discrete_sequence=['#1976d2']
            )
            fig.update_layout(
                xaxis_title="Probabilidade de Cancelamento",
                yaxis_title="Quantidade de Empresas"
            )
            st.plotly_chart(fig, use_container_width=True)


def pagina_expectativa_recebimento(dados, filtros_globais):
    """Página de expectativa de recebimento."""
    st.markdown("<h1 class='main-header'>💰 Expectativa de Recebimento</h1>", unsafe_allow_html=True)

    expectativa = dados.get('expectativa_resumo', pd.DataFrame())
    resumo = dados.get('resumo', pd.DataFrame())

    if expectativa.empty or resumo.empty:
        st.warning("Dados de expectativa não disponíveis.")
        return

    # Caixa de ajuda
    with st.expander("ℹ️ Como interpretar as projeções", expanded=False):
        st.markdown("""
        ### Metodologia de Projeção

        #### Valor Bruto
        Soma das parcelas esperadas no período, assumindo que todos os parcelamentos ativos
        continuarão sendo pagos normalmente.

        #### Valor Ajustado por Risco
        Valor bruto multiplicado pela probabilidade de sucesso baseada na classificação de risco:

        | Classificação | Probabilidade | Explicação |
        |---------------|---------------|------------|
        | **BAIXO** | 85% | Histórico consistente de pagamentos |
        | **MÉDIO** | 65% | Alguns sinais de atenção |
        | **ALTO** | 40% | Risco significativo de cancelamento |
        | **CRÍTICO** | 20% | Alta probabilidade de não recebimento |

        ### Exemplo de Cálculo
        Se uma empresa de risco ALTO tem R$ 100.000 esperados em 3 meses:
        - Valor Bruto: R$ 100.000
        - Valor Ajustado: R$ 100.000 × 40% = R$ 40.000

        ### Uso Recomendado
        - **Planejamento orçamentário:** Use o valor ajustado
        - **Meta otimista:** Use o valor bruto
        - **Análise de gap:** Compare bruto vs ajustado para medir risco
        """)

    # Dica de UX
    st.markdown("""
    <div class='ux-tip'>
        <span>O valor ajustado por risco é mais conservador e realista para planejamento. A diferença entre bruto e ajustado indica o valor em risco.</span>
    </div>
    """, unsafe_allow_html=True)

    r = resumo.iloc[0]

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "3 Meses (Bruto)",
            formatar_valor_milhoes(r.get('expectativa_3_meses', 0)),
            help="💵 **Expectativa Bruta - 3 Meses**\n\nSoma das parcelas esperadas nos próximos 3 meses.\n\n**Premissa:** Todos os parcelamentos ativos continuam sendo pagos.\n\n**Uso:** Cenário otimista."
        )
    with col2:
        bruto_3m = r.get('expectativa_3_meses', 0) or 0
        ajust_3m = r.get('expectativa_ajustada_3_meses', 0) or 0
        diff_3m = bruto_3m - ajust_3m
        st.metric(
            "3 Meses (Ajustado)",
            formatar_valor_milhoes(ajust_3m),
            delta=f"-{formatar_valor_milhoes(diff_3m)} em risco" if diff_3m > 0 else None,
            delta_color="inverse",
            help="💰 **Expectativa Ajustada - 3 Meses**\n\nValor esperado considerando a probabilidade de sucesso por classificação de risco.\n\n**Fórmula:** Bruto × Probabilidade de Sucesso\n\n**Uso:** Planejamento conservador."
        )
    with col3:
        st.metric(
            "12 Meses (Bruto)",
            formatar_valor_milhoes(r.get('expectativa_12_meses', 0)),
            help="💵 **Expectativa Bruta - 12 Meses**\n\nSoma das parcelas esperadas no próximo ano.\n\n**Premissa:** Todos os parcelamentos ativos continuam sendo pagos.\n\n**Uso:** Projeção anual otimista."
        )
    with col4:
        bruto_12m = r.get('expectativa_12_meses', 0) or 0
        ajust_12m = r.get('expectativa_ajustada_12_meses', 0) or 0
        diff_12m = bruto_12m - ajust_12m
        st.metric(
            "12 Meses (Ajustado)",
            formatar_valor_milhoes(ajust_12m),
            delta=f"-{formatar_valor_milhoes(diff_12m)} em risco" if diff_12m > 0 else None,
            delta_color="inverse",
            help="💰 **Expectativa Ajustada - 12 Meses**\n\nValor esperado considerando a probabilidade de sucesso por classificação de risco.\n\n**Fórmula:** Bruto × Probabilidade de Sucesso\n\n**Uso:** Planejamento anual conservador."
        )
    
    st.markdown("---")
    
    # Por classificação de risco
    st.markdown("### 📊 Expectativa por Classificação de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            expectativa,
            x='risco_empresa',
            y=['total_3_meses', 'ajustado_3_meses'],
            barmode='group',
            title='Próximos 3 Meses',
            labels={'value': 'Valor (R$)', 'variable': 'Tipo'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            expectativa,
            x='risco_empresa',
            y=['total_12_meses', 'ajustado_12_meses'],
            barmode='group',
            title='Próximos 12 Meses',
            labels={'value': 'Valor (R$)', 'variable': 'Tipo'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Probabilidades
    st.markdown("### 📋 Probabilidades de Sucesso por Risco")
    
    st.markdown("""
    | Classificação | Probabilidade de Sucesso |
    |---------------|--------------------------|
    | BAIXO | 85% |
    | MÉDIO | 65% |
    | ALTO | 40% |
    | CRÍTICO | 20% |
    """)
    
    st.info("""
    💡 **Nota:** O valor ajustado é calculado multiplicando a expectativa bruta 
    pela probabilidade de sucesso da classificação de risco da empresa.
    """)


def pagina_metodologia(dados, filtros_globais):
    """Página de metodologia."""
    st.markdown("<h1 class='main-header'>📚 Metodologia</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Sobre o Projeto PARCEL
    
    O **Projeto PARCEL** é um sistema de análise e monitoramento de parcelamentos 
    desenvolvido pela Secretaria de Estado da Fazenda de Santa Catarina (SEF/SC).
    
    ---
    
    ## 🗄️ Fontes de Dados
    
    | Tabela | Descrição |
    |--------|-----------|
    | `gecob.parcel_base` | Base unificada de parcelamentos |
    | `gecob.parcel_empresas` | Perfil de empresas com parcelamentos |
    | `gecob.parcel_empresas_score` | Score de risco das empresas |
    | `gecob.parcel_expectativa_recebimento` | Projeção de recebimentos |
    | `gecob.parcel_serie_temporal` | Evolução temporal |
    | `gecob.parcel_metricas_gerfe` | Métricas por regional |
    | `gecob.parcel_metricas_ges` | Métricas por setor |
    | `gecob.parcel_recuperamais` | Análise RECUPERA+ |
    | `gecob.parcel_alertas` | Alertas de risco |
    
    ---
    
    ## 📈 Score de Risco
    
    O score de risco (0-100) é calculado com base em 5 componentes:
    
    | Componente | Peso | Descrição |
    |------------|------|-----------|
    | Histórico de Cancelamentos | 30% | Taxa de cancelamento histórica |
    | Recência | 20% | Tempo desde último parcelamento |
    | Valor em Risco | 25% | Valor total de parcelamentos ativos |
    | Reincidência | 15% | Se empresa tem 3+ cancelamentos |
    | Comportamento | 10% | Classificação de comportamento |
    
    ### Classificação:
    - **CRÍTICO** (≥ 80): Requer ação imediata
    - **ALTO** (60-79): Requer atenção
    - **MÉDIO** (40-59): Monitoramento regular
    - **BAIXO** (< 40): Baixo risco
    
    ---
    
    ## 🤖 Machine Learning
    
    O modelo de ML utiliza **Gradient Boosting Classifier** para prever 
    a probabilidade de cancelamento de parcelamentos ativos.
    
    **Features utilizadas:**
    - Histórico de parcelamentos
    - Taxa de sucesso histórica
    - Valor médio de parcelamento
    - Tempo desde último parcelamento
    - Flags de reincidência
    
    ---
    
    ## 📊 Status dos Parcelamentos
    
    | Código | Status | Descrição |
    |--------|--------|-----------|
    | 1 | ATIVO | Em andamento |
    | 2 | CANCELADO | Cancelado (geralmente por inadimplência) |
    | 3 | SUSPENSO | Temporariamente suspenso |
    | 4 | PENDENTE AUTORIZAÇÃO | Aguardando aprovação |
    | 5 | QUITADO | Finalizado com sucesso |
    | 6 | PENDENTE 1ª PARCELA | Aguardando primeiro pagamento |
    | 7 | EXCLUÍDO | Excluído com pagamentos liberados |
    | 8 | SALDO TRANSFERIDO | Saldo transferido |
    
    ---
    
    ## 📞 Contato
    
    **Equipe de Inteligência Fiscal**  
    Receita Estadual de Santa Catarina  
    SEF/SC
    """)


# =============================================================================
# 7. APLICAÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do dashboard."""
    
    # Carregar dados
    engine = get_impala_engine()
    
    if engine is None:
        st.error("❌ Não foi possível conectar ao banco de dados.")
        st.info("Verifique as credenciais e a conexão de rede.")
        st.stop()
    
    # Carregar dados agregados (rápido)
    with st.spinner("Carregando dados..."):
        dados = carregar_dados_agregados(engine)
    
    if not dados:
        st.error("❌ Não foi possível carregar os dados.")
        st.stop()
    
    # =========================================================================
    # SIDEBAR - NAVEGAÇÃO E FILTROS
    # =========================================================================
    
    st.sidebar.markdown("## 📊 PARCEL")
    st.sidebar.markdown("*Monitoramento de Parcelamentos*")
    st.sidebar.markdown("---")
    
    # Menu de navegação
    paginas = {
        "🎯 Dashboard Executivo": pagina_dashboard_executivo,
        "📅 Análise Temporal": pagina_analise_temporal,
        "🗺️ Análise Regional (GERFE)": pagina_analise_regional,
        "🏭 Análise Setorial (GES)": pagina_analise_setorial,
        "🎯 Empresas por Risco": pagina_empresas_risco,
        "🔍 Drill-Down Empresa": pagina_drilldown_empresa,
        "🚨 Central de Alertas": pagina_alertas,
        "🔄 RECUPERA+": pagina_recuperamais,
        "💰 Expectativa de Recebimento": pagina_expectativa_recebimento,
        "🤖 Machine Learning": pagina_machine_learning,
        "📚 Metodologia": pagina_metodologia
    }
    
    pagina_selecionada = st.sidebar.radio(
        "📑 Navegação:",
        list(paginas.keys()),
        label_visibility="visible"
    )
    
    # Info resumida
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Resumo Rápido")
    
    resumo = dados.get('resumo', pd.DataFrame())
    if not resumo.empty:
        r = resumo.iloc[0]
        st.sidebar.metric("Parcelamentos Ativos", f"{int(r['qtd_ativos']):,}")
        st.sidebar.metric("Taxa de Sucesso", f"{r['taxa_sucesso_global_pct']:.1f}%")
        st.sidebar.metric("Alertas", f"{int(r['total_alertas']):,}")
    
    # Configurações visuais
    with st.sidebar.expander("🎨 Configurações", expanded=False):
        tema = st.selectbox(
            "Tema dos Gráficos",
            ["plotly", "plotly_white", "plotly_dark"],
            index=1,
            key='tema_graficos_sidebar'
        )
        st.session_state['tema_graficos'] = tema
    
    filtros_globais = {'tema': st.session_state.get('tema_graficos', 'plotly_white')}
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Receita Estadual SC - SEF")
    st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # Renderizar página selecionada
    paginas[pagina_selecionada](dados, filtros_globais)


if __name__ == "__main__":
    main()