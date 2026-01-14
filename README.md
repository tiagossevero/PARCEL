# PARCEL - Sistema de Análise e Monitoramento de Parcelamentos

Sistema de dashboard interativo para análise e monitoramento de parcelamentos tributários da Receita Estadual de Santa Catarina (SEF/SC).

## Sobre o Projeto

O PARCEL é uma aplicação web desenvolvida em Streamlit que oferece análises abrangentes dos acordos de parcelamento tributário em Santa Catarina. O sistema permite:

- Monitorar e analisar todos os acordos de parcelamento vigentes
- Acompanhar taxas de adimplência e sucesso dos parcelamentos
- Identificar empresas de alto risco através de algoritmos de pontuação
- Prever probabilidades de cancelamento utilizando Machine Learning
- Projetar expectativas de receita futura dos parcelamentos ativos
- Analisar parcelamentos por diferentes dimensões (regional, setorial, temporal)
- Acompanhar o programa especial RECUPERA+ (Lei nº 18.819/2024)
- Gerar dashboards executivos com KPIs e alertas acionáveis

## Funcionalidades

O dashboard possui **11 páginas principais**:

| Página | Descrição |
|--------|-----------|
| **Dashboard Executivo** | KPIs principais, evolução mensal, distribuição por categoria e alertas de risco |
| **Análise Temporal** | Evolução histórica de quantidade, valor, taxa de sucesso e ticket médio |
| **Análise Regional (GERFE)** | Ranking de sucesso por regional, treemap de valores e empresas de alto risco |
| **Análise Setorial (GES)** | Análise por gestão setorial e setores econômicos (CNAE) |
| **Empresas por Risco** | Classificação de risco das empresas com filtros e detalhamento |
| **Drill-Down Empresa** | Consulta detalhada de empresa específica com histórico completo |
| **Central de Alertas** | Gestão de alertas por prioridade e tipo |
| **RECUPERA+** | Métricas específicas do programa de recuperação fiscal |
| **Expectativa de Recebimento** | Projeções de receita de 3 e 12 meses com ajuste por risco |
| **Machine Learning** | Modelo preditivo de cancelamento com Gradient Boosting |
| **Metodologia** | Documentação técnica e metodológica do sistema |

## Tecnologias Utilizadas

### Framework Principal
- **Python 3.7+**
- **Streamlit** - Framework web para dashboards interativos

### Processamento de Dados
- **Pandas** - Manipulação e análise de dados
- **NumPy** - Computação numérica
- **SQLAlchemy** - ORM e gerenciamento de conexões

### Visualização
- **Plotly Express** - Gráficos interativos de alto nível
- **Plotly Graph Objects** - Gráficos customizados

### Machine Learning
- **Scikit-learn** - Algoritmos de ML
  - Gradient Boosting Classifier
  - Random Forest Classifier
  - StandardScaler, LabelEncoder
  - Métricas de classificação

### Banco de Dados
- **Apache Impala** - Engine SQL para Big Data
- Autenticação LDAP
- Conexão SSL/TLS

## Estrutura do Projeto

```
PARCEL/
├── PARCEL.py                    # Aplicação principal (Streamlit)
├── PARCEL_ Criar tbls.json      # Queries SQL para criação das tabelas
└── README.md                    # Documentação
```

## Tabelas do Banco de Dados

O sistema utiliza 11 tabelas analíticas no schema `gecob`:

| Tabela | Descrição |
|--------|-----------|
| `parcel_base` | Dados unificados de parcelamentos com 60+ campos enriquecidos |
| `parcel_empresas` | Perfis de empresas e agregações de histórico de pagamento |
| `parcel_empresas_score` | Pontuação de risco (0-100) para cada empresa |
| `parcel_expectativa_recebimento` | Projeções de receita futura esperada |
| `parcel_serie_temporal` | Evolução temporal dos parcelamentos |
| `parcel_metricas_gerfe` | Métricas agregadas por regional (GERFE) |
| `parcel_metricas_ges` | Métricas agregadas por gestão setorial (GES) |
| `parcel_recuperamais` | Análise específica do programa RECUPERA+ |
| `parcel_programas_especiais` | Análise de programas especiais (REFIS, etc.) |
| `parcel_alertas` | Alertas de risco para parcelamentos prioritários |
| `parcel_resumo_executivo` | KPIs consolidados para o dashboard |

## Metodologia de Score de Risco

O sistema calcula um score de risco de 0 a 100 (maior = mais arriscado) baseado em 5 componentes:

| Componente | Peso | Descrição |
|------------|------|-----------|
| Histórico de Cancelamentos | 30% | Taxa histórica de cancelamento |
| Recência | 20% | Tempo desde último parcelamento |
| Valor em Risco | 25% | Valor total de parcelamentos ativos |
| Reincidência | 15% | Flag de múltiplos cancelamentos |
| Comportamento | 10% | Classificação comportamental |

### Classificações de Risco

- **CRÍTICO** (Score ≥ 80): Requer ação imediata
- **ALTO** (Score 60-79): Requer atenção
- **MÉDIO** (Score 40-59): Monitoramento regular
- **BAIXO** (Score < 40): Baixo risco

## Machine Learning

### Modelo Implementado
**Algoritmo:** Gradient Boosting Classifier

### Features Utilizadas
- `total_parcelamentos` - Quantidade total de parcelamentos
- `qtd_cancelados` - Quantidade de cancelados
- `qtd_quitados` - Quantidade de quitados
- `taxa_sucesso_pct` - Taxa de sucesso histórica
- `valor_medio_parcelamento` - Valor médio do parcelamento
- `media_parcelas` - Média de parcelas
- `dias_desde_ultimo_parcelamento` - Dias desde último parcelamento
- `flag_reincidente` - Indicador de reincidência
- `flag_nunca_quitou` - Indicador de nunca quitou
- `flag_simples_nacional` - Indicador de Simples Nacional

### Parâmetros do Modelo
- Estimadores: 100
- Profundidade máxima: 5
- Taxa de aprendizado: 0.1
- Divisão de teste: 20%

## Instalação

### Pré-requisitos

- Python 3.7 ou superior
- Acesso à rede SEF/SC
- Credenciais LDAP válidas
- Conexão com `bdaworkernode02.sef.sc.gov.br:21050`

### Instalação de Dependências

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd PARCEL

# Instale as dependências
pip install streamlit pandas numpy plotly sqlalchemy scikit-learn impyla
```

### Configuração de Secrets

Crie o arquivo `.streamlit/secrets.toml`:

```toml
[impala_credentials]
user = "seu_usuario_ldap"
password = "sua_senha_ldap"
```

## Execução

```bash
streamlit run PARCEL.py
```

A aplicação será iniciada em `http://localhost:8501`

## Configuração do Banco de Dados

```python
IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
DATABASE = 'gecob'
```

## Códigos de Status dos Parcelamentos

| Código | Status | Descrição |
|--------|--------|-----------|
| 1 | ATIVO | Em andamento - pagamentos sendo realizados |
| 2 | CANCELADO | Cancelado (geralmente por inadimplência após 90 dias) |
| 3 | SUSPENSO | Temporariamente suspenso |
| 4 | PENDENTE AUTORIZAÇÃO | Aguardando aprovação |
| 5 | QUITADO | Concluído - todos os pagamentos realizados |
| 6 | PENDENTE 1ª PARCELA | Contratado mas primeira parcela não paga |
| 7 | EXCLUÍDO PAGTOS LIBERADOS | Excluído com pagamentos liberados |
| 8 | SALDO TRANSFERIDO | Saldo transferido para outro parcelamento |

## Categorias de Parcelamento

- **ICMS** - ICMS (principal receita)
- **DÍVIDA ATIVA** - Dívida ativa (tributos não pagos)
- **DECLARADO** - Declarado pelo contribuinte
- **RECUPERA+** - Programa especial de recuperação (Lei nº 18.819/2024)
- **IPVA** - Imposto sobre veículos
- **ITCMD** - Imposto sobre herança e doação
- **OUTROS** - Outras categorias

## Programa RECUPERA+

Programa especial de recuperação fiscal com base legal:
- Lei nº 18.819/2024
- Convênio ICMS nº 113/2023

### Abrangência
- Débitos de ICMS com fatos geradores até 31/12/2022
- Período de adesão: Janeiro - Maio de 2024

### Benefícios
- **Pagamento à vista:** até 95% de redução em juros/multas
- **Parcelamento (até 120x):** até 80% de redução

## Fontes de Dados

O sistema consome dados das seguintes tabelas upstream:

- `usr_sat_ctacte.tab_parcelamento_pedido` - Tabela principal de parcelamentos
- `usr_sat_ctacte.tab_cta_cte` - Transações de conta corrente fiscal
- `usr_sat_ods.vw_ods_contrib` - Cadastro de contribuintes
- `usr_sat_ods.vw_dva_divida_ativa` - Registros de dívida ativa
- `usr_sat_ods.vw_ods_pagamento` - Registros de pagamentos

## Organização do Código

O código está organizado nas seguintes seções:

1. **Linhas 1-48:** Imports e configuração inicial
2. **Linhas 51-188:** Estilização CSS para componentes UI
3. **Linhas 191-246:** Funções auxiliares (formatação, cores)
4. **Linhas 249-514:** Conexão com banco e funções de carga de dados
5. **Linhas 517-594:** Funções de Machine Learning
6. **Linhas 597-1858:** Funções de renderização das 11 páginas
7. **Linhas 1861-1941:** Função principal da aplicação

## Cache e Performance

- `@st.cache_resource` - Engine do banco de dados (persistente)
- `@st.cache_data(ttl=3600)` - Carga de dados (cache de 1 hora)

## Contato

**Secretaria de Estado da Fazenda de Santa Catarina (SEF/SC)**
Receita Estadual - Gerência de Cobrança (GECOB)

## Licença

Projeto desenvolvido para uso interno da Receita Estadual de Santa Catarina.

---

*Sistema desenvolvido pela SEF/SC - Receita Estadual de Santa Catarina*
