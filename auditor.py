import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import openai
import time

# =========================
# CONFIGURAÇÕES INICIAIS
# =========================

st.set_page_config(
    page_title="Análise de Pagamentos - PRIO",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Análise de Pagamentos a Fornecedores")

# =========================
# MENU LATERAL
# =========================

st.sidebar.image("PRIO_SEM_POLVO_PRIO_PANTONE_LOGOTIPO_Azul.png")

menu = st.sidebar.selectbox(
    "Navegação",
    ["📥 Upload de Base", "🔍 Análise Exploratória", "🚩 Red Flags & Duplicidades", "🤖 Machine Learning | Red Flags", "🧠 IA | Revisão dos Red Flags", "📊 Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Claudio - PRIO 🏴‍☠️")

# =========================
# UPLOAD E TRATAMENTO
# =========================

if menu == "📥 Upload de Base":
    st.subheader("📥 Upload da Base de Dados - SAP")

    file = st.file_uploader("Suba a base de pagamentos", type=["xlsx"])

    if file is not None:
        df = pd.read_excel(file, sheet_name="Exportação SAPUI5")
        st.success("✅ Base carregada com sucesso!")

        # ===================
        # RENOMEAÇÃO DE COLUNAS
        # ===================

        df.rename(columns={
            'Empresa': 'empresa',
            'Conta do Razão': 'conta_contabil',
            'Denom.longa cta.rz.': 'descricao_conta',
            'Txt.it.partida indv.': 'descricao_documento',
            'Moeda da empresa': 'moeda',
            'Nome de fornecedor': 'fornecedor',
            'Documento de compras': 'numero_po',
            'Lançamento contábil': 'lancamento'
        }, inplace=True)

        # ===================
        # PRÉ-PROCESSAMENTO
        # ===================

        df['data_lancamento'] = pd.to_datetime(df['Data de lançamento'], errors='coerce')

        df['ano_mes'] = df['data_lancamento'].dt.to_period('M').astype(str)
        df['valor'] = df['Mont.moeda empresa'].abs().round(0).astype(int)

        df['fornecedor'] = df['fornecedor'].astype(str).str.strip().str.upper()

        df['numero_po'] = df['numero_po'].astype(str).replace('nan', np.nan)

        # ===================
        # LIMPEZA DE DADOS
        # ===================

        df = df[~df['descricao_conta'].str.contains('ADIANTAMENTO', na=False, case=False)]

        fornecedores_excluir = [
    "15 OFICIO DE NOTAS DA COMARCA", "2 OFICIO DO REGISTRO DE PROTESTO", 
    "7 MINDS TRADUCOES CONSULTORIA EMPRESARIAL LTDA", "ABRASCA - ASS. BRAS. DAS CIAS ABERT",
    "ACE SEGURADORA SA", "ADMINISTRATION DES CONTRIBUTIONS DIRECTES",
    "ASA ASSESSORIA DE COMERCIO EXTERIOR LTDA", "ASSOCIACAO BRASILEIRA DOS PRODUTORES INDEPENDENTES DE",
    "ASSOCIACAO DE COMERCIO EXTERIOR DO BRASIL AEB", "ASSOCIACAO DOS REGISTRADORES DE TITULOS E",
    "AUSTRAL SEGURADORA S.A.", "B3 S.A. - BRASIL, BOLSA, BALCAO",
    "B3 SA BRASIL BOLSA BALCAO", "BANCO CENTRAL DO BRASIL", "BANCO DAYCOVAL S.A.",
    "BANCO ITAUCARD S.A.", "BTG PACTUAL INVESTMENT BANKING LTDA","BRADESCO SAUDE S/A", "CAIXA ECONOMICA FEDERAL",
    "CENTRE COMMUN DE LA SECURITE SOCIALE", "CENTRO DE INTEGRACAO EMPRESA ESCOLA E RIO DE JANEIRO",
    "CITIGROUP GOLBAL MARKETS LIMITED","EZZE SEGUROS S.A.", "FLASH TECNOLOGIA E PAGAMENTOS LTDA",
    "GOOGLE BRASIL INTERNET LTDA.", "INSTITUTO BRASILEIRO DE PETROLEO, GAS E",
    "INSTITUTO BRASILEIRO DO MEIO AMBIENTE E", "ITAU BBA INTERNATIONAL PLC", 
    "ITAU CORRETORA DE VALORES S/A", "J S ASSESSORIA ADUANEIRA LTDA", 
    "JUNTA COMERCIAL DO ESTADO DO RIO DE JANEIRO", "MINISTERIO DA ECONOMIA",
    "MINISTERIO DA FAZENDA", "MINISTERIO DA PREVIDENCIA SOCIAL",
    "MORGAN STANLEY & CO INTERNATIONAL P", "MS LOGISTICA INTERNACIONAL LTDA",
    "MUNICIPIO DA SERRA PREFEITURA MUNICIPAL DA SERRA", "MUNICIPIO DE ITAGUAI",
    "MUNICIPIO DE RIO DE JANEIRO", "MUNICIPIO DE SALVADOR", "MUNICIPIO DE SAO GONCALO",
    "MUNICIPIO DE SAO JOAO DA BARRA", "ODONTOPREV S.A.", "PETRO RIO JAGUAR PETROLEO LTDA",
    "PETRO RIO JAGUAR PETRÓLEO LTDA", "PETRO RIO JAGUAR PETRÓLEO S.A", "PETRO RIO SA",
    "PETROLANE SERVICOS EM PETROLEO LTDA", "PETRÓLEO BRASILEIRO S.A.","PRIO BRAVO LTDA.","Prio Bravo Ltda",
    "PREFEITURA DA CIDADE DO RIO DE JANE", "PREFEITURA MUNIC. DE SÃO JOÃO DA BA",
    "PREFEITURA MUNICIPAL DE VILA VELHA", "PRIO BRAVO LTDA.", "PRIO COMERCIALIZADORA LTDA",
    "PRIO FORTE S.A", "PRIO LUXEMBOURG HOLDING S.À R.L.", "PRIO O&G TRADING& SHIPPING GMBH",
    "PRIO TIGRIS LTDA MATRIZ", "PRUDENTIAL DO BRASIL VIDA EM GRUPO", 
    "SECRET DE EST DE FAZENDA - RJ", "SECRETARIA DE ESTADO DE FAZENDA - S", 
    "SECRETARIA DO TESOURO NACIONAL", "TRIBUNAL DE JUSTICA DO ESTADO DO", 
    "TRIBUNAL REGIONAL DO TRABALHO DA 1 REGIA"
]

        fornecedores_excluir = [f.strip().upper() for f in fornecedores_excluir]
        df = df[~df['fornecedor'].isin(fornecedores_excluir)]

        # ===================
        # SELEÇÃO FINAL DE COLUNAS
        # ===================

        df = df[[
            'empresa', 'conta_contabil', 'descricao_conta', 'descricao_documento',
            'moeda', 'fornecedor', 'ano_mes', 'valor', 'numero_po', 'lancamento'
        ]]

        st.success("🚀 Dados tratados e prontos para análise!")

        st.subheader("🔧 Dados Tratados")
        st.dataframe(df.head(20))

        st.session_state['df_tratado'] = df

    else:
        st.warning("⚠️ Faça o upload do arquivo Excel para prosseguir.")

# =========================
# ANÁLISE EXPLORATÓRIA
# =========================

elif menu == "🔍 Análise Exploratória":
    st.subheader("🔍 Análise Exploratória da Base de Pagamentos")

    if 'df_tratado' in st.session_state:
        df = st.session_state['df_tratado'].copy()

        st.markdown("#### 🔎 Filtros")

        ano_mes = st.multiselect(
            "Filtrar por Ano-Mês:",
            sorted(df['ano_mes'].unique()),
            default=sorted(df['ano_mes'].unique())
        )

        conta_contabil = st.multiselect(
            "Filtrar por Conta Contábil:",
            sorted(df['conta_contabil'].unique()),
            default=sorted(df['conta_contabil'].unique())
        )

        fornecedor = st.multiselect(
            "Filtrar por Fornecedor:",
            sorted(df['fornecedor'].unique())
        )

        df_filtro = df[
            (df['ano_mes'].isin(ano_mes)) &
            (df['conta_contabil'].isin(conta_contabil))
        ]
        if fornecedor:
            df_filtro = df_filtro[df_filtro['fornecedor'].isin(fornecedor)]

        st.markdown("---")

        total_pago = df_filtro['valor'].sum()
        qtd_lancamentos = df_filtro.shape[0]
        ticket_medio = total_pago / qtd_lancamentos if qtd_lancamentos > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Pago", f"R$ {total_pago:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        col2.metric("🧾 Lançamentos", f"{qtd_lancamentos:,}")
        col3.metric("💸 Ticket Médio", f"R$ {ticket_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        st.markdown("---")

        st.markdown("### 📈 Evolução Temporal dos Pagamentos")
        evolucao = df_filtro.groupby('ano_mes')['valor'].sum().reset_index()
        st.bar_chart(evolucao.set_index('ano_mes'))

        top_n = st.slider(
            "Selecione quantos fornecedores deseja visualizar no gráfico:",
            min_value=1, max_value=20, value=5
        )

        st.markdown(f"### 🏢 Top {top_n} Fornecedores por Valor")
        top_fornecedores = (
            df_filtro.groupby('fornecedor')['valor']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        st.bar_chart(top_fornecedores.set_index('fornecedor'))

        st.markdown("---")
        st.markdown("### 🔍 Tabela Detalhada dos Dados Filtrados")
        st.dataframe(df_filtro)

    else:
        st.warning("⚠️ Você precisa primeiro carregar e tratar a base na aba '📥 Upload de Base'.")
        
elif menu == "🚩 Red Flags & Duplicidades":

    if 'df_tratado' in st.session_state:
        df = st.session_state['df_tratado'].copy()

        st.markdown("#### ⚙️ Critérios utilizados para duplicidade:")
        st.markdown("""
        - Mesmo **Fornecedor**
        - Mesmo **Valor (sem centavos)**
        - Mesmo **Ano-Mês**
        - Se informado, mesmo **PO (número_po)**
        """)

        # ===================
        # GERAÇÃO DA CHAVE DE DUPLICIDADE
        # ===================

        def gerar_chave(row):
            if pd.notnull(row['numero_po']):
                return f"{row['fornecedor']}|{row['valor']}|{row['ano_mes']}|{row['numero_po']}"
            else:
                return f"{row['fornecedor']}|{row['valor']}|{row['ano_mes']}"

        df['chave_duplicidade'] = df.apply(gerar_chave, axis=1)

        # ===================
        # MARCAR DUPLICIDADES
        # ===================
        df['qtd_ocorrencias'] = df.groupby('chave_duplicidade')['chave_duplicidade'].transform('count')

        df_duplicados = df[df['qtd_ocorrencias'] > 1].copy()

        st.markdown("### 🚩 Pagamentos com Possíveis Duplicidades")
        st.write(f"🔎 Foram encontrados **{df_duplicados['chave_duplicidade'].nunique()} grupos** de possíveis duplicidades, "
                 f"totalizando **{df_duplicados.shape[0]} lançamentos**.")

        st.dataframe(df_duplicados)

        # ===================
        # DOWNLOAD DOS RESULTADOS
        # ===================
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8-sig')

        csv = convert_df_to_csv(df_duplicados)

        st.download_button(
            label="📥 Baixar Duplicidades em CSV",
            data=csv,
            file_name='duplicidades_detectadas.csv',
            mime='text/csv'
        )

    else:
        st.warning("⚠️ Você precisa primeiro carregar e tratar a base na aba '📥 Upload de Base'.")

elif menu == "🤖 Machine Learning | Red Flags":
    st.subheader("🤖 Machine Learning Supervisionado | Classificação de Risco com Balanceamento (SMOTE)")

    # =========================
    # CARREGAR BASE
    # =========================
    if 'df_tratado' in st.session_state:
        df = st.session_state['df_tratado'].copy()
    else:
        st.warning("⚠️ Você precisa executar primeiro a aba '📥 Upload de Base'.")
        st.stop()

    st.markdown("Este modelo aplica Random Forest com técnica de balanceamento SMOTE para lidar com desbalanceamento na detecção de Red Flags.")

    # =========================
    # GERAR RED FLAG AUTOMÁTICA
    # =========================

    if 'red_flag' not in df.columns:
        st.warning("⚠️ A coluna 'red_flag' não foi encontrada. Aplicando regra automática: valores acima de R$ 100.000 são considerados Red Flag.")
        df['red_flag'] = df['valor'].apply(lambda x: 'Sim' if x > 100000 else 'Não')

    # =========================
    # ENGENHARIA DE FEATURES
    # =========================

    qtd_pagamentos = df.groupby('fornecedor').size().to_dict()
    valor_medio = df.groupby('fornecedor')['valor'].mean().to_dict()

    df['qtd_pagamentos_fornecedor'] = df['fornecedor'].map(qtd_pagamentos)
    df['valor_medio_fornecedor'] = df['fornecedor'].map(valor_medio)

    df['centro_custo_nome'] = (
        df['descricao_documento']
        .str.extract(r'\((.*?)\)')[0]
        .fillna('Não Informado')
    )

    le_fornecedor = LabelEncoder()
    le_conta = LabelEncoder()
    le_centro = LabelEncoder()

    df['fornecedor_encoded'] = le_fornecedor.fit_transform(df['fornecedor'])
    df['conta_encoded'] = le_conta.fit_transform(df['conta_contabil'])
    df['centro_encoded'] = le_centro.fit_transform(df['centro_custo_nome'])

    df['ano_mes_ordinal'] = df['ano_mes'].astype('category').cat.codes

    # =========================
    # DEFINIÇÃO DO TARGET
    # =========================

    df['label'] = df['red_flag'].map({'Sim': 1, 'Não': 0})

    if df['label'].nunique() < 2:
        st.error("❌ O dataset possui apenas uma classe na variável alvo, mesmo após aplicar a regra automática. Ajuste seus dados ou a regra.")
        st.stop()

    # =========================
    # NORMALIZAÇÃO E SMOTE
    # =========================

    X = df[[
        'valor', 'qtd_pagamentos_fornecedor', 'valor_medio_fornecedor',
        'fornecedor_encoded', 'conta_encoded', 'centro_encoded',
        'ano_mes_ordinal'
    ]]

    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    # =========================
    # TREINAMENTO DO MODELO
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # =========================
    # AVALIAÇÃO DO MODELO
    # =========================

    y_pred = clf.predict(X_test)

    st.subheader("📊 Avaliação do Modelo (Após Balanceamento)")
    st.text(classification_report(y_test, y_pred))

    # =========================
    # APLICAÇÃO NA BASE COMPLETA
    # =========================

    df['probabilidade_redflag'] = clf.predict_proba(X_scaled)[:, 1]

    threshold = st.slider("Selecione o Threshold para Classificação", min_value=0.1, max_value=0.9, value=0.7)
    df['red_flag'] = df['probabilidade_redflag'].apply(lambda x: 'Sim' if x >= threshold else 'Não')

    st.subheader("🚦 Resultado do Modelo de Classificação")
    st.dataframe(df[[
        'fornecedor', 'valor', 'ano_mes', 'probabilidade_redflag', 'red_flag'
    ]])

    st.session_state['df_ml'] = df

    # =========================
    # DOWNLOAD DO RESULTADO
    # =========================

    csv = df.to_csv(index=False).encode('utf-8-sig')

    st.download_button(
        label="📥 Baixar Resultado da Classificação em CSV",
        data=csv,
        file_name='resultado_classificacao.csv',
        mime='text/csv'
    )
    
elif menu == "🧠 IA | Revisão dos Red Flags":
    client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

    st.subheader("🧠 Agente de IA | Revisão dos Red Flags com GPT-4o")

    if 'df_ml' in st.session_state:
        df = st.session_state['df_ml'].copy()

        st.markdown("O agente de IA revisa os pagamentos sinalizados pelo modelo de Machine Learning e fornece uma segunda opinião com justificativas precisas.")

        # =========================
        # APLICAR FILTROS
        # =========================

        st.markdown("### 🔎 Filtros")

        ano_mes = st.multiselect(
            "Filtrar por Ano-Mês:",
            sorted(df['ano_mes'].unique()),
            default=sorted(df['ano_mes'].unique())
        )

        red_flag = st.multiselect(
            "Filtrar por Red Flag do modelo de ML:",
            ["Sim", "Não"],
            default=["Sim", "Não"]
        )

        df_filtrado = df[
            (df['ano_mes'].isin(ano_mes)) &
            (df['red_flag'].isin(red_flag))
        ]

        st.dataframe(df_filtrado)

        # =========================
        # ESTIMATIVA DE TOKENS E CUSTO
        # =========================

        n_linhas = df_filtrado.shape[0]
        tokens_estimados = n_linhas * 700  # Aproximadamente 700 tokens por linha
        custo_estimado = tokens_estimados * 0.00001  # GPT-4o ~ $0.01 por 1k tokens (ajustável)

        st.markdown(f"**🔢 Tokens estimados:** {tokens_estimados:,}")
        st.markdown(f"**💲 Custo estimado:** ~ USD {custo_estimado:.4f}")

        # =========================
        # BOTÃO PARA EXECUTAR A ANÁLISE
        # =========================

        rodar_analise = st.button("🚀 Rodar Análise com IA")

        if rodar_analise:
            with st.spinner('🧠 Executando análise com IA, isso pode levar alguns minutos...'):
                df_filtrado['revisao_ia'] = ''
                df_filtrado['motivo_revisao'] = ''

                for idx, row in df_filtrado.iterrows():
                    prompt = f"""
Você é um auditor especializado em detecção de fraudes. Analise o seguinte pagamento:

- Fornecedor: {row['fornecedor']}
- Valor: R$ {row['valor']}
- Conta contábil: {row['conta_contabil']} - {row['descricao_conta']}
- Descrição do documento: {row['descricao_documento']}
- Mês de referência: {row['ano_mes']}
- Flag de ML: {row['red_flag']}

Pergunta:
O modelo de ML sinalizou como '{row['red_flag']}'. Você concorda? Responda 'Sim' ou 'Não' e explique o motivo de forma objetiva e precisa.
"""

                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=500
                        )

                        resposta = response.choices[0].message.content.strip()

                        if resposta.lower().startswith('sim'):
                            df_filtrado.at[idx, 'revisao_ia'] = 'Sim'
                        elif resposta.lower().startswith('não') or resposta.lower().startswith('nao'):
                            df_filtrado.at[idx, 'revisao_ia'] = 'Não'
                        else:
                            df_filtrado.at[idx, 'revisao_ia'] = 'Não Informado'

                        if ':' in resposta:
                            motivo = resposta.split(':', 1)[1].strip()
                        else:
                            motivo = resposta.strip()

                        df_filtrado.at[idx, 'motivo_revisao'] = motivo

                        time.sleep(0.5)  # Pequeno delay para não sobrecarregar API

                    except Exception as e:
                        st.error(f"Erro na chamada da API: {e}")
                        df_filtrado.at[idx, 'revisao_ia'] = 'Erro'
                        df_filtrado.at[idx, 'motivo_revisao'] = 'Erro na API'

            st.success("🚀 Revisão concluída!")

            st.markdown("### 📜 Resultado da Revisão pela IA")
            st.dataframe(df_filtrado)

            st.session_state['df_revisado'] = df_filtrado

            # =========================
            # DOWNLOAD DO RESULTADO
            # =========================

            csv = df_filtrado.to_csv(index=False).encode('utf-8-sig')

            st.download_button(
                label="📥 Baixar Resultado da IA em CSV",
                data=csv,
                file_name='revisao_ia_pagamentos.csv',
                mime='text/csv'
            )

    else:
        st.warning("⚠️ Você precisa rodar antes a aba '🤖 Machine Learning | Red Flags'.")

elif menu == "📊 Dashboard":
    st.subheader("📊 Dashboard Consolidado")

    if 'df_revisado' in st.session_state:
        df = st.session_state['df_revisado'].copy()

        st.markdown("### 🔎 Filtros")

        ano_mes = st.multiselect(
            "Filtrar por Ano-Mês:",
            sorted(df['ano_mes'].unique()),
            default=sorted(df['ano_mes'].unique())
        )

        fornecedor = st.multiselect(
            "Filtrar por Fornecedor:",
            sorted(df['fornecedor'].unique())
        )

        # Aplicação dos filtros
        df_filtro = df[df['ano_mes'].isin(ano_mes)]
        if fornecedor:
            df_filtro = df_filtro[df_filtro['fornecedor'].isin(fornecedor)]

        # ========================
        # INDICADORES PRINCIPAIS
        # ========================
        total_pago = df_filtro['valor'].sum()
        qtd_lancamentos = df_filtro.shape[0]
        qtd_redflag_ml = df_filtro[df_filtro['red_flag'] == 'Sim'].shape[0]
        qtd_redflag_ia = df_filtro[df_filtro['revisao_ia'] == 'Sim'].shape[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Total Pago", f"R$ {total_pago:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        col2.metric("🧾 Lançamentos", f"{qtd_lancamentos:,}")
        col3.metric("🚩 Red Flags ML", f"{qtd_redflag_ml}")
        col4.metric("🤖 Red Flags IA", f"{qtd_redflag_ia}")

        st.markdown("---")

        # ========================
        # GRÁFICO — COMPARATIVO TOTAL x ML x IA
        # ========================
        
        st.markdown("### 📊 Comparativo: Total vs Red Flags ML vs Red Flags IA")
        
        total_lancamentos = df_filtro.shape[0]
        qtd_redflag_ml = df_filtro[df_filtro['red_flag'] == 'Sim'].shape[0]
        qtd_redflag_ia = df_filtro[df_filtro['revisao_ia'] == 'Sim'].shape[0]
        
        # Montar dataframe para o gráfico
        df_comparativo = pd.DataFrame({
            'Categoria': ['Total de Lançamentos', 'Red Flags ML', 'Red Flags IA'],
            'Quantidade': [total_lancamentos, qtd_redflag_ml, qtd_redflag_ia]
        })
        
        st.bar_chart(
            data=df_comparativo.set_index('Categoria'),
            use_container_width=True
        )

        # ========================
        # LISTA DE ALERTAS CRÍTICOS (ML e IA = Sim)
        # ========================
        st.markdown("### 🔥 Alertas Críticos (ML e IA Concordam)")

        df_alertas = df_filtro[
            (df_filtro['red_flag'] == 'Sim') &
            (df_filtro['revisao_ia'] == 'Sim')
        ]

        st.dataframe(df_alertas)

        csv = df_alertas.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            label="📥 Baixar Alertas Críticos em CSV",
            data=csv,
            file_name='alertas_criticos.csv',
            mime='text/csv'
        )

    else:
        st.warning("⚠️ Você precisa executar a aba '🧠 IA | Revisão dos Red Flags' antes de visualizar o Dashboard.")

