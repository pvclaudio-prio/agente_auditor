import streamlit as st
import pandas as pd
import numpy as np

# =========================
# CONFIGURAÇÕES INICIAIS
# =========================

st.set_page_config(
    page_title="Análise de Pagamentos - PRIO",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Análise de Pagamentos a Fornecedores")
st.markdown("### Sistema de Detecção de Duplicidades e Red Flags - PRIO")

# =========================
# MENU LATERAL
# =========================

menu = st.sidebar.selectbox(
    "Navegação",
    ["📥 Upload de Base", "🔍 Análise Exploratória", "🚩 Red Flags & Duplicidades", "📊 Dashboard"]
)

st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Claudio - PRIO 🏴‍☠️")

# =========================
# UPLOAD E TRATAMENTO
# =========================

if menu == "📥 Upload de Base":
    st.subheader("📥 Upload da Base de Dados - SAP")

    file = st.file_uploader("Selecione o arquivo Excel extraído do SAPUI5", type=["xlsx"])

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

        fornecedores_excluir = [ ... ]  # (mantém a sua lista completa aqui)

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

        st.markdown("### 🏢 Distribuição por Centro de Custo")

        df_filtro['centro_custo_nome'] = (
            df_filtro['descricao_documento']
            .str.extract(r'\((.*?)\)')[0]
            .fillna('Não Informado')
        )

        dist_centro = (
            df_filtro.groupby('centro_custo_nome')['valor']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        st.bar_chart(dist_centro.set_index('centro_custo_nome'))

        st.markdown("---")
        st.markdown("### 🔍 Tabela Detalhada dos Dados Filtrados")
        st.dataframe(df_filtro)

    else:
        st.warning("⚠️ Você precisa primeiro carregar e tratar a base na aba '📥 Upload de Base'.")
