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

# =========================
# TÍTULO DA PÁGINA
# =========================

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
# UPLOAD E TRATAMENTO DA BASE
# =========================

if menu == "📥 Upload de Base":
    st.subheader("📥 Upload da Base de Dados - SAP")
    
    file = st.file_uploader("Selecione o arquivo Excel extraído do SAPUI5", type=["xlsx"])
    
    if file is not None:
        df = pd.read_excel(file, sheet_name="Exportação SAPUI5")
        st.success("✅ Base carregada com sucesso!")

        # Visualização inicial
        st.subheader("📄 Pré-visualização dos Dados")
        st.dataframe(df.head(20))

        # ===================
        # PRÉ-PROCESSAMENTO
        # ===================

        # Conversão de datas
        df['Data de lançamento'] = pd.to_datetime(df['Data de lançamento'], errors='coerce')

        # Criação de colunas auxiliares
        df['Ano_Mes'] = df['Data de lançamento'].dt.to_period('M').astype(str)
        df['Valor_absoluto'] = df['Mont.moeda empresa'].abs()

        # Padronizar nome do fornecedor
        df['Nome de fornecedor'] = df['Nome de fornecedor'].astype(str).str.strip().str.upper()

        # Tratar PO (Documento de compras)
        df['PO'] = df['Documento de compras'].astype(str).replace('nan', np.nan)

        st.success("🚀 Dados tratados e prontos para análise!")

        st.subheader("🔧 Dados Tratados")
        st.dataframe(df.head(20))

        # Opcional: salvar no session_state para as próximas etapas
        st.session_state['df_tratado'] = df

    else:
        st.warning("⚠️ Faça o upload do arquivo Excel para prosseguir.")

