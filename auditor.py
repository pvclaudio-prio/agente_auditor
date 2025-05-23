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
        # RENOMEAÇÃO INICIAL
        # ===================

        df.rename(columns={
            'Empresa': 'empresa',
            'Conta do Razão': 'conta_contabil',
            'Denom.longa cta.rz.': 'descricao_conta',
            'Txt.it.partida': 'descricao_documento',
            'Moeda da empresa': 'moeda',
            'Nome de fornecedor': 'fornecedor',
            'Documento de compras': 'numero_po'
        }, inplace=True)

        # ===================
        # PRÉ-PROCESSAMENTO
        # ===================

        # Conversão de datas
        df['data_lancamento'] = pd.to_datetime(df['Data de lançamento'], errors='coerce')

        # Colunas auxiliares
        df['ano_mes'] = df['data_lancamento'].dt.to_period('M').astype(str)
        df['valor'] = df['Mont.moeda empresa'].abs()

        # Padronização de fornecedor
        df['fornecedor'] = df['fornecedor'].astype(str).str.strip().str.upper()

        # Tratar PO (nulo ou não)
        df['numero_po'] = df['numero_po'].astype(str).replace('nan', np.nan)

        # ===================
        # LIMPEZA DE DADOS
        # ===================

        # Remover linhas cuja conta contábil contém 'ADIANTAMENTO'
        df = df[~df['descricao_conta'].str.contains('ADIANTAMENTO', na=False, case=False)]

        # Lista de fornecedores a excluir
        fornecedores_excluir = [
            "15 OFICIO DE NOTAS DA COMARCA", "2 OFICIO DO REGISTRO DE PROTESTO", 
            "7 MINDS TRADUCOES CONSULTORIA EMPRESARIAL LTDA", "ABRASCA - ASS. BRAS. DAS CIAS ABERT",
            "ACE SEGURADORA SA", "ADMINISTRATION DES CONTRIBUTIONS DIRECTES",
            "ASA ASSESSORIA DE COMERCIO EXTERIOR LTDA", "ASSOCIACAO BRASILEIRA DOS PRODUTORES INDEPENDENTES DE",
            "ASSOCIACAO DE COMERCIO EXTERIOR DO BRASIL AEB", "ASSOCIACAO DOS REGISTRADORES DE TITULOS E",
            "AUSTRAL SEGURADORA S.A.", "B3 S.A. - BRASIL, BOLSA, BALCAO",
            "B3 SA BRASIL BOLSA BALCAO", "BANCO CENTRAL DO BRASIL", "BANCO DAYCOVAL S.A.",
            "BANCO ITAUCARD S.A.", "BTG PACTUAL INVESTMENT BANKING LTDA", "CAIXA ECONOMICA FEDERAL",
            "CENTRE COMMUN DE LA SECURITE SOCIALE", "CENTRO DE INTEGRACAO EMPRESA ESCOLA E RIO DE JANEIRO",
            "CITIGROUP GOLBAL MARKETS LIMITED", "FLASH TECNOLOGIA E PAGAMENTOS LTDA",
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
            "PETROLANE SERVICOS EM PETROLEO LTDA", "PETRÓLEO BRASILEIRO S.A.",
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
        # SELEÇÃO DE COLUNAS
        # ===================

        df = df[[
            'empresa', 'conta_contabil', 'descricao_conta', 'descricao_documento',
            'moeda', 'fornecedor', 'ano_mes', 'valor', 'numero_po'
        ]]

        # ===================
        # VISUALIZAÇÃO FINAL
        # ===================

        st.success("🚀 Dados tratados e prontos para análise!")

        st.subheader("🔧 Dados Tratados")
        st.dataframe(df.head(20))

        # Salvar dataframe tratado no session_state para as próximas páginas
        st.session_state['df_tratado'] = df

    else:
        st.warning("⚠️ Faça o upload do arquivo Excel para prosseguir.")
