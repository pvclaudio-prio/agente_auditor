import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------
# Configurações da página
# --------------------------
st.set_page_config(
    page_title="Análise de Pagamentos",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Análise Inteligente de Pagamentos")
st.markdown("Aplicativo para clusterização, detecção de red flags e revisão inteligente com IA.")

# --------------------------
# Definindo abas
# --------------------------
st.sidebar.image("PRIO_SEM_POLVO_PRIO_PANTONE_LOGOTIPO_Azul.png")
aba = st.sidebar.radio(
    "Navegação",
    ["🏗️ Análise ML", "🤖 Agente IA", "📥 Download"]
)

# --------------------------
# Aba 1 - Análise Tradicional
# --------------------------
if aba == "🏗️ Análise ML":
    st.header("🏗️ Clusterização + Classificação + Red Flag")

    uploaded_file = st.file_uploader("📤 Faça upload da base de pagamentos (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("📄 Pré-visualização da base")
        st.dataframe(df.head())

        st.subheader("🎯 Selecione as colunas para análise")
        selected_columns = st.multiselect("Selecione as colunas numéricas para clusterização e classificação", df.columns)

        if selected_columns:
            st.success(f"Colunas selecionadas para análise: {selected_columns}")

            # 🔥 Placeholder para futura clusterização
            st.subheader("📊 Visualização dos Clusters (Em Desenvolvimento)")

            # Placeholder para plot
            st.info("Aqui aparecerá o gráfico de clusterização.")

            # 🔍 Placeholder para Grid Search dos pesos
            st.subheader("🛠️ Grid Search de Pesos das Features (Em Desenvolvimento)")

            st.info("Aqui será exibido um gráfico mostrando a influência de cada feature.")

            # 🏷️ Placeholder para Classificação e Red Flags
            st.subheader("🚩 Classificação e Identificação de Red Flags (Em Desenvolvimento)")

            # Simular uma coluna temporária de Red Flag
            df['Red Flag'] = np.random.choice(['Sim', 'Não'], size=len(df))

            st.dataframe(df.head())

            st.download_button(
                label="💾 Baixar base com Red Flags (Aba 1)",
                data=df.to_csv(index=False).encode('utf-8-sig'),
                file_name="base_red_flag_aba1.csv",
                mime='text/csv'
            )

# --------------------------
# Aba 2 - Agente GPT-4o
# --------------------------
elif aba == "🤖 Agente IA":
    st.header("🤖 Agente de IA - Revisão dos Red Flags")

    st.subheader("📤 Upload dos arquivos necessários")
    file_base_redflag = st.file_uploader("Base com Red Flag da Aba 1", type=["csv"], key="base1")
    file_base_original = st.file_uploader("Base de Pagamentos Original (SAP)", type=["xlsx"], key="base2")

    if file_base_redflag and file_base_original:
        df_redflag = pd.read_csv(file_base_redflag)
        df_original = pd.read_excel(file_base_original)

        st.subheader("📄 Pré-visualização das bases")
        st.write("🔸 Base com Red Flags")
        st.dataframe(df_redflag.head())

        st.write("🔸 Base Original")
        st.dataframe(df_original.head())

        st.subheader("🚀 Execução do Agente GPT-4o")

        if st.button("🔍 Rodar Agente IA"):
            # 🧠 Placeholder do Agente GPT
            st.info("🔧 O agente está analisando os dados... (Simulação)")

            df_final = df_original.copy()

            # Simular geração de Red Flag Revisado e Motivo
            df_final['Red Flag'] = df_redflag.get('Red Flag', np.random.choice(['Sim', 'Não'], size=len(df_final)))
            df_final['Red Flag Revisado'] = np.random.choice(['Sim', 'Não'], size=len(df_final))
            df_final['Motivo'] = np.where(
                df_final['Red Flag Revisado'] == 'Sim',
                'Valor fora do padrão esperado.',
                'Sem inconsistências encontradas.'
            )

            st.success("✅ Análise concluída.")
            st.dataframe(df_final.head())

            st.session_state['df_final'] = df_final  # Salvar para a aba de download

# --------------------------
# Aba 3 - Download Final
# --------------------------
elif aba == "📥 Download":
    st.header("📥 Download da Base Consolidada")

    if 'df_final' in st.session_state:
        st.subheader("🗂️ Base Consolidada com Red Flags e Motivos")
        st.dataframe(st.session_state['df_final'].head())

        st.download_button(
            label="💾 Baixar Base Final",
            data=st.session_state['df_final'].to_csv(index=False).encode('utf-8-sig'),
            file_name="base_pagamentos_final.csv",
            mime='text/csv'
        )
    else:
        st.warning("⚠️ Nenhuma base gerada ainda. Por favor, execute as etapas na Aba 1 e Aba 2.")


