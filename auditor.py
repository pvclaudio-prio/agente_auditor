import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import json
import time

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

# Configurar chave da OpenAI
openai.api_key = st.secrets["openai"]["api_key"]

# --------------------------
# Definindo abas
# --------------------------
st.sidebar.image("PRIO_SEM_POLVO_PRIO_PANTONE_LOGOTIPO_Azul.png")
aba = st.sidebar.radio(
    "📋 Menu",
    ["🏗️ Análise ML", "🤖 Agente IA", "📥 Download"]
)

# --------------------------
# Aba 1 - Análise Tradicional
# --------------------------
if aba == "🏗️ Análise ML":
    st.header("🏗️ Clusterização + Classificação + Red Flag")

    uploaded_file = st.file_uploader("📤 Faça upload da base de pagamentos (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
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
# Aba 2 - Agente GPT-4o com IA Real Robusto
# --------------------------
elif aba == "🤖 Agente IA":
    st.header("🤖 Agente de IA - Revisão dos Red Flags")

    if 'df_redflag' not in st.session_state:
        st.warning("⚠️ A base de Red Flags ainda não foi gerada. Por favor, execute a Aba 1 antes de usar esta aba.")
        st.stop()

    df_base = st.session_state['df_redflag'].copy()

    st.subheader("📄 Base com Red Flags (Aba 1)")
    st.dataframe(df_base.head())

    # 🔍 Filtros
    st.subheader("🔎 Filtros para execução do agente")

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_redflag = st.selectbox(
            "Red Flag",
            ["Todos", "Sim", "Não"]
        )

    with col2:
        fornecedores_unicos = sorted(df_base["Fornecedor"].dropna().unique().tolist()) if "Fornecedor" in df_base.columns else []
        filtro_fornecedor = st.multiselect(
            "Fornecedor",
            fornecedores_unicos,
            default=fornecedores_unicos
        )

    with col3:
        if "Data" in df_base.columns:
            df_base["Data"] = pd.to_datetime(df_base["Data"], errors='coerce')
            data_min = df_base["Data"].min().date()
            data_max = df_base["Data"].max().date()
            filtro_periodo = st.date_input(
                "Período:",
                [data_min, data_max]
            )
        else:
            filtro_periodo = None

    # Aplicar filtros
    df_filtrado = df_base.copy()

    if filtro_redflag != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Red Flag'] == filtro_redflag]

    if filtro_fornecedor:
        df_filtrado = df_filtrado[df_filtrado['Fornecedor'].isin(filtro_fornecedor)]

    if filtro_periodo:
        data_inicio, data_fim = filtro_periodo
        df_filtrado = df_filtrado[
            (df_filtrado["Data"].dt.date >= data_inicio) &
            (df_filtrado["Data"].dt.date <= data_fim)
        ]

    st.markdown(f"🔸 **{len(df_filtrado)} registros encontrados após aplicação dos filtros.**")
    st.dataframe(df_filtrado.head())

    # 💰 Estimativa de Custo
    st.subheader("💰 Estimativa de Custo")

    tokens_estimados_por_linha = 150  # Aproximação média
    custo_por_1000_tokens = 0.01  # Custo aproximado GPT-4o

    total_tokens = len(df_filtrado) * tokens_estimados_por_linha
    custo_estimado = (total_tokens / 1000) * custo_por_1000_tokens

    st.info(f"🔢 Tokens estimados: {total_tokens} tokens")
    st.info(f"💰 Custo estimado: **USD {custo_estimado:.4f}**")

    executar = st.button(f"🚀 Executar Agente GPT-4o para {len(df_filtrado)} registros")

    if executar:
        st.info("🔧 O agente está analisando os dados. Isso pode levar alguns minutos...")

        resultados = []
        progresso = st.progress(0)

        for idx, (i, row) in enumerate(df_filtrado.iterrows()):
            try:
                dados = row.dropna().to_dict()

                prompt = f"""
                Você é um auditor especialista em compliance e análise de pagamentos corporativos.

                Analise o seguinte lançamento de pagamento extraído do SAP:

                {dados}

                Este lançamento foi previamente classificado como Red Flag = {row.get('Red Flag', 'Desconhecido')}.

                Com base nas informações apresentadas, responda:
                1. Se o Red Flag deve ser mantido ("Sim") ou removido ("Não").
                2. O motivo da sua decisão, de forma objetiva, clara e técnica.

                Responda no formato JSON:
                {{
                  "Red Flag Revisado": "Sim ou Não",
                  "Motivo": "explicação detalhada"
                }}
                """

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )

                resposta = response.choices[0].message.content.strip()

                try:
                    resposta_json = json.loads(resposta)
                    red_flag_revisado = resposta_json.get("Red Flag Revisado", "Não")
                    motivo = resposta_json.get("Motivo", "Motivo não identificado.")
                except json.JSONDecodeError:
                    red_flag_revisado = "Não"
                    motivo = f"Erro no parsing da resposta: {resposta}"

                resultados.append({
                    **dados,
                    "Red Flag": row.get('Red Flag', ''),
                    "Red Flag Revisado": red_flag_revisado,
                    "Motivo": motivo
                })

                progresso.progress((idx + 1) / len(df_filtrado))

                time.sleep(1)  # Delay opcional para evitar rate limit

            except Exception as e:
                st.error(f"Erro ao processar linha {i}: {e}")
                resultados.append({
                    **row.to_dict(),
                    "Red Flag Revisado": "Erro",
                    "Motivo": f"Erro na execução do agente: {e}"
                })
                continue

        progresso.empty()

        df_final = pd.DataFrame(resultados)

        st.success("✅ Análise do agente concluída.")
        st.dataframe(df_final.head())

        st.session_state['df_final'] = df_final

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


