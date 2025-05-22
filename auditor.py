import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
import json
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def preprocessar_base(df: pd.DataFrame):
    """
    🔧 Função robusta de pré-processamento:
    - Limpa nomes de colunas
    - Remove espaços, quebras e caracteres invisíveis
    - Converte possíveis colunas numéricas armazenadas como texto
    - Remove colunas 100% nulas
    - Remove linhas totalmente vazias
    - Remove colunas duplicadas (se houver)

    ✅ Retorna:
    - DataFrame limpo
    - Lista de colunas numéricas detectadas
    """

    st.subheader("🧽 Executando pré-processamento da base")

    # 🔗 Limpeza dos nomes das colunas
    df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '').str.replace('\t', '')

    # 🔍 Remover colunas 100% nulas
    colunas_nulas = df.columns[df.isnull().all()].tolist()
    if colunas_nulas:
        st.warning(f"⚠️ As seguintes colunas foram removidas por serem 100% nulas: {colunas_nulas}")
        df = df.drop(columns=colunas_nulas)

    # 🔍 Remover linhas totalmente vazias
    linhas_vazias = df.isnull().all(axis=1).sum()
    if linhas_vazias > 0:
        st.warning(f"⚠️ {linhas_vazias} linhas totalmente vazias foram removidas.")
        df = df.dropna(how='all')

    # 🔍 Remover colunas duplicadas
    if df.columns.duplicated().any():
        st.warning("⚠️ Foram encontradas colunas duplicadas e elas foram removidas.")
        df = df.loc[:, ~df.columns.duplicated()]

    # 🔧 Tentativa de conversão forçada para numérico
    df_converted = df.copy()
    for col in df.columns:
        df_converted[col] = pd.to_numeric(df[col], errors='ignore')

    # 🔍 Detectar colunas numéricas
    colunas_numericas = df_converted.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if not colunas_numericas:
        st.warning("⚠️ Nenhuma coluna foi detectada como numérica inicialmente. Tentando forçar a conversão...")

        for col in df.columns:
            df_converted[col] = pd.to_numeric(df[col], errors='coerce')

        colunas_numericas = df_converted.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if not colunas_numericas:
        st.error("🚫 Nenhuma coluna numérica encontrada após o pré-processamento.")
    else:
        st.success(f"✅ Colunas numéricas detectadas: {colunas_numericas}")

    return df_converted, colunas_numericas

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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd


if aba == "🏗️ Análise ML":
    st.header("🏗️ Clusterização + Classificação + Red Flag")

    uploaded_file = st.file_uploader("📤 Faça upload da base de pagamentos (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.subheader("📄 Pré-visualização da base")
        st.dataframe(df.head())

        # 🔧 Limpeza dos nomes das colunas
        df.columns = df.columns.str.strip()

        # 🔍 Selecionar colunas numéricas
        colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if not colunas_numericas:
            st.warning("⚠️ Nenhuma coluna numérica detectada inicialmente.")
            st.stop()

        # 🔍 Verificar percentual de nulos e remover colunas com mais de 50% de nulos
        limite_nulos = 0.5  # 50%
        colunas_validas = [
            col for col in colunas_numericas
            if df[col].isnull().mean() < limite_nulos
        ]

        if not colunas_validas:
            st.error("🚫 Nenhuma coluna válida encontrada após filtro de nulos (>50%).")
            st.stop()

        st.info(f"✔️ Colunas válidas para análise: {colunas_validas}")

        # 🔧 Preencher nulos temporariamente para calcular variância
        df_fill = df[colunas_validas].fillna(0)

        # 🔧 Padronização dos dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_fill)

        # 📊 Calcular variância padronizada
        variancias = np.var(X_scaled, axis=0)
        peso_colunas = variancias / variancias.sum()

        df_pesos = pd.DataFrame({
            'Feature': colunas_validas,
            'Peso (%)': peso_colunas * 100
        }).sort_values(by="Peso (%)", ascending=False)

        st.subheader("📑 Tabela de Pesos das Variáveis")
        st.dataframe(df_pesos)

        # 📈 Plotar gráfico interativo
        fig_pesos = px.bar(
            df_pesos,
            x='Feature',
            y='Peso (%)',
            title="📊 Peso Estatístico Inicial das Variáveis",
            text_auto='.2f'
        )
        st.plotly_chart(fig_pesos, use_container_width=True)

        st.subheader("🎯 Selecione as colunas para clusterização e classificação")
        selected_columns = st.multiselect(
            "Selecione as colunas com maior relevância para o modelo",
            colunas_validas,
            default=df_pesos['Feature'].head(5).tolist()  # Sugere as top 5
        )

        if selected_columns:
            st.success(f"Colunas selecionadas: {selected_columns}")

            # 🔧 Padroniza novamente os dados selecionados
            X_scaled = scaler.fit_transform(df[selected_columns].fillna(0))

            # 🔍 Avaliação do melhor número de clusters
            st.subheader("🔢 Avaliação Automática do Número de Clusters")

            sil_scores = []
            inertias = []
            k_range = range(2, 10)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                labels = kmeans.labels_
                sil = silhouette_score(X_scaled, labels)
                sil_scores.append(sil)
                inertias.append(kmeans.inertia_)

            # Plot Elbow + Silhouette
            fig, ax1 = plt.subplots()

            color = 'tab:blue'
            ax1.set_xlabel('Número de Clusters')
            ax1.set_ylabel('Inertia (Elbow)', color=color)
            ax1.plot(k_range, inertias, marker='o', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()

            color = 'tab:green'
            ax2.set_ylabel('Silhouette Score', color=color)
            ax2.plot(k_range, sil_scores, marker='x', linestyle='--', color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            st.pyplot(fig)

            # 🏆 Melhor número de clusters
            melhor_k = k_range[sil_scores.index(max(sil_scores))]
            st.success(f"📈 Número sugerido de clusters: **{melhor_k}**")

            k_selecionado = st.number_input(
                "Ajuste manual do número de clusters (opcional)",
                min_value=2,
                max_value=10,
                value=melhor_k,
                step=1
            )

            # 🚀 Clusterização
            kmeans = KMeans(n_clusters=k_selecionado, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            df['Cluster'] = clusters

            # 📊 PCA para visualização
            st.subheader("📊 Visualização dos Clusters")
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df_plot = pd.DataFrame(components, columns=['Componente 1', 'Componente 2'])
            df_plot['Cluster'] = clusters.astype(str)

            fig = px.scatter(
                df_plot,
                x="Componente 1",
                y="Componente 2",
                color="Cluster",
                title="Distribuição dos Clusters",
                width=800,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # 🔥 Importância Real das Features com Random Forest
            st.subheader("🛠️ Importância Real das Features")

            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_scaled, clusters)

            importances = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': selected_columns,
                'Importância': importances
            }).sort_values(by="Importância", ascending=False)

            fig_pesos = px.bar(
                feature_importance,
                x="Feature",
                y="Importância",
                title="Importância das Features (Random Forest)",
                text_auto='.2f'
            )
            st.plotly_chart(fig_pesos, use_container_width=True)

            # 🚩 Classificação para Red Flag
            st.subheader("🚩 Classificação Supervisionada para Identificação de Red Flags")

            red_flag_cluster = st.multiselect(
                "Selecione os clusters que serão considerados como **Red Flag**",
                options=sorted(df['Cluster'].unique().tolist()),
                default=[max(df['Cluster'])]
            )

            df['Red Flag'] = df['Cluster'].apply(lambda x: 'Sim' if x in red_flag_cluster else 'Não')

            st.subheader("📑 Pré-visualização da base final")
            st.dataframe(df.head())

            # ✔️ Salvar no session_state para a Aba 2
            st.session_state['df_redflag'] = df

            st.download_button(
                label="💾 Baixar base com Red Flags (Aba 1)",
                data=df.to_csv(index=False, sep=";", encoding='utf-8-sig'),
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
        fornecedores_unicos = ["Todos"] + sorted(df_base["Fornecedor"].dropna().unique().tolist()) if "Fornecedor" in df_base.columns else []
        filtro_fornecedor = st.selectbox(
            "Fornecedor",
            fornecedores_unicos
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

    if filtro_fornecedor != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Fornecedor'] == filtro_fornecedor]

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


