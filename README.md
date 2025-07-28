
# 🛒 Agente de Compras

Sistema interativo em **Streamlit** com foco em **auditoria inteligente de pedidos de compras**, com integração ao **Teradata**, aplicação de **regras de compliance** e uso de **Inteligência Artificial (GPT-4o)** para detecção de fraudes, análise de red flags e revisão de casos suspeitos.

---

## 🚀 Funcionalidades Principais

- 📊 **Dashboard Analítico** com gráficos interativos (fornecedores, aprovadores, áreas).
- 🧠 **Agente de IA GPT-4o** para detecção automática de irregularidades (duplicidade, SoD, alçada, sobrepreço, compliance).
- ✅ **Revisão automatizada de flags** com parecer técnico usando IA.
- 🔎 **Filtros avançados**: pedido, fornecedor, material, área, aprovador, ano, mês.
- 📥 **Exportação de relatórios** em Excel com formatação.
- 🔒 **Conexão segura com Teradata** e uso de variáveis `.env` (OpenAI, DB).
- 📂 Análise de dados contábeis (MIRO, MIGO, adiantamentos, estornos).

---

## 🧱 Estrutura do Projeto

```
📁 agente-compras/
│
├── agente_compras.py     # Arquivo principal com o app Streamlit
├── Autorizadores.xlsx    # Base de usuários aprovadores
├── Base Materiais Compliance.xlsx  # Classificação de risco por grupo de materiais
├── .env                  # Chaves de API e credenciais de banco
├── requirements.txt      # Bibliotecas necessárias
└── README.md             # Esta documentação
```

---

## 🧪 Requisitos

- Python 3.9+
- Conexão ao banco **Teradata**
- Conta ativa na **OpenAI** com chave GPT-4o
- Variáveis de ambiente configuradas em `.env`:

```bash
OPENAI_API_KEY=sk-...
DB_HOST=...
DB_USER=...
DB_PASSWORD=...
```

---

## 📦 Instalação

```bash
# Clone o repositório
git clone https://github.com/seuusuario/agente-compras.git
cd agente-compras

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instale as dependências
pip install -r requirements.txt
```

---

## ▶️ Execução

```bash
streamlit run agente_compras.py
```

---

## 🧠 Inteligência Artificial

O projeto utiliza **OpenAI GPT-4o** em dois momentos:

1. **Agente Investigador**: Analisa os pedidos e identifica possíveis fraudes com base em critérios como:
   - Aprovação por pessoa indevida
   - Falta de segregação de funções (SoD)
   - Duplicidade de pedidos
   - Sobrepreço acima da média
   - Compliance crítico (consultorias, serviços, etc.)

2. **Agente Revisor**: Valida os casos sinalizados, gera parecer técnico e define se a flag é procedente.

---

## 📈 Dashboards

Inclui visualizações dinâmicas com Plotly:
- Distribuição por Fornecedor
- Gastos por Aprovador
- Alçada Efetiva vs Inefetiva
- Distribuição por Área

---

## 📤 Exportação

Relatórios exportados via `openpyxl` com formatação profissional:
- Pedidos filtrados
- Casos com red flags
- Base de inefetivos
- Resultado da auditoria com IA

---

## ⚠️ Regras de Auditoria Aplicadas

- 🔁 **Duplicidade de pedido** (mesmo valor, competência e fornecedor)
- ⚖️ **Alçada de aprovação** (valores e cargos autorizados)
- 🔐 **Segregação de Funções (SoD)** (Requisitante ≠ Aprovador ≠ Comprador)
- 📈 **Valor acima da média histórica**
- 🚩 **Materiais e serviços com risco elevado de compliance**
- 🧾 **Pedidos sem aprovador ou com aprovador sistêmico (FUS_/SAP)**

---

## 🧪 Tecnologias Utilizadas

- `Python`, `Streamlit`, `Pandas`, `Plotly`, `OpenAI`, `Teradatasql`
- `Scikit-learn` (normalização)
- `OpenPyXL` (formatação Excel)
- `dotenv`, `re`, `json`, `requests`

---

## 👨‍💻 Autor

**Carlos Vieira**  
Especialista em Auditoria

---
