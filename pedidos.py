import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import calendar
import time
import io
import teradatasql
import os
import json
import re
import openai
import urllib3
from dotenv import load_dotenv
import numpy as np
import ssl
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Font
from openpyxl.styles.numbers import BUILTIN_FORMATS
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import logging


st.set_page_config(layout = 'wide')

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.title('AGENTE DE COMPRAS :shopping_trolley:')

def formata_numero(valor, prefixo=''):
    for unidade in ['', 'mil', 'milhões', 'bilhões']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}'.strip()
        valor /= 1000
    return f'{prefixo} {valor:.2f} trilhões'

def converte_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')  # Exportar para Excel sem índice
    return output.getvalue()

def gerar_excel_formatado(df_export):
    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Pedidos_Agente"

    # Adiciona o cabeçalho
    for r_idx, row in enumerate(dataframe_to_rows(df_export, index=False, header=True)):
        ws.append(row)
        if r_idx == 0:
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")

    # Formata colunas numéricas e datas no padrão brasileiro
    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                if isinstance(cell.value, float):
                    cell.number_format = '#.##0,00'
                elif isinstance(cell.value, int):
                    cell.number_format = '0'
                elif isinstance(cell.value, str) and '/' in cell.value and len(cell.value) == 10:
                    cell.number_format = 'DD/MM/YYYY'
            except:
                pass
            # Autoajuste de largura
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max_length + 2

    wb.save(output)
    return output.getvalue()

# Credenciais de conexão

HOST = os.getenv("DB_HOST")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
SCHEMA = "AA_PRD_DDM"
SCHEMA2 = "AA_PRD_WRK"

@st.cache_data(show_spinner="🔍 Trazendo dados do Teradata...")
def base_teradata():
    try:
        with teradatasql.connect(host=HOST, user=USER, password=PASSWORD) as conn:
                    print("Conexão bem-sucedida!")
                    
                    with conn.cursor() as cur:
                        
                        query_itens = f"""
                            SELECT
                                "PurchaseOrder",
                                "PurchaseOrderItem",
                                "PurchaseOrderCategory",
                                "DocumentCurrency",
                                "MaterialGroup",
                                "Material",
                                "MaterialType",
                                "PurchaseOrderItemText",
                                "CompanyCode",
                                "IsFinallyInvoiced",
                                "NetAmount",
                                "GrossAmount",
                                "EffectiveAmount",
                                "NetPriceAmount",
                                "OrderQuantity",
                                "NetPriceQuantity",
                                "PurgDocPriceDate",
                                "PurchaseRequisition",
                                "RequisitionerName",
                                "PurchaseContract",
                                "AccountAssignmentCategory"
                            FROM {SCHEMA2}.I_PurchaseOrderItemAPI01
                            WHERE "PurchasingDocumentDeletionCode" = ''
                        """
                        cur.execute(query_itens)
                        columns_itens = [desc[0] for desc in cur.description]
                        df = pd.DataFrame(cur.fetchall(), columns=columns_itens)
                        
                        query_po = f"""
                            SELECT
                                "PurchaseOrder",
                                "CreatedByUser",
                                "PurchasingProcessingStatus",
                                "Supplier",
                                "ZZ1_Aprovador1_PDH",
                                "ZZ1_Aprovador2_PDH",
                                "ZZ1_Aprovador3_PDH",
                                "ZZ1_Aprovador4_PDH",
                                "PurgReleaseTimeTotalAmount",
                                "ExchangeRate",
                                "PurchaseOrderDate"
                            FROM {SCHEMA2}.I_PurchaseOrderAPI01
                        """
                        cur.execute(query_po)
                        columns_po = [desc[0] for desc in cur.description]
                        df_aprov = pd.DataFrame(cur.fetchall(), columns=columns_po)
                        
                        query_entrega= f"""
                            SELECT
                                "PurchaseOrder",
                                "PurchaseOrderItem",
                                "ScheduleLineDeliveryDate"
                            FROM {SCHEMA2}.I_PurOrdScheduleLineAPI01
                        """
                        cur.execute(query_entrega)
                        columns_entrega = [desc[0] for desc in cur.description]
                        df_entrega = pd.DataFrame(cur.fetchall(), columns=columns_entrega)
                        
                        query_rc= f"""
                            SELECT
                                "PurchaseRequisition",
                                "PurchaseRequisitionType"
                            FROM {SCHEMA2}.I_PurchaseRequisitionAPI01
                        """
                        cur.execute(query_rc)
                        columns_rc = [desc[0] for desc in cur.description]
                        df_rc = pd.DataFrame(cur.fetchall(), columns=columns_rc)
                        
                        query_fornecedor= f"""
                            SELECT
                                "Supplier",
                                "SupplierName"
                            FROM {SCHEMA2}.I_Supplier
                        """
                        cur.execute(query_fornecedor)
                        columns_fornecedor = [desc[0] for desc in cur.description]
                        df_fornecedor = pd.DataFrame(cur.fetchall(), columns=columns_fornecedor)
                        
                        query_journal= f"""
                            SELECT
                                "CompanyCode",
                                "CompanyCodeName",
                                "FiscalYear",
                                "AccountingDocument",
                                "LedgerGLLineItem",
                                "ReferenceDocument",
                                "ReversalReferenceDocument",
                                "GLAccount",
                                "GLAccountLongName",
                                "CostCenter",
                                "CostCenterName",
                                "BalanceTransactionCurrency",
                                "AmountInTransactionCurrency",
                                "GlobalCurrency",
                                "AmountInGlobalCurrency",
                                "FreeDefinedCurrency1",
                                "AmountInFreeDefinedCurrency1",
                                "PostingDate",
                                "DocumentDate",
                                "AccountingDocumentType",
                                "AccountingDocumentTypeName",
                                "AccountingDocCreatedByUser",
                                "DocumentItemText",
                                "OffsettingAccount",
                                "OffsettingAccountName",
                                "ClearingAccountingDocument",
                                "ClearingDate",
                                "PurchasingDocument",
                                "PurchasingDocumentItem",
                                "Material",
                                "MaterialName"
                            FROM {SCHEMA2}.I_JournalEntryItemCube
                            WHERE (TRIM("ReversalReferenceDocument") = '' OR "ReversalReferenceDocument" IS NULL)
                                AND "PurchasingDocument" <> ''
                        """
                        cur.execute(query_journal)
                        columns_journal = [desc[0] for desc in cur.description]
                        df_journal = pd.DataFrame(cur.fetchall(), columns=columns_journal)
                        
                        query_estorno= f"""
                            SELECT
                                "CompanyCode",
                                "FiscalYear",
                                "ReversalReferenceDocument"
                            FROM {SCHEMA2}.I_JournalEntryItemCube
                            WHERE (TRIM("ReversalReferenceDocument") <> '' OR "ReversalReferenceDocument" IS NOT NULL)
                                AND "PurchasingDocument" <> ''           
                        """
                        cur.execute(query_estorno)
                        columns_estorno = [desc[0] for desc in cur.description]
                        df_estorno = pd.DataFrame(cur.fetchall(), columns=columns_estorno)
                        
    except Exception as e:
        print(f'Identificamos o erro: {e}')
        
    return df, df_aprov, df_entrega, df_rc, df_fornecedor, df_journal, df_estorno

df, df_aprov, df_entrega, df_rc, df_fornecedor, df_journal, df_estorno = base_teradata()                   

# -----------------------------
# Tratamento da Base de Itens
# -----------------------------

df["PurgDocPriceDate"] = df["PurgDocPriceDate"].astype(str).str.replace("2202", "2022", regex=False)
df["PurgDocPriceDate"] = pd.to_datetime(df["PurgDocPriceDate"])

# -----------------------------
# Tratamento da Base de POs
# -----------------------------

#Retirar 08(Rejeitado), 02(Eliminado)

df_aprov["PurchasingProcessingStatus"] = df_aprov["PurchasingProcessingStatus"].str.strip()
df_aprov = df_aprov[~df_aprov["PurchasingProcessingStatus"].isin(["08","02"])]


df_aprov["Aprovador"] = df_aprov.apply(
lambda row: row["ZZ1_Aprovador4_PDH"] if row["ZZ1_Aprovador4_PDH"] != "N/A" else (
                        row["ZZ1_Aprovador3_PDH"] if row["ZZ1_Aprovador3_PDH"] != "N/A" else (
                            row["ZZ1_Aprovador2_PDH"] if row["ZZ1_Aprovador2_PDH"] != "N/A" else (
                                row["ZZ1_Aprovador1_PDH"] if row["ZZ1_Aprovador1_PDH"] != "N/A" else "Não Aprovado"
                            )
                        )
                    ), axis=1
                )

df_autorizadores = pd.read_excel("Autorizadores.xlsx", sheet_name="Autorizadores", header=0)
df_autorizadores = df_autorizadores.drop(df_autorizadores.columns[[0, 1, -1]], axis=1)
df_autorizadores.columns = df_autorizadores.iloc[0]
df_autorizadores = df_autorizadores[1:].reset_index(drop=True)
df_autorizadores = df_autorizadores.rename(columns={"Uusário SAP": "Aprovador", "Área Autorizador": "Area_Autorizador"})
df_autorizadores = df_autorizadores[["Aprovador", "Cargo", "Area_Autorizador"]]   

df_classificacao_risco = pd.read_excel("Base Materiais Compliance.xlsx", sheet_name="Classificação de Risco")
df_classificacao_risco = df_classificacao_risco.iloc[4:]
df_classificacao_risco.columns = df_classificacao_risco.iloc[0]
df_classificacao_risco = df_classificacao_risco[1:].reset_index(drop=True)
df_classificacao_risco.columns = df_classificacao_risco.iloc[0]
df_classificacao_risco = df_classificacao_risco.iloc[1:]
df_classificacao_risco = df_classificacao_risco.rename(columns={
                    "TIPO DE FORNECIMENTO": "Tipo de Fornecimento",
                    "GRUPO DE MERCADORIA": "PurchasingMaterialGroup",
                    "CATEGORIA": "Categoria",
                    "SUBCATEGORIA": "Subcategoria",
                    "Saúde e SeguranCa": "Saude e Seguranca",
                    "Operacional": "Operacional",
                    "Meio Ambiente": "Meio Ambiente",
                    "COMPLIANCE": "Compliance",
                    "RESULTADO": "Resultado"
                })

df_aprov["Aprovador"] = df_aprov["Aprovador"].replace({"":"Não Informado"})
df_aprov["PurgReleaseTimeTotalAmount"] = pd.to_numeric(df_aprov["PurgReleaseTimeTotalAmount"], errors="coerce").fillna(0).astype(int)
df_aprov = df_aprov.merge(df_autorizadores[["Aprovador", "Cargo", "Area_Autorizador"]], on="Aprovador", how="left")

df_aprov = df_aprov.drop_duplicates(subset=["PurchaseOrder"])
df = df.merge(df_aprov[['PurchaseOrder', 'CreatedByUser','PurchasingProcessingStatus','Supplier','PurchaseOrderDate','PurgReleaseTimeTotalAmount','ExchangeRate','Aprovador',"Cargo", "Area_Autorizador"]], on="PurchaseOrder", how="left")
df = df[~df["PurchasingProcessingStatus"].isna()]
df["Cargo"] = df["Cargo"].fillna("Não Informado")
df["Area_Autorizador"] = df["Area_Autorizador"].fillna("Não Informado")
df["PurgReleaseTimeTotalAmount"] = pd.to_numeric(
    df["PurgReleaseTimeTotalAmount"], errors="coerce"
).astype("Int64")

df["EffectiveAmount"] = df["EffectiveAmount"].astype(np.int64)
df["NetAmount"] = df["NetAmount"].astype(np.int64)
df["GrossAmount"] = df["GrossAmount"].astype(np.int64)
df["NetPriceAmount"] = df["NetPriceAmount"].astype("int")
df["OrderQuantity"] = df["OrderQuantity"].astype("int")
df["ExchangeRate"] = df["ExchangeRate"].astype("int")
df["NetPriceQuantity"] = df["NetPriceQuantity"].astype("int")

df["DocumentCurrency"] = df["DocumentCurrency"].str.strip()
is_brl = df["DocumentCurrency"] == "BRL"
df.loc[is_brl, "Valor_Item_BRL"] = df.loc[is_brl, "NetPriceAmount"]
df.loc[is_brl, "Valor_PO_BRL"] = df.loc[is_brl, "PurgReleaseTimeTotalAmount"]
df.loc[~is_brl, "Valor_Item_BRL"] = df.loc[~is_brl, "NetPriceAmount"] * df.loc[~is_brl, "ExchangeRate"]
df.loc[~is_brl, "Valor_PO_BRL"] = df.loc[~is_brl, "PurgReleaseTimeTotalAmount"] * df.loc[~is_brl, "ExchangeRate"]


df["Check_SOD"] = ((df["CreatedByUser"].str.strip() == df["Aprovador"].str.strip()) | 
                   (df["RequisitionerName"].str.strip() == df["Aprovador"].str.strip()) ).map({True: "Sim", False: "Não"})

df_classificacao_risco = df_classificacao_risco.rename(columns={'PurchasingMaterialGroup':'MaterialGroup'})
df = df.merge(df_classificacao_risco[['MaterialGroup', "Resultado"]], on="MaterialGroup", how="left")
df["Resultado"] = df["Resultado"].fillna("Muito Baixo")

conditions = [
    ((df["Valor_PO_BRL"] > 500000) & (df["Cargo"] == "Diretor")),
    ((df["Valor_PO_BRL"] > 15000) & (df["Cargo"].isin(["Diretor", "Gerente"]))),
    ((df["Cargo"].isin(["Coordenador", "Gerente", "Diretor"])))
]

choices = ["Efetivo"] * len(conditions)
df["Status_Aprovacao"] = np.select(conditions, choices, default="Não Efetivo")

df["chave_entrega"] = df["PurchaseOrder"] + df["PurchaseOrderItem"]
df_entrega["chave_entrega"] = df_entrega["PurchaseOrder"] + df_entrega["PurchaseOrderItem"]

df = df.merge(df_entrega[["chave_entrega", "ScheduleLineDeliveryDate"]], on = "chave_entrega", how = "left")
df = df.drop(columns=["chave_entrega"])

df = df.merge(df_fornecedor[["Supplier", "SupplierName"]], on = "Supplier", how = "left")

mapa_contabil = {
    "K":"Opex",
    "":"Não Informado",
    "A":"Imobilizado",
    "F":"Ordem Interna",
    "U":"Desconhecido"
    }

df["contabil"] = df["AccountAssignmentCategory"].str.strip().map(mapa_contabil)
df = df.drop(columns = ["PurchaseOrderCategory", "AccountAssignmentCategory"])

df["chave_fornecedor"] = (
    df["Supplier"].astype(str) +
    df["PurgReleaseTimeTotalAmount"].astype(str) +
    pd.to_datetime(df["PurchaseOrderDate"]).dt.month.astype(str) +
    pd.to_datetime(df["PurchaseOrderDate"]).dt.year.astype(str)
)

df_fornecedor = df.drop_duplicates(subset="PurchaseOrder")
df_fornecedor = df_fornecedor[df_fornecedor["PurgReleaseTimeTotalAmount"]>0]
df_fornecedor = df_fornecedor.groupby(["chave_fornecedor"])["PurchaseOrder"].agg("count").reset_index()
df_fornecedor = df_fornecedor.rename(columns={"PurchaseOrder":"Quantidade"})

df = df.merge(df_fornecedor[["chave_fornecedor", "Quantidade"]], on = "chave_fornecedor", how = "left")
df = df.fillna(0)

df["duplicidade"] = df["Quantidade"].apply(lambda x: "Sim" if x > 1 else "Não")
df = df.drop(columns=["Quantidade","chave_fornecedor"])

# -----------------------------
# Tratamento da Base do FI
# -----------------------------

df_journal["chave_estorno"] = df_journal["AccountingDocument"] + df_journal["FiscalYear"] + df_journal["CompanyCode"]
df_estorno["chave_estorno"] = df_estorno["ReversalReferenceDocument"] + df_estorno["FiscalYear"] + df_estorno["CompanyCode"]
df_estorno = df_estorno[df_estorno["ReversalReferenceDocument"] != '']

lista_estorno = list(df_estorno["chave_estorno"].unique())

df_journal = df_journal[~df_journal["chave_estorno"].isin(lista_estorno)]
df_journal = df_journal.drop(columns=["ReversalReferenceDocument","chave_estorno"])

df_journal["CostCenter"] = df_journal["CostCenter"].fillna("Não Informado")
df_journal["CostCenterName"] = df_journal["CostCenterName"].fillna("Não Informado")
df_journal["DocumentItemText"] = df_journal["DocumentItemText"].fillna("Não Informado")
df_journal["ClearingAccountingDocument"] = df_journal["ClearingAccountingDocument"].fillna("Não Compensado")
df_journal["ClearingDate"] = df_journal["ClearingDate"].fillna("Não Compensado")

df_journal["CostCenter"] = df_journal["CostCenter"].replace({"":"Não Informado"})
df_journal["CostCenterName"] = df_journal["CostCenterName"].replace({"":"Não Informado"})
df_journal["DocumentItemText"] = df_journal["DocumentItemText"].replace({"":"Não Informado"})
df_journal["ClearingAccountingDocument"] = df_journal["ClearingAccountingDocument"].replace({"":"Não Compensado"})
df_journal["ClearingDate"] = df_journal["ClearingDate"].replace({"":"Não Compensado"})
df_journal["DocumentItemText"] = df_journal["DocumentItemText"].replace({"":"Não Informado"})
df_journal["OffsettingAccountName"] = df_journal["OffsettingAccountName"].replace({"":"Não Informado"})
df_journal["PurchasingDocument"] = df_journal["PurchasingDocument"].replace({"":"Não Informado"})
df_journal["PurchasingDocument"] = df_journal["PurchasingDocument"].replace({"0":"Não Informado"})

df_journal["PostingDate"] = pd.to_datetime(df_journal["PostingDate"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_journal["DocumentDate"] = pd.to_datetime(df_journal["DocumentDate"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')

df_journal = df_journal[~df_journal["DocumentDate"].isnull()]

df_journal["AmountInTransactionCurrency"] = df_journal["AmountInTransactionCurrency"].astype("float")
df_journal["AmountInGlobalCurrency"] = df_journal["AmountInGlobalCurrency"].astype("float")
df_journal["AmountInFreeDefinedCurrency1"] = df_journal["AmountInFreeDefinedCurrency1"].astype("float")

#CENTRO DE CUSTO

df_cc = df_journal[df_journal["AccountingDocument"].astype(str).str.startswith("51")]
df_cc = df_cc[["CostCenter","CostCenterName","PurchasingDocument", "PurchasingDocumentItem"]]
df_cc["chave_miro"] = df_cc["PurchasingDocument"] + df_cc["PurchasingDocumentItem"]
df_cc = df_cc.drop_duplicates(subset="chave_miro")

#MIRO

df_miro = df_journal[df_journal["AccountingDocument"].astype(str).str.startswith("51")]
df_miro = df_miro[["ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate","PurchasingDocument",
                   "PurchasingDocumentItem","Material","MaterialName"]]

df_teste = df.copy()
df_teste["chave_miro"] = df_teste["PurchaseOrder"] + df_teste["PurchaseOrderItem"]
df_miro["chave_miro"] = df_miro["PurchasingDocument"] + df_miro["PurchasingDocumentItem"]

df_miro_grupo = df_miro.groupby("chave_miro")["AmountInFreeDefinedCurrency1"].agg("sum").reset_index()
df_miro = df_miro.drop_duplicates(subset="chave_miro")
df_miro_grupo = df_miro_grupo.merge(df_miro[["chave_miro","ReferenceDocument","PostingDate"]], on = "chave_miro", how = "left")
df_teste = df_teste.merge(df_miro_grupo[["chave_miro","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_miro", how = "left")
df_teste = df_teste.rename(columns={"ReferenceDocument":"MIRO","AmountInFreeDefinedCurrency1":"valor_miro","PostingDate":"data_miro"})

df_teste = df_teste.merge(df_cc[["chave_miro","CostCenter","CostCenterName"]], on = "chave_miro", how = "left")

df_isa = df.copy()

#MIGO

df_migo = df_journal[
    (df_journal["AccountingDocument"].astype(str).str.startswith("50")) &
    (df_journal["GLAccountLongName"] == "MATERIAIS EM ALMOXARIFADO")
]

df_migo = df_migo[["ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate","PurchasingDocument",
                   "PurchasingDocumentItem","Material","MaterialName"]]

df_teste["chave_migo"] = df_teste["PurchaseOrder"] + df_teste["PurchaseOrderItem"]
df_migo["chave_migo"] = df_migo["PurchasingDocument"] + df_migo["PurchasingDocumentItem"]
df_migo_grupo = df_migo.groupby("chave_migo")["AmountInFreeDefinedCurrency1"].agg("sum").reset_index()
df_migo = df_migo.drop_duplicates(subset="chave_migo")
df_migo_grupo = df_migo_grupo.merge(df_migo[["chave_migo","ReferenceDocument","PostingDate"]], on = "chave_migo", how = "left")

df_teste = df_teste.merge(df_migo[["chave_migo","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_migo", how = "left")
df_teste = df_teste.rename(columns={"ReferenceDocument":"MIGO","AmountInFreeDefinedCurrency1":"valor_migo","PostingDate":"data_migo"})

#Adiantamento

lista_adiantamento_contas = ["(-) ADIANTAMENTO P AQUISIÇÃO DE INVESTIMENTO","ADIANTAMENTO A FORNECEDOR NO EXTERIOR",
                        "ADIANTAMENTO A FORNECEDOR NO PAIS","ADIANTAMENTO P/AQUISICAO DE INVESTIMENTO"]           

df_adiantamento = df_journal[df_journal["GLAccountLongName"].isin(lista_adiantamento_contas)]
df_adiantamento = df_adiantamento.rename(columns={"PurchasingDocument": "PurchaseOrder"})
df_adiantamento = df_adiantamento[df_adiantamento["PurchaseOrder"] != "Não Informado"]

df_adiantamento_pagamento = df_adiantamento[df_adiantamento["AccountingDocumentType"]=="ZP"].copy()
df_adiantamento_fatura = df_adiantamento[df_adiantamento["AccountingDocumentType"].isin(["KR","KZ","KP"])].copy()

df_adiantamento_pagamento["chave_adto"] = df_adiantamento_pagamento["PurchaseOrder"] + df_adiantamento_pagamento["PurchasingDocumentItem"]
df_adiantamento_pagamento_agrupado = df_adiantamento_pagamento.groupby("chave_adto")["AmountInFreeDefinedCurrency1"].agg("sum").reset_index()
df_adiantamento_pagamento = df_adiantamento_pagamento.drop_duplicates(subset="chave_adto")
df_adiantamento_pagamento_agrupado = df_adiantamento_pagamento_agrupado.merge(df_adiantamento_pagamento[["chave_adto","ReferenceDocument","PostingDate"]],
                                                                              on = "chave_adto", how = "left")

df_adiantamento_fatura["chave_adto"] = df_adiantamento_fatura["PurchaseOrder"] + df_adiantamento_fatura["PurchasingDocumentItem"]
df_adiantamento_fatura_agrupado = df_adiantamento_fatura.groupby("chave_adto")["AmountInFreeDefinedCurrency1"].agg("sum").reset_index()
df_adiantamento_fatura = df_adiantamento_fatura.drop_duplicates(subset="chave_adto")
df_adiantamento_fatura_agrupado = df_adiantamento_fatura_agrupado.merge(df_adiantamento_fatura[["chave_adto","ReferenceDocument","PostingDate"]],
                                                                              on = "chave_adto", how = "left")

df_teste["chave_adto"] = df_teste["PurchaseOrder"] + df_teste["PurchaseOrderItem"]
df_isa["chave_adto"] = df_isa["PurchaseOrder"] + df_isa["PurchaseOrderItem"]

df_teste = df_teste.merge(df_adiantamento_pagamento_agrupado[["chave_adto","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_adto", how = "left")
df_teste = df_teste.rename(columns={"ReferenceDocument":"ADTO","AmountInFreeDefinedCurrency1":"valor_adto","PostingDate":"data_adto"})
df_isa = df_isa.merge(df_adiantamento_pagamento_agrupado[["chave_adto","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_adto", how = "left")
df_isa = df_isa.rename(columns={"ReferenceDocument":"ADTO","AmountInFreeDefinedCurrency1":"valor_adto","PostingDate":"data_adto"})

df_teste = df_teste.merge(df_adiantamento_fatura_agrupado[["chave_adto","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_adto", how = "left")
df_teste = df_teste.rename(columns={"ReferenceDocument":"ADTO_Fatura","AmountInFreeDefinedCurrency1":"fatura_adto","PostingDate":"data_fatura"})
df_isa = df_isa.merge(df_adiantamento_fatura_agrupado[["chave_adto","ReferenceDocument","AmountInFreeDefinedCurrency1","PostingDate"]], on = "chave_adto", how = "left")
df_isa = df_isa.rename(columns={"ReferenceDocument":"ADTO_Fatura","AmountInFreeDefinedCurrency1":"fatura_adto","PostingDate":"data_fatura"})

df_teste["fatura_adto"] = df_teste["fatura_adto"].fillna(0)
df_teste["saldo_adto"] = df_teste["valor_adto"] + df_teste["fatura_adto"]
df_isa["fatura_adto"] = df_isa["fatura_adto"].fillna(0)
df_isa["saldo_adto"] = df_isa["valor_adto"] + df_isa["fatura_adto"]

media_por_material = df_teste.groupby("Material")["NetPriceAmount"].mean().reset_index()
media_por_material = media_por_material.rename(columns={"NetPriceAmount": "media_material"})

df_teste = df_teste.merge(media_por_material, on="Material", how="left")

def checar_media(row):
    if row["MaterialType"].strip() == "SERV":
        return "Serviço - N/A"
    elif row["NetPriceAmount"] > row["media_material"] * 1.5:
        return "Sim"
    else:
        return "Não"

df_teste["check_media"] = df_teste.apply(checar_media, axis=1)
df_teste["media_material"] = df_teste["media_material"].round(2)

# -----------------------------
# Tratamento para o Aplicativo
# -----------------------------

df_app = df_teste[[
    "PurchaseOrder",
    "PurchaseOrderDate",
    "CompanyCode",
    "contabil",
    "PurchaseOrderItem",
    "Material",
    "PurchaseOrderItemText",
    "MaterialGroup",
    "Resultado",
    "OrderQuantity",
    "NetPriceQuantity",
    "NetPriceAmount",
    "PurgReleaseTimeTotalAmount",
    "DocumentCurrency",
    "Valor_Item_BRL",
    "Valor_PO_BRL",
    "PurchaseRequisition",
    "RequisitionerName",
    "CreatedByUser",
    "PurchaseContract",
    "Supplier",
    "SupplierName",
    "Aprovador",
    "Cargo",
    "Area_Autorizador",
    "CostCenter",
    "CostCenterName",
    "ADTO",
    "valor_adto",
    "data_adto",
    "ADTO_Fatura",
    "fatura_adto",
    "data_fatura",
    "saldo_adto",
    "ScheduleLineDeliveryDate",
    "MIGO",
    "data_migo",
    "valor_migo",
    "MIRO",
    "data_miro",
    "valor_miro",
    "duplicidade",
    "Check_SOD",
    "Status_Aprovacao",
    "check_media",
    "media_material"
    ]]

df_app = df_app.rename(columns={
    "PurchaseOrder":"Pedido",
    "PurchaseOrderDate":"Data Pedido",
    "CompanyCode":"Empresa",
    "contabil":"Classe Contabil",
    "PurchaseOrderItem":"Item",
    "Material":"Numero Material",
    "PurchaseOrderItemText":"Nome Material",
    "MaterialGroup":"Grupo do Material",
    "Resultado": "Compliance",
    "OrderQuantity":"Quantidade",
    "NetPriceQuantity":"Dimensao Quantidade",
    "NetPriceAmount":"Valor Item",
    "PurgReleaseTimeTotalAmount":"Valor Pedido",
    "DocumentCurrency":"Moeda",
    "Valor_Item_BRL":"Valor Item BRL",
    "Valor_PO_BRL":"Valor Pedido BRL",
    "PurchaseRequisition":"RC",
    "RequisitionerName":"Requisitante",
    "CreatedByUser":"Comprador",
    "PurchaseContract":"Contrato",
    "Supplier":"Numero Fornecedor",
    "SupplierName":"Nome Fornecedor",
    "Aprovador":"Aprovador",
    "Cargo":"Cargo",
    "Area_Autorizador":"Area",
    "CostCenter":"Numero CC",
    "CostCenterName":"Nome CC",
    "ADTO":"Doc Adto",
    "valor_adto":"Valor Adiantado",
    "data_adto":"Data Adiantamento",
    "ADTO_Fatura":"Doc Baixa Adto",
    "fatura_adto":"Baixa Adto",
    "data_fatura":"Data Baixa Adto",
    "saldo_adto":"Saldo Adto",
    "ScheduleLineDeliveryDate":"Data Entrega",
    "MIGO":"Migo",
    "data_migo":"Data Migo",
    "valor_migo":"Valor Migo",
    "MIRO":"Miro",
    "data_miro":"Data Miro",
    "valor_miro":"Valor Miro",
    "duplicidade":"Check Duplicidade",
    "Check_SOD":"Check SoD",
    "Status_Aprovacao":"Check Aprovador",
    "check_media":"Check Valor",
    "media_material": "Média do Material"
    })

df_app["Data Pedido"] = pd.to_datetime(df_app["Data Pedido"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Data Entrega"] = pd.to_datetime(df_app["Data Entrega"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Data Migo"] = pd.to_datetime(df_app["Data Migo"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Data Miro"] = pd.to_datetime(df_app["Data Miro"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Data Adiantamento"] = pd.to_datetime(df_app["Data Adiantamento"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Data Baixa Adto"] = pd.to_datetime(df_app["Data Baixa Adto"], errors="coerce", dayfirst=True).dt.strftime('%d/%m/%Y')
df_app["Check Aprovador"] = df_app["Check Aprovador"].replace({"Não Efetivo":"Inefetivo"})
df_app["Valor Linha BRL"] = (df_app["Quantidade"] * df_app["Valor Item BRL"]) / df_app["Dimensao Quantidade"]
df_app["Valor Linha BRL"] = df_app["Valor Linha BRL"].astype("int")
df_app["Numero CC"] = df_app["Numero CC"].fillna(df_app['Area'])
df_app["Nome CC"] = df_app["Nome CC"].fillna(df_app['Area'])
df_app = df_app.drop(['Area'], axis=1)

# -----------------------------
# Filtros para o Aplicativo
# -----------------------------

df_app["Data Pedido2"] = pd.to_datetime(df_app["Data Pedido"], errors="coerce", dayfirst=True)
df_app = df_app.sort_values(by= "Data Pedido2", ascending = False)

df_app["Ano"] = df_app["Data Pedido2"].dt.year.fillna(df_app["Data Pedido"].str[-4:]).astype("int").astype(str)
df_app["Mes"] = df_app["Data Pedido2"].dt.month.fillna(df_app["Data Pedido"].str[3:-5]).astype("int").astype(str)
df_app["Ano"] = df_app["Ano"].replace({"2202":"2022"})
df_app["Doc Adto"] = df_app["Doc Adto"].fillna("Sem Documento")

lista_pedidos = ["Todos"] + sorted(list(df_app["Pedido"].str.strip().unique()))
lista_fornecedores = ["Todos"] + sorted(list(df_app["Nome Fornecedor"].str.strip().unique()))
lista_areas = ["Todas"] + sorted(list(df_app["Nome CC"].str.strip().unique()))
lista_anos = ["Todos"] + sorted(df_app["Ano"].dropna().unique())
lista_meses = ["Todos"] + sorted(df_app["Mes"].dropna().unique(), key=lambda x: int(x))
lista_aprovadores = ["Todos"] + sorted(list(df_app["Aprovador"].unique()))
lista_materiais = ["Todos"] + sorted(df_app["Numero Material"].unique().astype(str).tolist())

st.sidebar.image("PRIO_SEM_POLVO_PRIO_PANTONE_LOGOTIPO_Azul.png")

with st.sidebar.expander("Pedidos"):
    pedidos_colados = st.text_area("Cole aqui a lista de pedidos: ", height=100)
    
    lista_pedidos_colados = []
    if pedidos_colados:
        lista_pedidos_colados = re.split(r"[,\n;\s]+", pedidos_colados)
        lista_pedidos_colados = [p.strip() for p in lista_pedidos_colados if p.strip() in lista_pedidos]
    
    pedidos_select = st.multiselect(
        "Ou selecione manualmente:",
        options=lista_pedidos,
        default=lista_pedidos_colados if lista_pedidos_colados else "Todos"
    )
    
with st.sidebar.expander("Fornecedor"):
    fornecedores_select = st.multiselect("Selecione:",lista_fornecedores,default="Todos")

with st.sidebar.expander("Material"):
    materiais_colados = st.text_area("Cole aqui a lista de materiais: ", height=100)
    
    lista_materiais_colados = []
    if materiais_colados:
        lista_materiais_colados = re.split(r"[,\n;\s]+", materiais_colados)
        lista_materiais_colados = [p.strip() for p in lista_materiais_colados if p.strip() in lista_materiais]
    
    materiais_select = st.multiselect(
        "Ou selecione manualmente:",
        options=lista_materiais,
        default=lista_materiais_colados if lista_materiais_colados else "Todos"
    )

with st.sidebar.expander("Area"):
    areas_select = st.multiselect("Selecione:",lista_areas,default="Todas")
    
with st.sidebar.expander("Aprovador"):
    aprovador_select = st.multiselect("Selecione:",lista_aprovadores,default="Todos")
    
with st.sidebar.expander("Ano"):
    anos_select = st.multiselect("Selecione:",lista_anos,default="2025")

with st.sidebar.expander("Mes"):
    meses_select = st.multiselect("Selecione:",lista_meses,default="Todos")

check_duplicidade = st.sidebar.toggle("Verificar Duplicidade")
check_sod = st.sidebar.toggle("Verificar SoD")
check_aprovador = st.sidebar.toggle("Verificar Alçada")
check_valor = st.sidebar.toggle("Verificar Valor")

if "Todos" not in pedidos_select:
    df_app = df_app[df_app["Pedido"].isin(pedidos_select)]
    
if "Todos" not in fornecedores_select:
    df_app = df_app[df_app["Nome Fornecedor"].isin(fornecedores_select)]
    
if "Todos" not in materiais_select:
    df_app = df_app[df_app["Numero Material"].isin(materiais_select)]

if "Todas" not in areas_select:
    df_app = df_app[df_app["Nome CC"].isin(areas_select)]

if "Todos" not in aprovador_select:
    df_app = df_app[df_app["Aprovador"].isin(aprovador_select)]
    
if "Todos" not in anos_select:
    df_app = df_app[df_app["Ano"].isin(anos_select)]
    
if "Todos" not in meses_select:
    df_app = df_app[df_app["Mes"].isin(meses_select)]

if check_duplicidade:
    df_app = df_app[df_app["Check Duplicidade"]=="Sim"]

if check_sod:
    df_app = df_app[df_app["Check SoD"]=="Sim"]

if check_aprovador:
    df_app = df_app[df_app["Check Aprovador"]=="Inefetivo"]

if check_valor:
    df_app = df_app[df_app["Check Valor"]=="Sim"]
    

# Criação do Agente com o Chat GPT --------------------------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="🔍 Analisando os pedidos com IA...")
def executar_auditoria(df_agente_filtrado):
    # Função para verificar JSON válido
    def extract_json_objects_from_response(response_content):
        logging.basicConfig(level=logging.INFO)
        json_objects = re.findall(r'\{.*?\}', response_content, re.DOTALL)
        parsed = []

        for obj in json_objects:
            try:
                parsed.append(json.loads(obj))
            except json.JSONDecodeError:
                logging.warning(f"Erro ao decodificar JSON: {obj[:100]}")
        
        return parsed
    
    def is_valid_json(response_content):
        try:
            json.loads(response_content)
            return True
        except ValueError:
            return False

    # Função para limpar delimitadores de código da resposta da API
    def clean_json_response(text):
        if text.startswith("```json") and text.endswith("```"):
            return text[7:-3].strip()
        return text.strip()

    # Função para realizar chamadas à API OpenAI
    def invoke_openai(prompt):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Você é um investigador sênior e experiente que identifica fraudes financeiras."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                verify=False  # ambiente com SSL desabilitado
            )
            response.raise_for_status()
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            return clean_json_response(content)
        except Exception as e:
            st.error(f"Erro na chamada da API OpenAI: {e}")
            return ""

    # Função para processar um chunk de dados
    def process_chunk(chunk):
        # Agrupar por "Numero PO" e consolidar dados antes de enviar ao modelo
        grouped_chunk = chunk.groupby("Pedido", as_index=False).agg({
            "Numero Material": "first",
            "Nome Material": "first",
            "Nome Fornecedor": "first",
            "Quantidade": "first",
            "Valor Item BRL": "first",
            "Valor Linha BRL": "first",
            "Requisitante": "first",
            "Comprador": "first",
            "Aprovador": "first",
            "Média do Material": "first",
            "Check Duplicidade": list,
            "Check SoD": list,
            "Check Aprovador": list,
            "Check Valor": list,
            "Compliance": list
        }).reset_index()

        prompt_template = """
        #Objetivos
        
        Seu objetivo é identificar possíveis fraudes e erros na base de dados de pedidos de compras.
        O futuro da empresa e o emprego de seus funcionários dependem da precisão de suas análises.
        
        #Contexto
        
        Como investigador experiente, considere os seguintes cenários:
        - Pedidos de compra de mesmo valor, competência (mês) e fornecedor, considerando apenas pedidos com números distintos. Utilize o {{"Check Duplicidade"}} como balizador.
        - Pedidos de compra sem a devida alçada de aprovação e segregação de funções. Utilize {{Check Aprovador}} para verificar a alçada de aprovação.
        Utilize {{Checl SoD}} para verificar a segregação de funções.
        - Pedidos de compra sem aprovadores ou com aprovador sistêmico. Aprovadores que começam com FUS_ ou SAP são sistêmicos.
        - Pedidos de compra para serviços de maior risco (Consultoria, Patrocínios, Serviços Financeiros).Utilize também a coluna {{Compliance}} como balizadora.
        - Pedidos com itens acima da média histórica. A coluna Check Valor informa se é aplicável e se está acima da média histórica e a coluna Média do Material
        informa qual a média histórica daquele material. Caso o motivo do seu flag seja o pedido estar acima da média histórica, informe o nome do material
        {{Nome Material}}, o valor unitário praticado {{"Valor Item BRL"}} e a sua média histórica {{Média do Material}}.
        
        #Saída
        Retorne um JSON com:
        [
          {{
            "Pedido": "123456",
            "Nome Fornecedor": "Fornecedor Y",
            "Nome Material": "DISJUNTOR TRIP 426P PPIENK",
            "Quantidade": 10.0;
            "Valor Item BRL": 1,234.56
            "Valor Linha BRL": 12,345.60,
            "Valor PO - R$": 123,456.78,
            "Requisitante": "Pedro Henrique",
            "Comprador": "Larissa Silva",
            "Aprovador": "João Paulo",
            "Motivo": "O material DISJUNTOR TRIP 426P PPIENK historicamente foi adquirido por 123.34 e neste pedido foi adquirido por 234.56."
          }},
          ...
        ]
        
        #Fonte da busca
        Aqui estão os dados: {dados}
        """

        dados_json = grouped_chunk.to_json(orient='records')
        prompt = prompt_template.format(dados=dados_json)

        response = invoke_openai(prompt)
        if response:
            try:
                # Extrair apenas os objetos JSON da resposta
                red_flags = extract_json_objects_from_response(response)
                if isinstance(red_flags, list) and red_flags:
                    print(f"✅ JSONs válidos extraídos: {len(red_flags)} itens.")
                    return red_flags
                else:
                    print("⚠️ Nenhum JSON válido encontrado na resposta.")
                    return []
            except json.JSONDecodeError:
                print("❌ Resposta da API não é um JSON válido.")
                return []
        else:
            print("❌ Resposta inválida ou vazia do modelo.")
            return []

    # Função principal do agente de auditoria
    def auditor(file_path):
        
        required_columns = [
            "Pedido",
            "Numero Material",
            "Nome Material",
            "Nome Fornecedor",
            "Quantidade",
            "Valor Item BRL",
            "Valor Linha BRL",
            "Requisitante",
            "Comprador",
            "Aprovador",
            "Média do Material",
            "Check Duplicidade",
            "Check SoD",
            "Check Aprovador",
            "Check Valor",
            "Compliance"
        ]
        
        df_entrada = file_path
        df_entrada = df_entrada[required_columns]
        if df_entrada is None or df_entrada.empty:
            return None

        # Processar dados em chunks
        chunk_size = 100
        red_flags = []
        for start in range(0, len(df_entrada), chunk_size):
            chunk = df_entrada.iloc[start:start + chunk_size]
            red_flags.extend(process_chunk(chunk))

        # Gerar DataFrame com os resultados
        df_flags = pd.DataFrame(red_flags)
        if df_flags.empty:
            st.info("Nenhuma suspeita encontrada.")
            st.success(f"🔍 {len(df_flags)} casos suspeitos identificados.")
            return None

        # Retornar resultados
        return df_flags
    
    # Remove o bloco `if __name__ == "__main__":`
    file_path = df_agente_filtrado
    df_flags = auditor(file_path)

    if df_flags is not None and not df_flags.empty:
        print("Fraudes identificadas:")
        print(df_flags.to_string(index=False))
    else:
        print("Nenhuma suspeita encontrada.")

    # Retorna a variável para uso fora da função
    
    converte_excel(df_flags)
    
    return df_flags

@st.cache_data
def consultar_openai(prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except Exception as e:
        print("❌ Erro ao consultar a OpenAI:", e)
        return ""

def extrair_resposta(texto):
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        json_matches = re.findall(r'\{[^}]+\}', texto)
        parsed = []
        for match in json_matches:
            try:
                item = json.loads(match)
                parsed.append(item)
            except:
                continue
        return parsed
    
@st.cache_data(show_spinner="🔍 Analisando os pedidos com IA...")
def verificar_fraude_por_po(df_flags, df_agente_filtrado, chunk_size=5):
   
    if df_flags is None or df_flags.empty:
        st.info("⚠️ Nenhuma flag identificada para revisão.")
        return pd.DataFrame()

    resultados = []

    for i in range(0, len(df_flags), chunk_size):
        chunk = df_flags.iloc[i:i + chunk_size]
        dados_completos = []

        for _, row in chunk.iterrows():
            numero_po = row['Pedido']
            motivo_flag = row['Motivo']

            dados_po = df_agente_filtrado[df_agente_filtrado["Pedido"] == numero_po].to_dict(orient='records')
            if dados_po:
                dados_completos.append({
                    "Pedido": numero_po,
                    "Motivo Flag": motivo_flag,
                    "Dados Pedido": json.loads(json.dumps(dados_po[0], default=str))
                })

        if not dados_completos:
            continue

        prompt = f"""
Você é um auditor especialista em fraudes de compras.

Você recebeu os seguintes casos suspeitos. Cada caso tem:
- Um motivo levantado pelo primeiro agente (`Motivo Flag`),
- E os dados completos do pedido (`Dados Pedido`).

Para cada caso:
- Indique se o motivo da flag é **procedente** com base nos dados do pedido.
- Se sim, escreva um **Parecer Revisor** curto, claro e técnico.
- Se não, diga que **não há evidência suficiente** ou que o motivo não é procedente.

Responda **somente com JSON válido**, com estrutura como esta:
[
  {{
    "Pedido": "123456",
    "Procedente": true,
    "Parecer Revisor": "Concordo. A PO está acima do limite e sem aprovador identificado."
  }},
  {{
    "Pedido": "456789",
    "Procedente": false,
    "Parecer Revisor": "Discordo. A PO está dentro do limite e com aprovador identificado."
  }},
  ...
]

Casos:
{json.dumps(dados_completos, ensure_ascii=False)}
        """

        print(f"🔹 Enviando arquivos {i // chunk_size + 1} com {len(dados_completos)} casos")
        resposta = consultar_openai(prompt)
        print("📩 Resposta da IA:")
        print(resposta[:500])

        pareceres = extrair_resposta(resposta)
        for p in pareceres:
            if "Pedido" in p and "Procedente" in p:
                p["Pedido"] = str(p["Pedido"])
                p["Parecer Revisor"] = p.get("Parecer Revisor", "")
                resultados.append(p)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.columns = [str(c).strip().replace("_", " ") for c in df_resultados.columns]

    if "Pedido" not in df_resultados.columns:
        print("⚠️ Nenhum parecer válido foi retornado pela IA.")
        return pd.DataFrame()

    df_flags["Pedido"] = df_flags["Pedido"].astype(str)
    df_resultados["Pedido"] = df_resultados["Pedido"].astype(str)

    df_agente = df_flags.merge(df_resultados.drop_duplicates(), on="Pedido", how="left")

    return df_agente

# -----------------------------
# Bases para Gráficos
# -----------------------------

grafico_fornecedores = df_app.groupby("Nome Fornecedor")["Valor Linha BRL"].agg("sum").reset_index().sort_values(by="Valor Linha BRL", ascending=False)
grafico_aprovadores = df_app.groupby("Aprovador")["Valor Linha BRL"].agg("sum").reset_index().sort_values(by="Valor Linha BRL", ascending=False)
grafico_areas = df_app.groupby("Nome CC")["Valor Linha BRL"].agg("sum").reset_index().sort_values(by="Valor Linha BRL", ascending=False)
grafico_alcada = df_app
grafico_alcada = grafico_alcada.drop_duplicates(subset="Pedido")
grafico_alcada = grafico_alcada.groupby("Check Aprovador")["Pedido"].agg("count").reset_index()
df_inefetivos = df_app[df_app["Check Aprovador"]=="Inefetivo"]

df_agente = df_app[["Pedido","Numero Material", "Nome Material", "Compliance","Quantidade", "Valor Item BRL","Valor Linha BRL", 
                    "Valor Pedido BRL", "Requisitante", "Nome CC", "Comprador", "Aprovador", "Nome Fornecedor", 
                    "Check Duplicidade","Check SoD", "Check Aprovador", "Check Valor","Média do Material", "Ano", "Mes"]]

aba1, aba2, aba3 = st.tabs(["Base de Pedidos", "Gráficos", "Agente"])

with aba1:
    valor = df_app['Valor Linha BRL'].sum()
    valor_formatado = f"R$ {valor:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    st.metric("Valor Total - BRL", valor_formatado)
    st.dataframe(df_app.drop(columns=["Data Pedido2","Ano","Mes"]))
    st.markdown(f'A tabela possui **{df_app.shape[0]}** linhas e **{df_app.shape[1]}** colunas')
    
    # Remover colunas auxiliares
    df_exportar = df_app.drop(columns=["Data Pedido2", "Ano", "Mes"])
    
    st.download_button(
        label="📥 Baixar relatório de pedidos",
        data=gerar_excel_formatado(df_exportar),
        file_name="pedidos_agente.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


with aba2:
    numero_fornecedores = st.number_input("Quantos fornecedores deseja visualizar?", min_value=1, value=5)
    grafico_fornecedores = grafico_fornecedores.nlargest(numero_fornecedores,"Valor Linha BRL")
    grafico_fornecedores["Valor Formatado"] = grafico_fornecedores["Valor Linha BRL"].apply(lambda x: formata_numero(x,"R$"))
    fig = px.bar(
                grafico_fornecedores,
                x="Nome Fornecedor",
                y="Valor Linha BRL",
                text = "Valor Formatado",
                title=f'Distribuição pelo top {numero_fornecedores} fornecedores'
            )
    st.plotly_chart(fig, use_container_width=True)
    
    numero_aprovadores = st.number_input("Quantos aprovadores deseja visualizar?", min_value=1, value=5)
    grafico_aprovadores = grafico_aprovadores.nlargest(numero_aprovadores,"Valor Linha BRL")
    grafico_aprovadores["Valor Formatado"] = grafico_aprovadores["Valor Linha BRL"].apply(lambda x: formata_numero(x,"R$"))
    fig = px.bar(
                grafico_aprovadores,
                x="Aprovador",
                y="Valor Linha BRL",
                text = "Valor Formatado",
                title=f'Distribuição pelo top {numero_aprovadores} aprovadores'
            )
    st.plotly_chart(fig, use_container_width=True)
    
    numero_areas = st.number_input("Quantos áreas deseja visualizar?", min_value=1, value=5)
    grafico_areas = grafico_areas.nlargest(numero_areas,"Valor Linha BRL")
    grafico_areas["Valor Formatado"] = grafico_areas["Valor Linha BRL"].apply(lambda x: formata_numero(x,"R$"))
    fig = px.bar(
                grafico_areas,
                x="Nome CC",
                y="Valor Linha BRL",
                text = "Valor Formatado",
                title=f'Distribuição pelo top {numero_areas} áreas'
            )
    st.plotly_chart(fig, use_container_width=True)
    
with aba3:

    st.title("Análise dos Agentes")

    # 1. Seletor de ano e mês com base no df_agente original
    anos_disponiveis = sorted(df_agente["Ano"].unique()) + ['Todos']
    meses_disponiveis = sorted(df_agente["Mes"].unique()) + ['Todos']
    compradores_disponiveis = sorted(df_agente["Comprador"].unique()) + ['Todos']
    
    ano = st.multiselect("Selecione o Ano", anos_disponiveis, default = 'Todos')
    mes = st.multiselect("Selecione o Mês", meses_disponiveis, default = 'Todos')
    comprador = st.multiselect("Selecione o comprador", compradores_disponiveis, default = 'Todos')
    valor_minimo = st.number_input(
        "Filtrar valor dos pedidos (BRL)", 
        min_value=0, 
        value=100000, 
        step=10000
    )

    df_agente_filtrado = df_agente[df_agente["Valor Pedido BRL"] > valor_minimo].copy()
    
    if 'Todos' not in  ano:
        df_agente_filtrado = df_agente_filtrado[df_agente_filtrado['Ano'].isin(ano)]
    
    if 'Todos' not in mes:
        df_agente_filtrado = df_agente_filtrado[df_agente_filtrado['Mes'].isin(mes)]
        
    if 'Todos' not in comprador:
        df_agente_filtrado = df_agente_filtrado[df_agente_filtrado['Comprador'].isin(comprador)]
        

    # 3. Inicializar estado de controle
    if "executar_analise" not in st.session_state:
        st.session_state.executar_analise = False
    if "executar_agora" not in st.session_state:
        st.session_state.executar_agora = False

    # 4. Botões de controle
    with st.container():
        col_executar, col_resetar = st.columns(2)

        with col_executar:
            if st.button("▶️ Executar Análise do Agente", key="btn_executar"):
                st.session_state.executar_agora = True

        with col_resetar:
            if st.button("🔁 Resetar Análise", key="btn_resetar"):
                st.session_state.executar_agora = False
                st.session_state.executar_analise = False
                st.success("Análise resetada.")

    # 5. Execução somente ao clicar no botão
    if st.session_state.executar_agora and not st.session_state.executar_analise:
        with st.spinner("🔄 Gerando arquivos e analisando..."):

            df_flags = executar_auditoria(df_agente_filtrado)
            df_agente_resultado = verificar_fraude_por_po(df_flags, df_agente_filtrado)

            if df_flags.empty:
                st.info("✅ Nenhum caso com 'Flag: Sim' foi identificado.")
            else:
                st.success(f"Análise concluída. {df_agente_resultado.shape[0]} casos com red flags.")
                st.title("Pedidos flagados pelo Agente")
                st.dataframe(df_agente_resultado)
                st.download_button(
                    label="📥 Baixar análises dos agentes",
                    data=gerar_excel_formatado(df_agente_resultado),
                    file_name="analise_agente.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        # Atualiza flags de estado após execução
        st.session_state.executar_analise = True
        st.session_state.executar_agora = False

    # 6. Gráfico de Alçada
    st.title("Pedidos com problema de alçada")

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df_inefetivos.drop(columns=["Check Aprovador", "Data Pedido2", "Ano", "Mes"]))
        st.markdown(f'A tabela possui **{df_inefetivos.shape[0]}** linhas e **{df_inefetivos.shape[1]}** colunas')
        
        st.download_button(
            label="📥 Baixar relatório de inefetivos",
            data=gerar_excel_formatado(df_inefetivos),
            file_name="base_inefetivos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        fig2 = px.pie(
            grafico_alcada,
            names="Check Aprovador",
            values="Pedido",
            color="Check Aprovador",
            color_discrete_map={"Efetivo": "green", "Inefetivo": "red"},
            hole=0.4
        )
        fig2.update_traces(textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)
        
    # 7. Bases Agente
    st.title("Base de Dados do Agente")
    st.dataframe(df_agente_filtrado)
    st.markdown(f'A tabela possui **{df_agente_filtrado.shape[0]}** linhas e **{df_agente_filtrado.shape[1]}** colunas')
    
    st.download_button(
        label="📥 Baixar relatório do Agente",
        data=gerar_excel_formatado(df_agente_filtrado),
        file_name="base_agente.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
            
