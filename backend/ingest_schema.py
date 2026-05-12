from __future__ import annotations

import os
import pyodbc
from dotenv import load_dotenv

from backend.schema_store import SchemaVectorStore

load_dotenv()

ODBC = os.getenv("SQLSERVER_ODBC")

SCHEMA_DIR = "backend/vanna_chromadb/schema_store"

store = SchemaVectorStore(
    persist_dir=SCHEMA_DIR,
    collection_name="tiara_schema",
    embedding_mode="default",
)


# Diccionario semantico

SPANISH_ALIASES = {
    "FactInternetSales":                        "ventas internet ventas online pedidos clientes SalesAmount OrderQuantity ingresos monto total",
    "FactResellerSales":                        "ventas reseller revendedor distribuidor SalesAmount OrderQuantity ingresos monto total",
    "DimProduct":                               "productos articulos items ProductName nombre producto",
    "DimProductCategory":                       "categorias de productos categoria EnglishProductCategoryName",
    "DimProductSubcategory":                    "subcategorias de productos subcategoria EnglishProductSubcategoryName",
    "DimCustomer":                              "clientes compradores CustomerName nombre cliente",
    "DimDate":                                  "fechas calendario año mes dia year month CalendarYear FullDateAlternateKey fecha",
    "DimSalesTerritory":                        "territorios regiones paises zona geografica SalesTerritoryRegion SalesTerritoryCountry",
    "DimEmployee":                              "empleados vendedores personal EmployeeName nombre empleado",
    "DimReseller":                              "reseller revendedor distribuidor ResellerName nombre reseller",
    "DimGeography":                             "geografia pais ciudad estado Country City StateProvinceName",
    "DimPromotion":                             "promociones descuentos PromotionName DiscountPct oferta",
    "DimCurrency":                              "moneda divisa CurrencyName CurrencyAlternateKey",
    "DimAccount":                               "cuentas contables AccountDescription AccountType",
    "DimDepartmentGroup":                       "departamentos grupos DepartmentGroupName",
    "DimOrganization":                          "organizacion empresa OrganizationName CurrencyKey",
    "DimScenario":                              "escenarios presupuesto ScenarioName",
    "FactSalesQuota":                           "cuota presupuesto objetivo meta ventas SalesAmountQuota",
    "FactProductInventory":                     "inventario stock productos UnitsBalance MovementDate",
    "FactFinance":                              "finanzas contabilidad Amount AccountKey",
    "FactCallCenter":                           "call center llamadas WagesAmount LaborHours",
    "FactCurrencyRate":                         "tipo de cambio moneda tasa AverageRate EndOfDayRate",
    "FactInternetSalesReason":                  "razones motivos ventas internet SalesReasonKey",
    "FactSurveyResponse":                       "encuestas respuestas SurveyResponseKey",
    "FactAdditionalInternationalProductDescription": "descripcion internacional producto ProductDescription",
    "FactResellerSalesXL_CCI":                  "ventas reseller copia XL CCI (no usar en produccion)",
    "FactResellerSalesXL_PageCompressed":       "ventas reseller copia XL comprimida (no usar en produccion)",
    "NewFactCurrencyRate":                      "tipo de cambio nuevo moneda AverageRate",
    "ProspectiveBuyer":                         "compradores potenciales prospectos clientes futuros",
    "DatabaseLog":                              "log base de datos auditoria cambios",
    "AdventureWorksDWBuildVersion":             "version build base de datos",
    "sysdiagrams":                              "diagramas sistema base de datos",
}


# DB CONNECTION

def get_connection():
    if not ODBC:
        raise RuntimeError("SQLSERVER_ODBC no configurado en .env")
    return pyodbc.connect(ODBC)


# TABLE LIST

def fetch_tables(cursor):

    cursor.execute("""
    SELECT
        s.name AS schema_name,
        t.name AS table_name
    FROM sys.tables t
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    ORDER BY s.name, t.name
    """)

    return cursor.fetchall()


# COLUMNS

def fetch_columns(cursor, schema, table):

    cursor.execute("""
    SELECT
        c.name,
        ty.name,
        c.max_length,
        c.is_nullable
    FROM sys.columns c
    JOIN sys.types ty ON c.user_type_id = ty.user_type_id
    JOIN sys.tables t ON c.object_id = t.object_id
    JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = ? AND t.name = ?
    ORDER BY c.column_id
    """, schema, table)

    return cursor.fetchall()


# PRIMARY KEYS

def fetch_primary_keys(cursor, schema, table):

    cursor.execute("""
    SELECT c.name
    FROM sys.indexes i
    JOIN sys.index_columns ic
        ON i.object_id = ic.object_id
        AND i.index_id = ic.index_id
    JOIN sys.columns c
        ON ic.object_id = c.object_id
        AND ic.column_id = c.column_id
    JOIN sys.tables t
        ON i.object_id = t.object_id
    JOIN sys.schemas s
        ON t.schema_id = s.schema_id
    WHERE i.is_primary_key = 1
      AND s.name = ?
      AND t.name = ?
    """, schema, table)

    return [r[0] for r in cursor.fetchall()]


# FOREIGN KEYS

def fetch_fk_relations(cursor, schema, table):

    cursor.execute("""
    SELECT
        parent_col.name,
        ref_schema.name,
        ref_table.name,
        ref_col.name
    FROM sys.foreign_key_columns fk
    JOIN sys.tables parent_table
        ON fk.parent_object_id = parent_table.object_id
    JOIN sys.columns parent_col
        ON fk.parent_object_id = parent_col.object_id
        AND fk.parent_column_id = parent_col.column_id
    JOIN sys.tables ref_table
        ON fk.referenced_object_id = ref_table.object_id
    JOIN sys.columns ref_col
        ON fk.referenced_object_id = ref_col.object_id
        AND fk.referenced_column_id = ref_col.column_id
    JOIN sys.schemas parent_schema
        ON parent_table.schema_id = parent_schema.schema_id
    JOIN sys.schemas ref_schema
        ON ref_table.schema_id = ref_schema.schema_id
    WHERE parent_schema.name = ?
      AND parent_table.name = ?
    """, schema, table)

    return cursor.fetchall()


# Detección de tipo de tabla (FACT / DIM)

def detect_table_type(table_name: str):

    name = table_name.lower()

    if name.startswith("fact"):
        return "Fact"

    if name.startswith("dim"):
        return "Dimension"

    if "bridge" in name:
        return "Bridge"

    return "Table"


# BUILD DOCUMENT

def build_schema_doc(schema, table, columns, pks, relations):

    table_type = detect_table_type(table)
    aliases = SPANISH_ALIASES.get(table, "")

    column_lines = []

    for name, typ, length, nullable in columns:
        null_txt = "NULL" if nullable else "NOT NULL"
        column_lines.append(f"{name} ({typ}) {null_txt}")

    cols = "\n".join(column_lines)
    pk_text = ", ".join(pks) if pks else "None"

    rel_lines = []
    for r in relations:
        rel_lines.append(f"{schema}.{table}.{r[0]} -> {r[1]}.{r[2]}.{r[3]}")
    rels = "\n".join(rel_lines) if rel_lines else "None"

    doc = f"""
Tabla: {schema}.{table}

Tipo de tabla: {table_type}

Conceptos relacionados: {aliases}

Columnas:
{cols}

Primary Key:
{pk_text}

Relaciones (esta tabla referencia a):
{rels}

Uso:
- Tablas Fact contienen métricas (ventas, cantidades, montos).
- Tablas Dim contienen atributos descriptivos.
""".strip()

    return doc

# INGEST

def ingest():

    conn = get_connection()
    cursor = conn.cursor()

    tables = fetch_tables(cursor)

    print(f"\nTablas encontradas: {len(tables)}\n")

    count = 0

    for schema, table in tables:

        columns = fetch_columns(cursor, schema, table)
        pks = fetch_primary_keys(cursor, schema, table)
        relations = fetch_fk_relations(cursor, schema, table)

        doc = build_schema_doc(
            schema,
            table,
            columns,
            pks,
            relations,
        )

        store.upsert(
            ids=[f"{schema}.{table}"],
            documents=[doc],
            metadatas=[{"schema": schema, "table": table}],
        )

        print(f"Indexed {schema}.{table}")
        count += 1

    print(f"\nTotal tablas indexadas: {count}")

# MAIN

if __name__ == "__main__":

    print("\nIngestando esquema SQL → Chroma\n")

    ingest()

    print("\nSchema ingest terminado\n")