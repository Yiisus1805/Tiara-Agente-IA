from __future__ import annotations

import os
import pyodbc
from dotenv import load_dotenv

from backend.schema_store import SchemaVectorStore

load_dotenv()

ODBC = os.getenv("SQLSERVER_ODBC")


# Diccionario semantico
# Cada entrada combina: sinónimos en español, nombres de columnas clave y
# patrones de preguntas naturales para mejorar la recuperación semántica.

SPANISH_ALIASES = {

    # ── TABLAS FACT (métricas / transacciones) ──────────────────────────────

    "FactInternetSales": (
        "ventas internet ventas online pedidos compras canal directo "
        "ventas totales total de ventas ingresos totales cuanto se vendio en total "
        "SalesAmount OrderQuantity TotalProductCost TaxAmt Freight UnitPrice "
        "ingresos monto total facturación revenue "
        "unidades vendidas importe total productos vendidos "
        "cuantos pedidos top ventas ranking ventas mejor cliente "
        "tendencia de ventas ventas por año crecimiento de ventas "
        "cuanto ingreso generaron las ventas online"
    ),
    "FactResellerSales": (
        "ventas reseller revendedor distribuidor canal indirecto canal distribucion "
        "ventas totales total de ventas ingresos totales cuanto se vendio en total "
        "SalesAmount OrderQuantity TotalProductCost UnitPrice EmployeeKey ResellerKey "
        "ingresos monto total facturación ventas por distribuidor "
        "tendencia de ventas ventas por año crecimiento de ventas "
        "cuanto vendio el reseller ranking distribuidores "
        "ventas a traves de socios comerciales "
        "vendedores empleados representantes de ventas fuerza de ventas "
        "quien vendio mas mejor vendedor desempeno del vendedor ventas por empleado "
        "FactResellerSales tiene EmployeeKey para unirse a DimEmployee"
    ),
    "FactSalesQuota": (
        "cuota presupuesto objetivo meta target SalesAmountQuota CalendarYear "
        "objetivo de ventas meta de ventas quota cumplimiento de meta "
        "cuanto se esperaba vender proyeccion de ventas "
        "vs objetivo vendedor alcanzó su meta"
    ),
    "FactProductInventory": (
        "inventario stock existencias almacen bodega "
        "UnitsBalance MovementDate UnitCost OnHandQuantity "
        "inventario disponible stock disponible productos en bodega "
        "cuantas unidades hay disponibles de un producto "
        "nivel de inventario rotacion de inventario"
    ),
    "FactFinance": (
        "finanzas contabilidad cifras financieras Amount AccountKey "
        "FinanceKey OrganizationKey DepartmentGroupKey ScenarioKey "
        "balance estado de resultados presupuesto financiero "
        "datos financieros por departamento por escenario"
    ),
    "FactCallCenter": (
        "call center servicio al cliente llamadas operadores "
        "WagesAmount LaborHours Calls OrdersPerOperator ServiceGrade "
        "automaticResponses nivel de servicio horas laborales "
        "eficiencia del call center atencion al cliente"
    ),
    "FactCurrencyRate": (
        "tipo de cambio moneda tasa conversion divisa "
        "AverageRate EndOfDayRate CurrencyKey DateKey "
        "cambio de divisas conversion de moneda dolar euro peso "
        "cuanto valia la moneda en una fecha"
    ),
    "FactInternetSalesReason": (
        "razones motivos causa ventas internet SalesReasonKey "
        "por que compraron motivo de compra razon de venta online "
        "factores que influyeron en la venta"
    ),
    "FactSurveyResponse": (
        "encuestas respuestas satisfaccion cliente "
        "SurveyResponseKey DateKey CustomerKey "
        "retroalimentacion feedback opinion del cliente "
        "que piensa el cliente nivel de satisfaccion"
    ),
    "FactAdditionalInternationalProductDescription": (
        "descripcion internacional producto idioma CultureID "
        "ProductDescription descripcion del producto en otros idiomas "
        "traduccion nombre producto internacional"
    ),
    "FactResellerSalesXL_CCI": (
        "ventas reseller copia XL CCI tabla auxiliar no usar en produccion"
    ),
    "FactResellerSalesXL_PageCompressed": (
        "ventas reseller copia XL comprimida tabla auxiliar no usar en produccion"
    ),
    "NewFactCurrencyRate": (
        "tipo de cambio actualizado nueva tabla moneda AverageRate EndOfDayRate"
    ),

    # ── TABLAS DIMENSION (atributos / descriptores) ─────────────────────────

    "DimProduct": (
        "productos articulos items catalogo de productos SKU codigo "
        "ProductName EnglishProductName SpanishProductName FrenchProductName "
        "StandardCost ListPrice Color Size Weight Class Style FinishedGoodsFlag "
        "nombre producto precio costo lista de precios color talla peso "
        "que productos existen cuales son los productos disponibles "
        "informacion del articulo detalle del producto"
    ),
    "DimProductCategory": (
        "categorias principales de productos clasificacion primer nivel "
        "EnglishProductCategoryName SpanishProductCategoryName FrenchProductCategoryName "
        "a que categoria pertenece el producto agrupacion de productos "
        "categoria bicicletas ropa accesorios componentes"
    ),
    "DimProductSubcategory": (
        "subcategorias de productos segundo nivel clasificacion "
        "EnglishProductSubcategoryName SpanishProductSubcategoryName FrenchProductSubcategoryName "
        "ProductCategoryKey subdivision de categoria "
        "a que subcategoria pertenece el producto tipo de articulo"
    ),
    "DimCustomer": (
        "clientes compradores consumidores base de clientes "
        "CustomerKey FirstName LastName EmailAddress Phone BirthDate Gender "
        "MaritalStatus YearlyIncome TotalChildren EnglishEducation "
        "nombre cliente apellido correo telefono edad sexo ingresos "
        "informacion del cliente datos demograficos "
        "quien compro cuales son los clientes mejores clientes top clientes "
        "perfil del comprador cliente mas valioso"
    ),
    "DimDate": (
        "fechas calendario tiempo periodo dimension de tiempo "
        "CalendarYear CalendarQuarter CalendarSemester MonthNumberOfYear "
        "FiscalYear FiscalQuarter FiscalSemester "
        "FullDateAlternateKey DayNumberOfMonth DayNumberOfWeek "
        "EnglishDayNameOfWeek EnglishMonthName SpanishMonthName "
        "año mes dia semana trimestre semestre "
        "cuando durante en el año en el mes en el trimestre "
        "fecha de la venta periodo rango de fechas año fiscal "
        "comparar periodos mes anterior año anterior"
    ),
    "DimSalesTerritory": (
        "territorios de ventas regiones paises zona geografica de ventas "
        "SalesTerritoryRegion SalesTerritoryCountry SalesTerritoryGroup "
        "donde se vendio en que region en que pais "
        "australia canada united states estados unidos germany alemania "
        "france francia united kingdom reino unido japan japon "
        "north america europe pacific norteamerica europa pacifico "
        "ventas por region ventas por pais desempeno territorial "
        "filtrar ventas por pais filtrar ventas por region "
        "FactInternetSales y FactResellerSales se unen a esta tabla directamente via SalesTerritoryKey"
    ),
    "DimEmployee": (
        "empleados vendedores personal fuerza de ventas colaboradores "
        "EmployeeKey FirstName LastName Title EmailAddress Phone "
        "HireDate BirthDate Gender MaritalStatus SalesPersonFlag "
        "nombre empleado apellido cargo titulo puesto fecha contratacion "
        "vendedor representante de ventas quien vendio "
        "mejor vendedor ranking de vendedores desempeno del empleado"
    ),
    "DimReseller": (
        "reseller revendedor distribuidor socio de canal partner "
        "ResellerKey ResellerName Phone BusinessType ResellerAlternateKey "
        "NumberEmployees YearOpened AnnualRevenue AnnualSales "
        "nombre reseller tipo de negocio distribuidor autorizado "
        "quienes son los distribuidores mejores distribuidores "
        "tamaño del distribuidor revenue del distribuidor"
    ),
    "DimGeography": (
        "geografia ubicacion ciudad estado provincia codigo postal "
        "GeographyKey City StateProvinceName PostalCode "
        "donde vive el cliente direccion lugar de residencia "
        "en que ciudad en que estado en que provincia "
        "ubicacion del cliente datos de residencia del cliente "
        "clientes por ciudad clientes por estado clientes por codigo postal "
        "NO usar para filtrar ventas por pais o region — usar DimSalesTerritory para eso"
    ),
    "DimPromotion": (
        "promociones descuentos ofertas campañas marketing "
        "PromotionKey PromotionName PromotionAlternateKey "
        "DiscountPct PromotionType PromotionCategory "
        "EnglishPromotionName SpanishPromotionName "
        "StartDate EndDate MinQty MaxQty "
        "descuento aplicado tipo de promocion porcentaje descuento "
        "ventas con descuento cuanto descuento se aplico "
        "efecto de la promocion en ventas"
    ),
    "DimCurrency": (
        "moneda divisa tipo de moneda "
        "CurrencyKey CurrencyAlternateKey CurrencyName "
        "dolar USD euro EUR peso nombre de la moneda "
        "en que moneda se realizaron las ventas conversion"
    ),
    "DimAccount": (
        "cuentas contables plan de cuentas contabilidad "
        "AccountKey AccountAlternateKey AccountDescription AccountType "
        "Operator CustomMembers ValueType "
        "tipo de cuenta descripcion contable activo pasivo ingreso gasto"
    ),
    "DimDepartmentGroup": (
        "departamentos grupos organizacionales estructura "
        "DepartmentGroupKey DepartmentGroupName ParentDepartmentGroupKey "
        "area departamento grupo jerarquia departamental "
        "a que departamento pertenece organigrama"
    ),
    "DimOrganization": (
        "organizacion empresa estructura corporativa "
        "OrganizationKey OrganizationName PercentageOwnership CurrencyKey "
        "nombre empresa subsidiaria porcentaje de propiedad "
        "estructura organizacional grupo empresarial"
    ),
    "DimScenario": (
        "escenarios presupuesto proyeccion planeacion financiera "
        "ScenarioKey ScenarioName "
        "escenario real actual vs presupuesto vs proyeccion "
        "comparacion de escenarios financieros"
    ),
    "ProspectiveBuyer": (
        "compradores potenciales prospectos clientes futuros leads "
        "ProspectiveBuyerKey FirstName LastName EmailAddress BirthDate "
        "Gender MaritalStatus YearlyIncome TotalChildren "
        "posibles clientes base de prospectos candidatos a comprar "
        "perfil del prospecto campañas de adquisicion"
    ),
    "DatabaseLog": (
        "log auditoria historial de cambios base de datos "
        "DatabaseLogID PostTime DatabaseUser Event Schema Object TSQL XmlEvent "
        "quien hizo que cambio cuando registro de actividad "
        "trazabilidad auditoria de base de datos"
    ),
    "AdventureWorksDWBuildVersion": (
        "version build metadata sistema "
        "DBVersion VersionDate "
        "version de la base de datos informacion del sistema build actual"
    ),
    "sysdiagrams": (
        "diagramas sistema esquema visual metadata "
        "name principal diagram definition "
        "diagrama entidad relacion ER estructura visual"
    ),
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

_NUMERIC_TYPES = {
    "int", "bigint", "smallint", "tinyint",
    "decimal", "numeric", "float", "real",
    "money", "smallmoney",
}

_DATE_TYPES = {"date", "datetime", "datetime2", "smalldatetime", "datetimeoffset"}

_TEXT_TYPES = {"varchar", "nvarchar", "char", "nchar", "text", "ntext"}


def _classify_columns(columns):
    metrics, dimensions, temporals, keys = [], [], [], []
    for name, typ, *_ in columns:
        t = typ.lower()
        n = name.lower()
        if "key" in n or n.endswith("id"):
            keys.append(name)
        elif t in _NUMERIC_TYPES:
            metrics.append(name)
        elif t in _DATE_TYPES:
            temporals.append(name)
        elif t in _TEXT_TYPES:
            dimensions.append(name)
    return metrics, dimensions, temporals


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

    metrics, dimensions, temporals = _classify_columns(columns)

    metric_hint = ""
    if table_type == "Fact" and metrics:
        metric_hint = (
            f"\nMétricas agregables (SUM / AVG / COUNT / STDEV):\n"
            f"{', '.join(metrics)}\n"
            f"\nAtributos para GROUP BY / filtros:\n"
            f"{', '.join(dimensions) if dimensions else 'ver columnas *Key para JOIN'}\n"
            f"\nColumnas temporales (filtrar / agrupar por fecha):\n"
            f"{', '.join(temporals) if temporals else 'usar DateKey → JOIN con DimDate'}\n"
        )
    elif table_type == "Dimension" and (dimensions or temporals):
        metric_hint = (
            f"\nAtributos textuales (GROUP BY / filtros / etiquetas):\n"
            f"{', '.join(dimensions)}\n"
        )
        if temporals:
            metric_hint += f"\nColumnas de fecha:\n{', '.join(temporals)}\n"

    doc = f"""
Tabla: {schema}.{table}

Tipo de tabla: {table_type}

Conceptos relacionados: {aliases}

Columnas:
{cols}
{metric_hint}
Primary Key:
{pk_text}

Relaciones (esta tabla referencia a):
{rels}

Uso:
- Tablas Fact contienen métricas (ventas, cantidades, montos): usar SUM, AVG, COUNT, STDEV, RANK.
- Tablas Dim contienen atributos descriptivos: usar para GROUP BY, etiquetas y filtros.
- Para estadísticas por grupo: JOIN Fact + Dim → GROUP BY columna Dim → AGG(métrica Fact).
- Para tendencias temporales: JOIN con DimDate → GROUP BY CalendarYear / CalendarQuarter.
- Para rankings: ORDER BY AGG DESC con TOP N, o usar RANK() OVER (ORDER BY AGG DESC).
- Para crecimiento YoY: subconsulta o LAG() OVER (ORDER BY CalendarYear).
""".strip()

    return doc

# ALL FOREIGN KEYS (single query for join path indexing)

def fetch_all_fk_relations(cursor):
    cursor.execute("""
    SELECT
        ps.name, pt.name, pc.name,
        rs.name, rt.name, rc.name
    FROM sys.foreign_key_columns fk
    JOIN sys.tables  pt ON fk.parent_object_id     = pt.object_id
    JOIN sys.columns pc ON fk.parent_object_id     = pc.object_id
                       AND fk.parent_column_id     = pc.column_id
    JOIN sys.tables  rt ON fk.referenced_object_id = rt.object_id
    JOIN sys.columns rc ON fk.referenced_object_id = rc.object_id
                       AND fk.referenced_column_id = rc.column_id
    JOIN sys.schemas ps ON pt.schema_id = ps.schema_id
    JOIN sys.schemas rs ON rt.schema_id = rs.schema_id
    ORDER BY pt.name, pc.name
    """)
    return cursor.fetchall()


# JOIN PATH DOCUMENTS

def build_join_path_docs(all_fks):
    """Genera documentos de join paths directos y cadenas de 2 saltos."""
    docs = []

    # Índice: from_table -> [(from_col, to_schema, to_table, to_col)]
    fk_index: dict = {}
    for p_schema, p_table, p_col, r_schema, r_table, r_col in all_fks:
        fk_index.setdefault(p_table, []).append((p_col, r_schema, r_table, r_col))

    # Paths directos (1 salto)
    for p_schema, p_table, p_col, r_schema, r_table, r_col in all_fks:
        doc_id = f"rel:{p_table}.{p_col}:{r_table}.{r_col}"
        p_ctx = SPANISH_ALIASES.get(p_table, "")[:200]
        r_ctx = SPANISH_ALIASES.get(r_table, "")[:200]
        doc = (
            f"JOIN: {p_schema}.{p_table}.{p_col} → {r_schema}.{r_table}.{r_col}\n"
            f"SQL: JOIN {r_schema}.{r_table} ON {p_table}.{p_col} = {r_table}.{r_col}\n"
            f"Contexto {p_table}: {p_ctx}\n"
            f"Contexto {r_table}: {r_ctx}"
        )
        docs.append({
            "id": doc_id,
            "doc": doc,
            "meta": {"type": "join_path", "from_table": p_table, "to_table": r_table},
        })

    # Cadenas de 2 saltos (A → B → C)
    for p_schema, p_table, p_col, r_schema, r_table, r_col in all_fks:
        for r2_col, r2_schema, r2_table, r2_rcol in fk_index.get(r_table, []):
            if r2_table == p_table:
                continue  # evitar ciclos
            doc_id = f"chain:{p_table}.{p_col}:{r_table}.{r2_col}:{r2_table}"
            p_ctx  = SPANISH_ALIASES.get(p_table,  "")[:150]
            r_ctx  = SPANISH_ALIASES.get(r_table,  "")[:150]
            r2_ctx = SPANISH_ALIASES.get(r2_table, "")[:150]
            doc = (
                f"JOIN CHAIN: {p_table} → {r_table} → {r2_table}\n"
                f"Para acceder a {r2_table} desde {p_table}:\n"
                f"  JOIN {r_schema}.{r_table} ON {p_table}.{p_col} = {r_table}.{r_col}\n"
                f"  JOIN {r2_schema}.{r2_table} ON {r_table}.{r2_col} = {r2_table}.{r2_rcol}\n"
                f"Contexto {p_table}: {p_ctx}\n"
                f"Contexto {r_table}: {r_ctx}\n"
                f"Contexto {r2_table}: {r2_ctx}"
            )
            docs.append({
                "id": doc_id,
                "doc": doc,
                "meta": {
                    "type": "join_chain",
                    "from_table": p_table,
                    "via_table": r_table,
                    "to_table": r2_table,
                },
            })

    return docs


def ingest_join_paths(target_store, cursor):
    """Indexa relaciones FK como documentos independientes en el schema store."""
    import logging
    logger = logging.getLogger(__name__)

    all_fks = fetch_all_fk_relations(cursor)
    docs = build_join_path_docs(all_fks)

    for d in docs:
        target_store.upsert(
            ids=[d["id"]],
            documents=[d["doc"]],
            metadatas=[d["meta"]],
        )

    logger.info("Join paths indexados: %d", len(docs))
    print(f"Join paths indexados: {len(docs)}")
    return len(docs)


# INGEST

def ingest(target_store=None):
    import logging
    logger = logging.getLogger(__name__)

    if target_store is None:
        target_store = SchemaVectorStore(
            persist_dir="backend/vanna_chromadb/schema_store",
            collection_name=os.getenv("SCHEMA_COLLECTION") or "tiara_schema",
            embedding_mode="default",
        )

    conn = get_connection()
    cursor = conn.cursor()

    tables = fetch_tables(cursor)

    logger.info("Tablas encontradas: %d", len(tables))
    print(f"\nTablas encontradas: {len(tables)}\n")

    count = 0

    for schema, table in tables:

        columns = fetch_columns(cursor, schema, table)
        pks = fetch_primary_keys(cursor, schema, table)
        relations = fetch_fk_relations(cursor, schema, table)

        doc = build_schema_doc(schema, table, columns, pks, relations)

        target_store.upsert(
            ids=[f"{schema}.{table}"],
            documents=[doc],
            metadatas=[{"schema": schema, "table": table}],
        )

        logger.info("Indexed %s.%s", schema, table)
        print(f"Indexed {schema}.{table}")
        count += 1

    logger.info("Total tablas indexadas: %d", count)
    print(f"\nTotal tablas indexadas: {count}")

    # Ingestar join paths como documentos independientes
    print("\nIndexando join paths y cadenas FK...\n")
    ingest_join_paths(target_store, cursor)

    conn.close()

# MAIN

if __name__ == "__main__":

    print("\nIngestando esquema SQL → Chroma\n")

    ingest()

    print("\nSchema ingest terminado\n")