#!/bin/bash
# ── entrypoint.sh ────────────────────────────────────────────────────

set -e

DB_NAME="AdventureWorks2016_EXT"
BAK_PATH="/var/opt/mssql/backup/AdventureWorks2016_EXT.bak"
SQLCMD="/opt/mssql-tools18/bin/sqlcmd"
MAX_WAIT=90
WAITED=0

/opt/mssql/bin/sqlservr &
SQL_PID=$!

echo "⏳ Esperando que SQL Server acepte conexiones..."

until $SQLCMD -S localhost -U sa -P "$SA_PASSWORD" -No -Q "SELECT 1" &>/dev/null; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "❌ SQL Server no respondió en ${MAX_WAIT}s."
        exit 1
    fi
    echo "   ... intento en ${WAITED}s"
    sleep 3
    WAITED=$((WAITED + 3))
done

echo "✅ SQL Server listo."

DB_EXISTS=$($SQLCMD -S localhost -U sa -P "$SA_PASSWORD" -No -h -1 -Q \
    "SET NOCOUNT ON; SELECT COUNT(*) FROM sys.databases WHERE name = '${DB_NAME}'" \
    2>/dev/null | tr -d ' \r\n')

if [ "$DB_EXISTS" = "1" ]; then
    echo "📦 Base de datos '${DB_NAME}' ya existe — omitiendo restore."
else
    if [ ! -f "$BAK_PATH" ]; then
        echo "❌ Archivo de backup no encontrado en $BAK_PATH"
        echo "   Coloca el archivo AdventureWorks2016_EXT.bak en la carpeta db-backup/"
    else
        echo "🔄 Restaurando '${DB_NAME}' desde $BAK_PATH ..."
        $SQLCMD -S localhost -U sa -P "$SA_PASSWORD" -No -Q "
RESTORE DATABASE [${DB_NAME}]
FROM DISK = '${BAK_PATH}'
WITH
    MOVE '${DB_NAME}_Data' TO '/var/opt/mssql/data/${DB_NAME}.mdf',
    MOVE '${DB_NAME}_Log'  TO '/var/opt/mssql/data/${DB_NAME}_log.ldf',
    MOVE '${DB_NAME}_mod'  TO '/var/opt/mssql/data/${DB_NAME}_mod',
    REPLACE, RECOVERY, STATS = 10
"
        echo "✅ Restore completado."
    fi
fi

wait $SQL_PID