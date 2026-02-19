# Tiara-Agente-IA
Una aplicación web que aprovecha modelos de lenguaje avanzados para traducir preguntas de negocio en consultas SQL prácticas. La aplicación ejecuta estas consultas en una base de datos y visualiza los resultados, lo que ayuda a los usuarios a obtener información de sus datos de forma rápida y eficiente.

## Características
- **Consultas en lenguaje natural:**
transforme preguntas comerciales (por ejemplo, "¿Por qué no alcanzamos nuestro objetivo de margen neto este mes?") en consultas SQL utilizando GPT-4.

- **Integración del esquema de base de datos:**
cargue y almacene en caché automáticamente el esquema de base de datos para garantizar que las consultas SQL hagan referencia únicamente a tablas y columnas válidas.

-- **Ejecución de consultas SQLite**:
ejecuta consultas SQL en una base de datos SQLite (o alternativa) y devuelve los resultados en formato JSON.

-- **Visualización de datos**:
genere gráficos basados ​​en resultados de consultas 

- **Interfaz web:**
una interfaz Flask simple y responsiva permite a los usuarios ingresar sus preguntas, enviarlas y ver los resultados junto con visualizaciones.

## Estructura del Proyecto
├── backend/
│   ├── vanna_chromadb/    # Carpeta donde vivirá el RAG – hay que probar si se genera automático 
│   ├── app.py             # Tu API con FastAPI
│   ├── agent_logic.py     # Lógica de Vanna + ChromaDB
│   ├── database.py        # Conexión a SQL Server
│   └── .env               # La llave de OpenAI
├── frontend/
│   ├── index.html         # 
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── chat.js
├── Dockerfile             # 
└── requirements.txt       # 

