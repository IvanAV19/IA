# ResearchAssist — Servidor MCP con RAG para Investigación en IA/ML

Sistema que combina un **servidor MCP** (Model Context Protocol) con **RAG** (Retrieval-Augmented Generation) para crear un agente conversacional especializado en Inteligencia Artificial y Machine Learning. El agente puede consultar papers científicos indexados, explicar conceptos, ejecutar código y resumir textos.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    mcp_client.py                    │
│   Agente ResearchAssist  ·  Bucle ReAct async       │
│   (LangChain + ChatOllama + MultiServerMCPClient)   │
└──────────────────────┬──────────────────────────────┘
                       │ stdio (subproceso)
┌──────────────────────▼──────────────────────────────┐
│                    mcp_server.py                    │
│   6 herramientas MCP  ·  FastMCP                    │
│   3 RAG (ChromaDB)  ·  3 no-RAG (LLM directo)      │
└──────────┬────────────────────────────┬─────────────┘
           │                            │
  ┌────────▼────────┐         ┌─────────▼────────┐
  │   ChromaDB      │         │  Ollama (remoto)  │
  │   chroma_db/    │         │  llama3.1:8b      │
  └────────▲────────┘         └──────────────────┘
           │
┌──────────┴──────────────────────────────────────────┐
│              indexador_avanzado.py                  │
│   Indexado inicial + watchdog (monitoreo continuo)  │
│   Lee PDFs  ·  Genera embeddings  ·  Persiste en DB │
└─────────────────────────────────────────────────────┘
```

---

## Archivos del Proyecto

### `indexador_avanzado.py` — Indexador automático de documentos

Proceso **independiente** que gestiona la base de datos vectorial. Debe ejecutarse antes del servidor.

**Funciones principales:**
- **Escaneo inicial** de todos los PDFs existentes en `pdfs/`
- **Monitoreo continuo** mediante `watchdog`: detecta nuevos PDFs añadidos a la carpeta y los indexa automáticamente
- **Deduplicación inteligente**: consulta ChromaDB para no reindexar documentos ya procesados (`_is_indexed_in_chroma`)
- **Caché en memoria** para evitar reprocesado dentro de la misma sesión
- Extrae texto con `PyPDF2`, lo fragmenta con `RecursiveCharacterTextSplitter` (chunks de 1000 chars, solapamiento de 200) y genera embeddings con Ollama

---

### `mcp_server.py` — Servidor MCP con 6 herramientas

Servidor FastMCP que expone herramientas al agente vía protocolo `stdio`. Se lanza automáticamente como subproceso del cliente, **no se ejecuta manualmente**.

#### Herramientas RAG (consultan ChromaDB)

| Herramienta | Descripción |
|---|---|
| `search_documentation(query)` | Búsqueda semántica en los papers indexados. Devuelve los 3 fragmentos más relevantes. |
| `list_available_documents()` | Lista los nombres de todos los PDFs indexados en ChromaDB. |
| `get_full_document(filename)` | Recupera el texto completo de un paper por su nombre de archivo. |

#### Herramientas no-RAG (no consultan ChromaDB)

| Herramienta | Descripción |
|---|---|
| `execute_python_code(code)` | Ejecuta código Python y devuelve la salida de consola. |
| `summarize_text(text, max_words)` | Resume un texto usando el LLM directamente (sin base de datos). |
| `explain_concept(concept, level)` | Explica un concepto de IA/ML con nivel básico/intermedio/avanzado usando el LLM. |

---

### `mcp_client.py` — Cliente y agente conversacional

Implementa **ResearchAssist**, un agente conversacional experto en IA/ML con un bucle ReAct (Reasoning + Acting) completamente asíncrono.

**Componentes:**
- `SYSTEM_PROMPT` con 7 reglas de obligatoriedad que fuerzan el uso de herramientas según el tipo de pregunta
- `ResearchAssistAgent.chat()` — método `async` que implementa el bucle ReAct:
  1. Envía el mensaje al LLM con las herramientas disponibles
  2. Si el LLM solicita una herramienta, la ejecuta con `await tool.ainvoke()`
  3. Añade el resultado al historial y vuelve al paso 1
  4. Cuando no hay más herramientas que llamar, devuelve la respuesta final
- `MultiServerMCPClient` (API `>= 0.1.0`): lanza `mcp_server.py` como subproceso y obtiene las herramientas vía `await client.get_tools()`

---

## Requisitos

```
Python version: 3.12 o 3.13
Servidor Ollama accesible con el modelo llama3.1:8b-instruct-q4_K_M
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## Uso

### Paso 1 — Indexar los documentos (Terminal 1)

```bash
python indexador_avanzado.py
```

Escanea la carpeta `pdfs/`, indexa los PDFs nuevos en ChromaDB y queda en escucha para detectar nuevos archivos. Puede detenerse con `Ctrl+C` una vez finalizado el indexado inicial si no se necesita monitoreo continuo.

> Para añadir nuevos papers: copia el PDF a la carpeta `pdfs/` mientras el indexador está en ejecución. Se procesará automáticamente.

### Paso 2 — Iniciar el agente (Terminal 2)

```bash
python mcp_client.py
```

Lanza el servidor MCP internamente y abre la interfaz conversacional. Escribe `salir` para terminar.

---

## Ejemplos de uso

```
Tú: ¿Qué papers tienes disponibles?
Tú: ¿Qué dice la documentación sobre los modelos de lenguaje grandes?
Tú: Explícame qué es LoRA a nivel básico
Tú: Resume este texto: [pegar fragmento de texto]
Tú: Muéstrame el contenido del paper 2602.09678v1.pdf
Tú: Ejecuta este código: print(2 ** 10)
```

---

## Estructura de directorios

```
tarea_MCP_RAG/
├── indexador_avanzado.py   # Indexador + watchdog
├── mcp_server.py           # Servidor MCP (6 herramientas)
├── mcp_client.py           # Cliente / agente ResearchAssist
├── requirements.txt        # Dependencias Python
├── pdfs/                   # Papers PDF a indexar
│   └── *.pdf
└── chroma_db/              # Base de datos vectorial (generada automáticamente)
```

---

## Configuración

La URL del servidor Ollama se configura en `mcp_server.py` y `mcp_client.py` e `indexador_avanzado.py`:

```python
OLLAMA_BASE_URL = "http://http://10.42.69.253:11434"
OLLAMA_MODEL    = "llama3.1:8b-instruct-q4_K_M"
```
