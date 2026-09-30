import os
import sys
import io

# Base de datos vectorial y embeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

# Servidor MCP
from mcp.server.fastmcp import FastMCP

# CONFIGURACIÓN
# OLLAMA_BASE_URL = "http://10.42.69.229:11434"
OLLAMA_BASE_URL = "http://10.42.69.253:11434"
OLLAMA_MODEL    = "llama3.1:8b-instruct-q4_K_M"
DB_PATH         = r"chroma_db"

# INICIALIZACIÓN DEL ALMACÉN VECTORIAL
embeddings  = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = Chroma(
    collection_name="rag_collection",
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# LLM para herramientas no-RAG (sin acceso a ChromaDB)
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# SERVIDOR MCP
mcp = FastMCP("ResearchAssist-RAG-Server")


# Herramientas RAG

# Herramienta 1: Búsqueda semántica en papers
@mcp.tool()
def search_documentation(query: str) -> str:
    """Busca información relevante en los papers de investigación indexados en ChromaDB.
    Úsala para responder preguntas sobre modelos de IA, técnicas de ML o resultados de investigación.

    Args:
        query: Término o pregunta a buscar en los papers.
    """
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No se encontró información relevante en los papers indexados."
    resultados = []
    for d in docs:
        fuente = os.path.basename(d.metadata.get("source", "desconocido"))
        resultados.append(f"[Fragmento de: {fuente}]\n{d.page_content}")
    return "\n\n---\n\n".join(resultados)


# Herramienta 2: Listar papers disponibles
@mcp.tool()
def list_available_documents() -> str:
    """Lista todos los papers de investigación indexados en la base de datos vectorial.
    Úsala para saber qué documentación está disponible antes de buscar.
    """
    results = vectorstore.get(include=["metadatas"])
    docs: dict = {}
    for meta in results["metadatas"]:
        fname = meta.get("filename", "Desconocido")
        if fname not in docs:
            docs[fname] = meta
    if not docs:
        return "No hay documentos indexados en la base de datos."
    lineas = [f"- {fname}" for fname in docs]
    return "Papers disponibles:\n" + "\n".join(lineas)


# Herramienta 3: Recuperar paper completo
@mcp.tool()
def get_full_document(filename: str) -> str:
    """Recupera el texto completo de un paper indexado, dado su nombre de archivo.
    Úsala cuando necesites leer el contenido íntegro de un documento específico.

    Args:
        filename: Nombre del archivo PDF (p.ej. '2602.09678v1.pdf').
    """
    results = vectorstore.get(
        where={"filename": filename},
        include=["documents", "metadatas"]
    )
    if not results["ids"]:
        return f"Documento '{filename}' no encontrado en la base de datos."
    pares = sorted(
        zip(
            [m.get("chunk_index", i) for i, m in enumerate(results["metadatas"])],
            results["documents"]
        )
    )
    return "\n\n".join([texto for _, texto in pares])


# Herramientas NO-RAG

# Herramienta 4: Ejecutar código Python
@mcp.tool()
def execute_python_code(code: str) -> str:
    """Ejecuta un fragmento de código Python y devuelve la salida de consola.
    Úsala para verificar o demostrar ejemplos de código relacionados con IA/ML.
    NO accede a la base de datos vectorial.

    Args:
        code: Código Python válido a ejecutar.
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {"__builtins__": __builtins__})  # noqa: S102
        output = buffer.getvalue()
        return output if output else "Código ejecutado con éxito (sin salida de consola)."
    except Exception as exc:
        return f"Error en la ejecución: {exc}"
    finally:
        sys.stdout = old_stdout


# Herramienta 5: Resumir texto
@mcp.tool()
def summarize_text(text: str, max_words: int = 150) -> str:
    """Resume un fragmento de texto largo en un número máximo de palabras.
    Úsala para condensar fragmentos de papers o explicaciones extensas.
    NO consulta la base de datos vectorial; trabaja directamente con el texto recibido.

    Args:
        text: Texto a resumir.
        max_words: Número máximo aproximado de palabras del resumen (por defecto 150).
    """
    prompt = (
        f"Resume el siguiente texto en un máximo de {max_words} palabras, "
        f"en español, conservando los puntos clave:\n\n{text}"
    )
    response = llm.invoke(prompt)
    return response.content


# Herramienta 6: Explicar concepto de IA/ML
@mcp.tool()
def explain_concept(concept: str, level: str = "intermedio") -> str:
    """Genera una explicación clara de un concepto de Inteligencia Artificial o Machine Learning.
    NO usa la base de datos vectorial; se basa en el conocimiento general del modelo.
    Útil para entender conceptos antes de profundizar en los papers.

    Args:
        concept: Nombre del concepto a explicar (p.ej. 'attention mechanism', 'LoRA', 'RAG').
        level: Nivel de profundidad: 'básico', 'intermedio' o 'avanzado' (por defecto 'intermedio').
    """
    prompt = (
        f"Explica el concepto de '{concept}' en el contexto de la Inteligencia Artificial "
        f"y el Machine Learning a un nivel {level}. "
        f"Responde en español, de forma clara y estructurada, con ejemplos si es posible."
    )
    response = llm.invoke(prompt)
    return response.content


# PUNTO DE ENTRADA
if __name__ == "__main__":
    mcp.run(transport="stdio")
