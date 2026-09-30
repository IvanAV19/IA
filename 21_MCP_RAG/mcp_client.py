import asyncio
import sys
import json
from pathlib import Path

# LangChain + Ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

# Adaptador MCP con LangChain
from langchain_mcp_adapters.client import MultiServerMCPClient

# CONFIGURACIÓN
# OLLAMA_URL  = "http://10.42.69.229:11434"
OLLAMA_URL = "http://10.42.69.253:11434"
MODEL_NAME  = "llama3.1:8b-instruct-q4_K_M"

# Ruta absoluta al servidor MCP
SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")

# Prompt de sistema con cláusulas de obligatoriedad
SYSTEM_PROMPT = """\
Eres ResearchAssist, un asistente experto en Inteligencia Artificial y Machine Learning.
Tienes acceso a papers de investigación indexados y a herramientas de análisis.
Responde SIEMPRE en español.

REGLAS OBLIGATORIAS:
1. Para cualquier pregunta técnica sobre IA/ML, DEBES usar 'search_documentation'
   para fundamentar tu respuesta con los papers indexados.
2. Si el usuario pregunta qué papers o documentos tienes disponibles, usa
   'list_available_documents'.
3. Si necesitas el contenido completo de un paper, usa 'get_full_document'.
4. Si el usuario pide que expliques un concepto de IA/ML sin buscar en los papers,
   usa 'explain_concept' con el nivel de profundidad adecuado.
5. Si el usuario te pide resumir un texto extenso, usa 'summarize_text'.
6. Si vas a mostrar o demostrar código Python relacionado con IA/ML, DEBES
   verificarlo primero con 'execute_python_code'.
7. Siempre cita el nombre del archivo PDF del que proviene la información de los papers.
"""


# AGENTE
class PyAssistAgent:
    """Agente que se comunica con el servidor MCP via stdio."""

    def __init__(self, tools: list):
        self.llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_URL,
            temperature=0,
        ).bind_tools(tools)

        # Mapa nombre + tool callable para el bucle de razonamiento
        self.tool_map = {t.name: t for t in tools}

        self.messages = [SystemMessage(content=SYSTEM_PROMPT)]

    async def chat(self, user_input: str) -> str:
        self.messages.append(HumanMessage(content=user_input))

        # Bucle ReAct
        while True:
            response = await self.llm.ainvoke(self.messages)
            self.messages.append(response)

            if not response.tool_calls:
                return response.content

            for tc in response.tool_calls:
                tool_name = tc["name"]
                selected  = self.tool_map.get(tool_name)
                print(f"  [MCP] Ejecutando herramienta: {tool_name}...")

                if selected:
                    output = await selected.ainvoke(tc["args"])
                else:
                    output = f"Error: herramienta '{tool_name}' no encontrada."

                self.messages.append(
                    ToolMessage(content=str(output), tool_call_id=tc["id"])
                )


# INTERFAZ DE USUARIO
async def main():
    print("=" * 60)
    print("ResearchAssist — Cliente MCP (Ollama Llama 3.1)")
    print(f"Servidor: {OLLAMA_URL}")
    print("=" * 60)
    print("Conectando al servidor MCP...", end=" ", flush=True)

    # MultiServerMCPClient
    client = MultiServerMCPClient(
        {
            "research_assist_rag": {
                "command": sys.executable,
                "args": [SERVER_SCRIPT],
                "transport": "stdio",
            }
        }
    )
    tools = await client.get_tools()
    print(f"OK ({len(tools)} herramientas disponibles)")
    for t in tools:
        print(f"  · {t.name}")

    agent = PyAssistAgent(tools)
    print("\nAgente listo. Escribe 'salir' para terminar.\n")

    while True:
        try:
            pregunta = input("Tú: ").strip()
            if pregunta.lower() in ("salir", "quit", "exit"):
                break
            if not pregunta:
                continue

            print("RAG-narok_DIVAN pensando...", end="\r", flush=True)
            respuesta = await agent.chat(pregunta)
            print(f"RAG-narok_DIVAN: {respuesta}\n")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")

    print("\n¡Hasta luego!")


if __name__ == "__main__":
    asyncio.run(main())
