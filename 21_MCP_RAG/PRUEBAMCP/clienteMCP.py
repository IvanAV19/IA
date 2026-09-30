import asyncio
import sys
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- CONFIGURACIÓN ---
SERVER_PATH = "/home/bigdata/PRUEBAMCP/mcp_server.py"
PYTHON_EXE = "/home/bigdata/miniconda3/envs/ollama/bin/python"

server_params = StdioServerParameters(
    command=PYTHON_EXE,
    args=[SERVER_PATH],
    env=None
)

async def main():
    print("🔄 Iniciando conexión...")
    
    # 1. Configuración del LLM
    llm = ChatOllama(
        #model='llama3.1:8b-instruct-q4_K_M',
        model='ministral-3:14b',
        temperature=0,
        base_url="http://10.42.69.229:11434",
        num_predict=500
    )

    # 2. Conexión protegida
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ Conexión MCP establecida.")

                # Cargar herramientas
                mcp_tools = await load_mcp_tools(session)
                print(f"🛠️ Herramientas detectadas: {[t.name for t in mcp_tools]}")

                # 3. Configurar el Agente
                memory = MemorySaver()
                app = create_react_agent(
                    llm,
                    tools=mcp_tools,
                    checkpointer=memory
                )

                # 4. Consulta
                pregunta = "¿Qué notas hay? dime el contenido de todas"
                print(f"\n🚀 PROCESANDO: {pregunta}")

                try:
                    inputs = {"messages": [("user", pregunta)]}
                    config = {"configurable": {"thread_id": "sesion_consola"}}
                    
                    # Ejecución del agente
                    result = await app.ainvoke(inputs, config=config)
                    
                    print(f"\n🤖 [RESPUESTA]:\n{result['messages'][-1].content}")
                except Exception as e:
                    print(f"❌ Error durante la ejecución del agente: {e}")

    except Exception as e:
        print(f"❌ Error de conexión/sesión: {e}")

# --- PUNTO DE ENTRADA PARA TERMINAL ---
if __name__ == "__main__":
    print("🎬 Arrancando cliente desde script .py...")
    try:
        # En scripts .py fuera de Jupyter, usamos asyncio.run para gestionar el bucle
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Programa detenido por el usuario.")
        sys.exit(0)