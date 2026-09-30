import asyncio
import sys
import os

from langchain_core.messages import SystemMessage 
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# CONFIGURACIÓN DINÁMICA DEL SERVIDOR
ruta_servidor = os.path.join(os.path.dirname(__file__), "server_calculadora.py")

server_params = StdioServerParameters(
    command=sys.executable, 
    args=[ruta_servidor], 
    env=None
)

async def ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)

async def main():
    print(f"Iniciando Cliente MCP")
    print(f"Conectando al servidor en: {ruta_servidor}")

    # Configuramos el LLM
    llm = ChatOllama(
        #model='llama3.1',
        #model='ministral-3:14b',
        #model='llama3-groq-tool-use',
        model='llama3.1:8b-instruct-q4_K_M',
        temperature=0,
        #base_url="http://localhost:11434"
        base_url="http://10.42.69.229:11434"
    )
    
    # Conexión al servidor MCP
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Cargamos herramientas
                tools = await load_mcp_tools(session)
                print(f"Tools cargadas: {[t.name for t in tools]}")

                # Creamos el Agente ReAct
                memory = MemorySaver()
                
                agent_executor = create_react_agent(
                    llm, 
                    tools=tools, 
                    checkpointer=memory
                )

                # Definimos el prompt del sistema aquí
                system_instruction = "Eres un asistente matemático útil. Usa las herramientas disponibles para responder."

                # Bucle de chat
                config = {"configurable": {"thread_id": "session_math_1"}}
                
                while True:
                    try:
                        user_input = await ainput("\nPregunta (o 'salir') >>> ")
                        if user_input.lower() in ["salir", "exit"]:
                            break

                        print("Procesando...")
                        
                        messages = [
                            SystemMessage(content=system_instruction),
                            ("user", user_input)
                        ]
                        
                        async for event in agent_executor.astream(
                            {"messages": messages}, 
                            config=config,
                            stream_mode="values"
                        ):
                            message = event["messages"][-1]
                            
                            # Mostramos uso de herramientas
                            if hasattr(message, "tool_calls") and message.tool_calls:
                                for tc in message.tool_calls:
                                    print(f"   [Agente usa herramienta]: {tc['name']} args={tc['args']}")
                            
                            # Mostramos respuesta final
                            elif message.type == "ai" and not message.tool_calls:
                                print(f"\nRESPUESTA:\n{message.content}")
                                
                    except Exception as inner_e:
                        print(f"Error procesando mensaje: {inner_e}")

    except Exception as e:
        print(f"\nERROR FATAL DE CONEXIÓN O EJECUCIÓN:\n{e}")
        print("\nConsejo: Verifica que server_calculadora.py esté en la misma carpeta.")

if __name__ == "__main__":
    asyncio.run(main())