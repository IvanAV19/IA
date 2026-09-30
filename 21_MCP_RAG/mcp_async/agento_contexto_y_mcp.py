import asyncio
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- CONFIGURACIÓN DEL SERVIDOR MCP ---

server_params = StdioServerParameters(
    command=r"C:\\Users\\ivan.alonsovidal\\AppData\\Local\\miniconda3\\envs\\mcp_server\\python.exe",
    args=["E:\CE-BigData & IA\Modelos de IA\mcp_server\mcp_server.py"], # Servidor mcp
    env=None
)
async def ainput(prompt: str) -> str:
    # Esto permite que el loop de asyncio siga corriendo mientras el usuario escribe
    return await asyncio.to_thread(input, prompt)

async def main():
    # Configurar el LLM
    llm = ChatOllama(
            #model='ministral-3:14b',
            #model='llama3-groq-tool-use',
            model='llama3.1:8b-instruct-q4_K_M',
            temperature=0,
            verbose=True,
            #base_url="http://localhost:11434"
            base_url="http://10.42.69.229:11434"
            #base_url="http://192.168.0.30:11434"
            #base_url="http://10.219.114.51:11434"
        )
    
    #  conexión MCP
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            

            # Esto importa automáticamente todas las tools que el servidor ofrece
            mcp_tools = await load_mcp_tools(session)
            

            # Configurar memoria del agente
            memory = MemorySaver()
            system_prompt = "Eres un asistente que usa herramientas MCP para ayudar al usuario. Responde en español."
            
            app = create_react_agent(
                llm, 
                tools=mcp_tools, 
                checkpointer=memory,
                prompt=system_prompt
            )

            # Ejecutar consulta de prueba
            pregunta = "¿Qué herramientas tienes disponibles en el servidor MCP?"
            while True:
                # 1. Entrada de usuario asíncrona
                pregunta = await ainput("Pregunta>>> ")
                
                if pregunta.lower() in ["salir", "exit", "quit"]:
                    print("Chat terminado")
                    break
                config = {"configurable": {"thread_id": "mcp_user_1"}}
                
                print(f"--- Consulta: {pregunta} ---")
                
                inputs = {"messages": [("user", pregunta)]}
                
                # En entorno async usamos astream
                async for s in app.astream(inputs, config=config, stream_mode="values"):
                    message = s["messages"][-1]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"   [MCP: Llamando a {message.tool_calls[0]['name']}...]")
                
                print(f"\nRESPUESTA FINAL:\n{message.content}")

if __name__ == "__main__":
    asyncio.run(main())