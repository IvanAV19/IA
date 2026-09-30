import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Gestor_de_Notas")
NOTES_DIR = "/home/bigdata/PRUEBAMCP/mis_notas"

if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)

@mcp.tool()
def listar_notas() -> str:
    """Lista archivos .txt disponibles."""
    try:
        files = [f for f in os.listdir(NOTES_DIR) if f.endswith('.txt')]
        return "\n".join(files) if files else "No hay notas."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def leer_nota(nombre_archivo: str) -> str:
    """Muesta el contenido de una nota."""
    path = os.path.join(NOTES_DIR, nombre_archivo)
    try:
        if not os.path.exists(path):
            return "Error: El archivo no existe."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()
