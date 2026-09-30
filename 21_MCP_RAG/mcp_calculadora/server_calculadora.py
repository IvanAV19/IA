from mcp.server.fastmcp import FastMCP

# Inicializamos el servidor FastMCP
mcp = FastMCP("Servidor Matemático")

@mcp.tool()
def calcular_potencia(base: float, exponente: float) -> str:
    """
    Calcula la potencia de un número dado una base y un exponente.
    """
    try:
        resultado = base ** exponente
        return f"El resultado de {base} elevado a {exponente} es {resultado}"
    except Exception as e:
        return f"Error matemático: {str(e)}"

@mcp.tool()
def es_numero_primo(numero: int) -> str:
    """
    Verifica si un número entero es primo.
    """
    if numero < 2:
        return f"{numero} no es primo."
    
    for i in range(2, int(numero ** 0.5) + 1):
        if numero % i == 0:
            return f"{numero} NO es primo (es divisible por {i})."
            
    return f"¡Sí! {numero} es un número primo."

if __name__ == "__main__":
    # Inicia el servidor en modo stdio
    mcp.run()