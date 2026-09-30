from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# FUNCIÓN NUEVA PARA VER LO QUE RECUPERA
def inspeccionar_documentos(docs):
    print(f"\n--- [DEBUG] Se encontraron {len(docs)} fragmentos relevantes ---")
    for i, doc in enumerate(docs):
        # Imprime el nombre del archivo y los primeros 100 caracteres del contenido
        source = doc.metadata.get('source', 'Desconocido')
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   Fragmento {i+1} del archivo '{os.path.basename(source)}':")
        print(f"   \"{content_preview}...\"\n")
    
    # Es obligatorio devolver los docs para que la cadena continúe
    return docs

def chat():
    embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M", base_url="http://10.42.69.229:11434")

    print("Cargando base de datos vectorial...")
    
    # Ruta con la r"" para evitar errores de Windows
    path_db = r"E:\CE-BigData & IA\Modelos de IA\RAG\chroma_db"
    
    vectorstore = Chroma(
        persist_directory=path_db, 
        embedding_function=embeddings
    )
    
    # k=3 significa que recuperará los 3 trozos más parecidos
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0, base_url="http://10.42.69.229:11434")

    template = """Responde a la pregunta basándote ÚNICAMENTE en el siguiente contexto:
    {context}
    
    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Inyectamos la función 'inspeccionar_documentos' después del retriever
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            # Recupera docs -> Los imprime en consola -> Los formatea a texto
            "context": retriever | inspeccionar_documentos | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt 
        | llm 
        | StrOutputParser()
    )

    print("Agente listo. Escribe 'salir' para terminar.")
    
    while True:
        pregunta = input("\nPregunta: ")
        if pregunta.lower() in ['salir', 'quit']: break
        
        print("\nGenerando respuesta...", end="", flush=True)
        # El print del debug saldrá aquí automáticamente
        
        full_response = ""
        print("\n--- Respuesta del LLM ---")
        for chunk in rag_chain.stream(pregunta):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")

if __name__ == "__main__":
    chat()