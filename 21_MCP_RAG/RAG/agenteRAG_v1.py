from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def chat():
    embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M", base_url="http://10.42.69.229:11434")

    # Carga la base de datos que creó el indexador
    vectorstore = Chroma(
        persist_directory=r"E:\\CE-BigData & IA\\Modelos de IA\\RAG\\chroma_db", 
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever()

    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0, base_url="http://10.42.69.229:11434")

    template = """Responde basándote solo en el contexto: {context}\nPregunta: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    print("Agente RAG iniciado. Escribe 'salir' para terminar.")
    while True:
        pregunta = input("\nPregunta: ")
        if pregunta.lower() in ['salir', 'quit']: break
        
        print("Respuesta: ", end="")
        for chunk in rag_chain.stream(pregunta):
            print(chunk, end="", flush=True)
        print() # Salto de línea al final

if __name__ == "__main__":
    chat()