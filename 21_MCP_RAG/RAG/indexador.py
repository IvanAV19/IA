from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

def crear_indice():
    # Define la ruta donde están tus PDFs
    path_carpeta = r"E:\\CE-BigData & IA\\Modelos de IA\\RAG\\pdfs"

    print(f"Cargando archivos PDF desde: {path_carpeta}...")
    
    # Usamos DirectoryLoader para cargar todos los archivos .pdf
    loader = DirectoryLoader(
        path_carpeta, 
        glob="./*.pdf",          # Filtro para cargar solo PDFs
        loader_cls=PyPDFLoader   # Usamos PyPDFLoader para procesar cada archivo encontrado
    )
    
    try:
        docs = loader.load()
        print(f"Se han cargado {len(docs)} páginas de documentos.")
    except Exception as e:
        print(f"Error al cargar documentos: {e}")
        return

    if not docs:
        print("No se encontraron documentos PDF en la carpeta.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.1:8b-instruct-q4_K_M", base_url="http://10.42.69.229:11434")

    print("Creando base de datos vectorial (esto puede tardar)...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        # La ruta donde se guardará la "memoria" del agente
        persist_directory=r"E:\\CE-BigData & IA\\Modelos de IA\\RAG\\chroma_db" 
    )
    print("Base de datos guardada con éxito en 'chroma_db'")

if __name__ == "__main__":
    crear_indice()