import time
import os
import hashlib
import sys
from pathlib import Path
from typing import List

# IMPORTACIONES
import chromadb 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import PyPDF2

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# CONFIGURACIÓN
OLLAMA_BASE_URL = "http://10.42.69.229:11434"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
# Ajustamos estas rutas si es necesario (estas son las rutas relativas)
DOCS_DIR = r"pdfs"
DB_PATH = r"chroma_db"

class RAGVectorStore:
    def __init__(self, persist_directory: str):
        print(f"Conectando a Ollama ({OLLAMA_MODEL})...")
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL, 
            base_url=OLLAMA_BASE_URL
        )
        
        print("Inicializando ChromaDB...")
        
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        print("Base de datos vectorial lista.")

    def add_documents(self, documents: List[Document]):
        if documents:
            self.vectorstore.add_documents(documents)
            
    def delete_document_by_source(self, filepath: str):
        try:
            # Recuperamos documentos por metadata
            result = self.vectorstore.get(where={"source": filepath})
            if result['ids']:
                self.vectorstore.delete(ids=result['ids'])
        except Exception:
            pass 

class DocumentIndexer(FileSystemEventHandler):
    def __init__(self, docs_directory: str, vector_store: RAGVectorStore):
        self.docs_directory = Path(docs_directory)
        self.vector_store = vector_store
        self.supported_extensions = ['.pdf']
        self.indexed_files = {} 
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

    def _get_file_hash(self, filepath: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _is_indexed_in_chroma(self, filename: str) -> bool:
        """Consulta ChromaDB para saber si el documento ya fue indexado."""
        result = self.vector_store.vectorstore.get(
            where={"filename": filename},
            include=["metadatas"]
        )
        return len(result["ids"]) > 0
    
    def index_file(self, filepath: Path):
        try:
            if filepath.suffix not in self.supported_extensions or not filepath.exists():
                return

            # Calculamos el hash actual del archivo
            file_hash = self._get_file_hash(filepath)

            # Caché en memoria: mismo proceso, mismo hash -> ya procesado
            if self.indexed_files.get(str(filepath)) == file_hash:
                return

            # Consulta persistente a ChromaDB: ya indexado en sesiones anteriores
            if str(filepath) not in self.indexed_files and self._is_indexed_in_chroma(filepath.name):
                self.indexed_files[str(filepath)] = file_hash  # sincronizamos caché
                return

            print(f"Procesando: {filepath.name}...")
            
            text = ""
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content: text += content + "\n"

            if not text.strip():
                print(f"PDF vacío: {filepath.name}")
                return

            # Limpieza y re-indexado
            self.vector_store.delete_document_by_source(str(filepath))
            
            doc = Document(page_content=text, metadata={"source": str(filepath), "filename": filepath.name})
            chunks = self.text_splitter.split_documents([doc])
            
            self.vector_store.add_documents(chunks)
            self.indexed_files[str(filepath)] = file_hash
            
            print(f"Indexado: {len(chunks)} fragmentos.")
            
        except Exception as e:
            print(f"Error en {filepath.name}: {e}")

    # Eventos Watchdog
    def on_created(self, event):
        if not event.is_directory: self.index_file(Path(event.src_path))
    def on_modified(self, event):
        if not event.is_directory: self.index_file(Path(event.src_path))

def main():
    print("="*50)
    print("INDEXADOR RAG")
    print("="*50)
    
    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
    
    try:
        store = RAGVectorStore(DB_PATH)
        indexer = DocumentIndexer(DOCS_DIR, store)
        
        print("Escaneando carpeta...")
        for f in Path(DOCS_DIR).glob("*.pdf"):
            indexer.index_file(f)

        observer = Observer()
        observer.schedule(indexer, str(DOCS_DIR), recursive=False)
        observer.start()
        
        print(f"\nMonitoreando: {DOCS_DIR}")
        print("Presiona Ctrl+C para salir.")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        observer.stop()
        print("\nDeteniendo...")

if __name__ == "__main__":
    main()