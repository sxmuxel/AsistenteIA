import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuración

KNOWLEDGE_DIR = "knowledge_base"
CHROMA_DIR    = "chroma_db"
COLLECTION    = "soporte_tecnico"
EMBED_MODEL   = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100

# ── Funciones

def load_documents(directory: str) -> list[dict]:
    """
    Carga todos los archivos .txt del directorio indicado.
    Retorna una lista de diccionarios con 'content' y 'source'.
    """
    documents = []
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt en '{directory}'. "
            "Asegúrate de que la carpeta knowledge_base/ contiene los manuales."
        )

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        filename = os.path.basename(filepath)
        documents.append({"content": content, "source": filename})
        print(f"  [✓] Cargado: {filename} ({len(content)} caracteres)")

    return documents


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Divide un texto en chunks de tamaño aproximado, con solapamiento.
    La división se hace preferentemente en saltos de línea dobles (párrafos)
    para preservar la coherencia semántica de cada chunk.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            line_break       = text.rfind("\n",   start, end)

            if paragraph_break != -1 and paragraph_break > start:
                end = paragraph_break
            elif line_break != -1 and line_break > start:
                end = line_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end - overlap > start else end

    return chunks


def build_vector_store(documents: list[dict]) -> int:
    """
    Construye la base de datos vectorial:
      - Genera chunks de cada documento.
      - Crea embeddings con sentence-transformers.
      - Almacena todo en ChromaDB.
    Retorna el número total de chunks almacenados.
    """
    # Inicializar ChromaDB persistente
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # Eliminar colección anterior si existe (para re-indexar limpio)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"\n  [↺] Colección anterior '{COLLECTION}' eliminada para re-indexar.")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}   # Similitud por coseno
    )

    # Cargar modelo de embeddings (descarga automática la primera vez)
    print(f"\n  [→] Cargando modelo de embeddings: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("  [✓] Modelo cargado.\n")

    total_chunks = 0

    for doc in documents:
        chunks = split_into_chunks(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  [→] '{doc['source']}' → {len(chunks)} chunks generados")

        # Generar embeddings para todos los chunks del documento de una sola vez
        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        # Preparar datos para insertar en ChromaDB
        ids        = [f"{doc['source']}_chunk_{i}" for i in range(len(chunks))]
        metadatas  = [{"source": doc["source"], "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids        = ids,
            documents  = chunks,
            embeddings = embeddings,
            metadatas  = metadatas
        )

        total_chunks += len(chunks)

    return total_chunks


# ── Main

def main():
    print("=" * 60)
    print("  SISTEMA RAG - INGESTA DE DOCUMENTOS")
    print("  Asistente de Soporte Técnico Windows")
    print("=" * 60)

    # 1. Cargar documentos
    print(f"\n[1/3] Cargando documentos desde '{KNOWLEDGE_DIR}/'...")
    documents = load_documents(KNOWLEDGE_DIR)
    print(f"\n  Total documentos cargados: {len(documents)}")

    # 2. Crear chunks, embeddings y almacenar en ChromaDB
    print("\n[2/3] Creando chunks y vectorizando...")
    total = build_vector_store(documents)

    # 3. Resumen
    print(f"\n[3/3] Ingesta completada.")
    print(f"  Total de chunks almacenados en ChromaDB: {total}")
    print(f"  Base de datos guardada en: '{CHROMA_DIR}/'")
    print("\n[✓] La base de conocimientos está lista para consultas.")
    print("    Ejecuta 'python main.py' para iniciar el asistente.")
    print("=" * 60)


if __name__ == "__main__":
    main()