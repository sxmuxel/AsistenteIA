import os
import sys
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

# ── Configuración

CHROMA_DIR   = "chroma_db"
COLLECTION   = "soporte_tecnico"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "phi3"
TOP_K        = 4      # Número de chunks a recuperar por consulta

# ── Inicialización de recursos

def load_text_file(path: str) -> str:
    """Carga un archivo de texto con manejo de errores."""
    if not os.path.exists(path):
        print(f"[ERROR] Archivo no encontrado: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def initialize_retriever():
    if not os.path.exists(CHROMA_DIR):
        print("[ERROR] La base de datos vectorial no existe.")
        print("        Ejecuta primero: python ingest.py")
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collections = [c.name for c in client.list_collections()]
    if COLLECTION not in collections:
        print(f"[ERROR] La colección '{COLLECTION}' no existe en ChromaDB.")
        print("        Ejecuta primero: python ingest.py")
        sys.exit(1)

    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(EMBED_MODEL)

    return collection, model


# ── Núcleo RAG

def retrieve_context(query: str, collection, model, top_k: int) -> tuple[str, list[str]]:
    """
    Recupera los chunks más relevantes de la base de datos vectorial.

    Retorna:
        - context_text (str): texto concatenado de los chunks recuperados.
        - sources (list[str]): lista de fuentes (nombres de archivo).
    """
    # Generar embedding de la consulta del usuario
    query_embedding = model.encode([query]).tolist()[0]

    # Búsqueda de similitud en ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks     = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    # Mostrar información de retrieval en consola (modo debug)
    print("\n  [RAG] Chunks recuperados:")
    sources = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        fuente = meta.get("source", "desconocido")
        sources.append(fuente)
        similitud = round((1 - dist) * 100, 1)
        print(f"    {i+1}. {fuente} | Similitud: {similitud}% | {chunk[:60]}...")

    # Construir el bloque de contexto para el prompt
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        fuente = meta.get("source", "desconocido")
        context_parts.append(f"[Fuente: {fuente}]\n{chunk}")

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources


def build_prompt(few_shot: str, context: str, question: str) -> str:
    """
    Construye el prompt final para el LLM con:
    - Ejemplos few-shot para guiar el formato de salida.
    - Contexto recuperado delimitado con XML tags.
    - Pregunta del usuario delimitada con XML tags.
    """
    return f"""
{few_shot}

<CONTEXTO_RECUPERADO>
{context}
</CONTEXTO_RECUPERADO>

<PREGUNTA_USUARIO>
{question}
</PREGUNTA_USUARIO>
"""


def query_llm(system_prompt: str, user_prompt: str) -> str:
    """Envía el prompt al LLM local a través de Ollama y retorna la respuesta."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"[ERROR] No se pudo conectar con Ollama: {e}\nAsegúrate de que Ollama esté ejecutándose."


# ── Interfaz de usuario

def print_header():
    print("\n" + "=" * 60)
    print("  ASISTENTE DE SOPORTE TÉCNICO WINDOWS  |  RAG + IA Local")
    print("  Base de conocimientos: Manuales técnicos Windows")
    print("  Modelo LLM: phi3 (Ollama)  |  Embeddings: all-MiniLM-L6-v2")
    print("=" * 60)
    print("  Describe tu problema técnico y el asistente lo analizará.")
    print("  Escribe 'salir' para terminar.\n")


def main():
    # Cargar recursos estáticos
    system_prompt = load_text_file("prompts/system_prompt.txt")
    few_shot      = load_text_file("examples/few_shot_examples.txt")

    # Inicializar componentes RAG
    print("[→] Inicializando base de datos vectorial y modelo de embeddings...")
    collection, embed_model = initialize_retriever()
    print("[✓] Sistema RAG listo.\n")

    print_header()

    while True:
        try:
            pregunta = input("Usuario: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAsistente: ¡Hasta luego!")
            break

        if not pregunta:
            continue

        if pregunta.lower() in ("salir", "exit", "quit"):
            print("\nAsistente: ¡Hasta luego!")
            break

        # ── Pipeline RAG
        # Paso 1: Recuperar contexto relevante de ChromaDB
        context, sources = retrieve_context(pregunta, collection, embed_model, TOP_K)

        # Paso 2: Construir el prompt enriquecido con el contexto
        prompt_usuario = build_prompt(few_shot, context, pregunta)

        # Paso 3: Generar respuesta con el LLM
        print("\n  [→] Generando respuesta con el LLM...\n")
        respuesta = query_llm(system_prompt, prompt_usuario)

        # Paso 4: Mostrar respuesta
        print("\nAsistente:\n")
        print(respuesta)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()