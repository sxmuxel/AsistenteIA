import os
import sys
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Carga las variables desde el archivo .env

# ── Configuración

CHROMA_DIR   = "chroma_db"
COLLECTION   = "soporte_tecnico"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "gemini-1.5-flash"
TOP_K        = 4

# Configurar Gemini con la API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY no encontrada.")
    print("        Crea un archivo .env basándote en .env.example")
    print("        y pega tu API Key de Gemini.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

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
    query_embedding = model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks     = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    print("\n  [RAG] Chunks recuperados:")
    sources = []
    for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
        fuente = meta.get("source", "desconocido")
        sources.append(fuente)
        similitud = round((1 - dist) * 100, 1)
        print(f"    {i+1}. {fuente} | Similitud: {similitud}% | {chunk[:60]}...")

    context_parts = []
    for chunk, meta in zip(chunks, metadatas):
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
    """Envía el prompt a Gemini y retorna la respuesta."""
    try:
        model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=system_prompt
        )
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"[ERROR] No se pudo conectar con Gemini: {e}\nVerifica que tu GEMINI_API_KEY sea válida."


# ── Interfaz de usuario

def print_header():
    print("\n" + "=" * 60)
    print("  ASISTENTE DE SOPORTE TÉCNICO WINDOWS  |  RAG Híbrido")
    print("  Base de conocimientos: Manuales técnicos Windows")
    print("  Modelo LLM: gemini-1.5-flash (Gemini API)")
    print("  Embeddings: all-MiniLM-L6-v2 (local)")
    print("=" * 60)
    print("  Describe tu problema técnico y el asistente lo analizará.")
    print("  Escribe 'salir' para terminar.\n")


def main():
    system_prompt = load_text_file("prompts/system_prompt.txt")
    few_shot      = load_text_file("examples/few_shot_examples.txt")

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
        context, sources = retrieve_context(pregunta, collection, embed_model, TOP_K)
        prompt_usuario   = build_prompt(few_shot, context, pregunta)

        print("\n  [→] Generando respuesta con Gemini...\n")
        respuesta = query_llm(system_prompt, prompt_usuario)

        print("\nAsistente:\n")
        print(respuesta)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()