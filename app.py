import os
import sys
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Carga las variables desde el archivo .env

# ── Configuración

CHROMA_DIR  = "chroma_db"
COLLECTION  = "soporte_tecnico"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "gemini-3.1-flash-lite"
TOP_K       = 4

# Configurar Gemini con la API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error(
        "⚠️ **GEMINI_API_KEY** no encontrada.\n\n"
        "Crea un archivo `.env` en la raíz del proyecto basándote en `.env.example` "
        "y pega tu API Key de Gemini."
    )
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ── Utilidades

def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        st.error(f"Archivo no encontrado: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@st.cache_resource(show_spinner="Cargando base de datos vectorial y modelo de embeddings...")
def initialize_retriever():
    if not os.path.exists(CHROMA_DIR):
        st.error("La base de datos vectorial no existe. Ejecuta primero: python ingest.py")
        st.stop()

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collections = [c.name for c in client.list_collections()]
    if COLLECTION not in collections:
        st.error(f"La colección '{COLLECTION}' no existe. Ejecuta primero: python ingest.py")
        st.stop()

    collection = client.get_collection(COLLECTION)
    model = SentenceTransformer(EMBED_MODEL)
    return collection, model


def retrieve_context(query: str, collection, model, top_k: int):
    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    sources_info = []
    context_parts = []

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        fuente    = meta.get("source", "desconocido")
        similitud = round((1 - dist) * 100, 1)
        sources_info.append({"source": fuente, "similarity": similitud, "preview": chunk[:120]})
        context_parts.append(f"[Fuente: {fuente}]\n{chunk}")

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources_info


def build_prompt(few_shot: str, context: str, question: str) -> str:
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
    try:
        model = genai.GenerativeModel(
            model_name=LLM_MODEL,
            system_instruction=system_prompt
        )
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"❌ **Error al conectar con Gemini:** {e}\n\nVerifica que tu `GEMINI_API_KEY` sea válida."


# ── Configuración de la página

st.set_page_config(
    page_title="Soporte Técnico IA",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos personalizados

st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
        border-left: 4px solid #3b82f6;
    }
    .main-header h1 {
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0 0 6px 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Burbujas del chat */
    .user-bubble {
        background: #1e40af;
        color: #e0f2fe;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin: 12px 0 12px 60px;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .assistant-bubble {
        background: #0f172a;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 16px 16px 16px 4px;
        padding: 18px 22px;
        margin: 12px 60px 12px 0;
        font-size: 0.93rem;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Tarjetas de fuentes */
    .source-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
    }
    .source-card .similarity {
        color: #34d399;
        font-weight: 600;
    }
    .source-card .filename {
        color: #60a5fa;
        font-weight: 500;
    }

    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* Input del chat */
    .stChatInput > div {
        background: #1e293b !important;
        border-color: #334155 !important;
        border-radius: 12px !important;
    }

    /* Botones */
    .stButton > button {
        background: #1e40af;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        border: none;
    }

    /* Métricas sidebar */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        text-align: center;
    }
    .metric-box .value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #3b82f6;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-box .label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ── Inicialización del estado de la sesión

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "sources_log" not in st.session_state:
    st.session_state.sources_log = []


# ── Cargar recursos

system_prompt = load_text_file("prompts/system_prompt.txt")
few_shot      = load_text_file("examples/few_shot_examples.txt")
collection, embed_model = initialize_retriever()


# ── Sidebar

with st.sidebar:
    st.markdown("## 🖥️ Asistente de Soporte Técnico con IA")

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.sources_log = []
        st.rerun()

    show_sources = st.toggle("Mostrar fuentes RAG", value=True)
    show_debug   = st.toggle("Modo debug (similitudes)", value=False)        

    st.markdown("### Sistema")
    st.markdown(f"""
    <div class="metric-box">
        <div class="value">{st.session_state.total_queries}</div>
        <div class="label">Consultas realizadas</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Setup")
    st.markdown(f"""
    - **LLM:** `{LLM_MODEL}`
    - **Embeddings:** `all-MiniLM-L6-v2`
    - **Vector DB:** ChromaDB
    - **Top-K chunks:** {TOP_K}
    """)

    st.markdown("### Manuales disponibles")
    st.markdown("""
    - `manual_hardware.txt`
    - `manual_redes.txt`
    - `manual_rendimiento.txt`
    - `manual_software.txt`
    """)

# ── Área principal

st.markdown("""
<div class="main-header">
    <h1>Asistente de Soporte Técnico Windows</h1>
    <p>RAG Híbrido · ChromaDB · all-MiniLM-L6-v2 (local) · gemini-3.1-flash (nube)</p>
</div>
""", unsafe_allow_html=True)

# Mensaje de bienvenida si no hay historial
if not st.session_state.messages:
    st.info(
        "👋 **¡Hola!** Describe tu problema técnico con Windows y te ayudaré a diagnosticarlo y resolverlo.\n\n"
        "**Ejemplos:** lentitud del sistema, problemas de Wi-Fi, pantalla azul, disco al 100%, errores de instalación..."
    )

# ── Mostrar historial del chat

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

        # Mostrar fuentes si están disponibles y el toggle está activo
        if show_sources and "sources" in msg:
            with st.expander("Fuentes consultadas en la base de conocimientos"):
                for src in msg["sources"]:
                    if show_debug:
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="filename">📄 {src['source']}</span> &nbsp;|&nbsp;
                            <span class="similarity">Similitud: {src['similarity']}%</span><br>
                            <span style="color:#64748b;">...{src['preview']}...</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        fuentes_unicas = list(dict.fromkeys(s["source"] for s in msg["sources"]))
                        for f in fuentes_unicas:
                            st.markdown(f"- 📄 `{f}`")
                        break  # Evitar loop para fuentes únicas


# ── Input del usuario

if prompt := st.chat_input("Describe tu problema técnico..."):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1

    # Mostrar burbuja del usuario
    st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

    # Pipeline RAG
    with st.spinner("Buscando en la base de conocimientos..."):
        context, sources_info = retrieve_context(prompt, collection, embed_model, TOP_K)

    with st.spinner("Generando respuesta con el LLM..."):
        prompt_llm = build_prompt(few_shot, context, prompt)
        respuesta  = query_llm(system_prompt, prompt_llm)

    # Guardar y mostrar respuesta
    st.session_state.messages.append({
        "role":    "assistant",
        "content": respuesta,
        "sources": sources_info
    })
    st.session_state.sources_log.extend(sources_info)

    st.markdown(f'<div class="assistant-bubble">{respuesta}</div>', unsafe_allow_html=True)

    if show_sources:
        with st.expander("Fuentes consultadas en la base de conocimientos"):
            if show_debug:
                for src in sources_info:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="filename">📄 {src['source']}</span> &nbsp;|&nbsp;
                        <span class="similarity">Similitud: {src['similarity']}%</span><br>
                        <span style="color:#64748b;">...{src['preview']}...</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                fuentes_unicas = list(dict.fromkeys(s["source"] for s in sources_info))
                for f in fuentes_unicas:
                    st.markdown(f"- 📄 `{f}`")