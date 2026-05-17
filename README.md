# 🖥️ Asistente de Soporte Técnico con IA Híbrida mediante RAG

## Descripción del Proyecto

Asistente de inteligencia artificial diseñado para actuar como un **analista de soporte técnico**, capaz de diagnosticar problemas comunes en computadoras con Windows y sugerir soluciones paso a paso basándose en una base de conocimientos real.

El sistema implementa un flujo **RAG (Retrieval-Augmented Generation)** híbrido con interfaz gráfica web: los documentos técnicos se vectorizan y almacenan localmente en ChromaDB, y ante cada consulta se recuperan los fragmentos más relevantes antes de generar la respuesta con el LLM. Esto garantiza respuestas fundamentadas en información verificada, sin alucinaciones.

---

## Tecnologías Utilizadas

| Componente | Herramienta | Propósito |
|---|---|---|
| **LLM** | Google Gemini 3.1 Flash Lite (API) | Generación de respuestas en lenguaje natural |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Vectorización local de texto |
| **Base vectorial** | ChromaDB (persistente) | Almacenamiento y búsqueda por similitud coseno |
| **Interfaz gráfica** | Streamlit | GUI web interactiva con historial de chat |
| **Lenguaje** | Python | Orquestación del pipeline RAG |

---

## Estructura del Proyecto

```
AsistenteIA/
├── app.py                      # Interfaz gráfica (Streamlit)
├── ingest.py                   # Pipeline de ingesta y vectorización
├── main.py                     # Chatbot de terminal (alternativo)
├── requirements.txt            # Dependencias del proyecto
├── .env.example                # Plantilla de variables de entorno
├── .gitignore
│
├── knowledge_base/             # Manuales técnicos (fuentes de conocimiento)
│   ├── manual_hardware.txt
│   ├── manual_redes.txt
│   ├── manual_rendimiento.txt
│   └── manual_software.txt
│
├── prompts/
│   └── system_prompt.txt       # Prompt del sistema con reglas anti-alucinación
│
├── examples/
│   └── few_shot_examples.txt   # Ejemplos de formato de respuesta esperado
│
└── chroma_db/                  # Base de datos vectorial generada por ingest.py
```

---

## Base de Conocimientos

| Archivo | Contenido |
|---|---|
| `manual_rendimiento.txt` | Diagnóstico de lentitud, disco al 100%, optimización de inicio |
| `manual_redes.txt` | Problemas de Wi-Fi, Internet, Bluetooth, diagnóstico con ping/ipconfig |
| `manual_software.txt` | Aplicaciones que no abren, BSOD, errores de instalación, impresoras |
| `manual_hardware.txt` | Disco duro, RAM, temperatura, dispositivos USB |

---

## Arquitectura del Sistema RAG

```
┌─────────────────────────────────────────────────────────────┐
│              FASE DE INGESTA (100% local)                   │
│                                                             │
│  knowledge_base/*.txt  →  Chunks (600 chars, overlap 100)  │
│         ↓                                                   │
│  all-MiniLM-L6-v2  →  Embeddings (384 dimensiones)        │
│         ↓                                                   │
│  ChromaDB (similitud coseno, persistente en chroma_db/)    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              FASE DE CONSULTA (híbrido)                     │
│                                                             │
│  Usuario (GUI Streamlit)                                    │
│         ↓                                                   │
│  Embedding de la pregunta  →  all-MiniLM-L6-v2  [LOCAL]   │
│         ↓                                                   │
│  Búsqueda similitud coseno  →  Top-4 chunks  [LOCAL]       │
│         ↓                                                   │
│  Construcción del prompt aumentado:              [LOCAL]    │
│    [System Prompt] + [Few-Shot] + [Contexto] + [Pregunta]  │
│         ↓                                                   │
│  Gemini 2.5 Flash Lite (Google API) → Respuesta [NUBE ☁️]  │
│         ↓                                                   │
│  GUI: muestra respuesta + fuentes consultadas   [LOCAL]    │
└─────────────────────────────────────────────────────────────┘
```

---

## Proceso de Ingesta y Vectorización

### 1. Carga de Documentos
Se leen todos los archivos `.txt` del directorio `knowledge_base/`. Cada archivo corresponde a un manual técnico independiente.

### 2. Chunking
El texto de cada documento se divide en fragmentos de **600 caracteres** con un solapamiento de **100 caracteres**. La división se realiza preferentemente en saltos de párrafo (`\n\n`) para preservar la coherencia semántica de cada chunk.

```python
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
```

### 3. Generación de Embeddings
Se utiliza el modelo `all-MiniLM-L6-v2` de sentence-transformers para convertir cada chunk en un vector de **384 dimensiones**. Este modelo fue elegido por:
- Ejecución completamente local (sin API externa)
- Alto rendimiento en tareas de recuperación semántica en inglés y español
- Velocidad de inferencia apropiada para hardware de consumo
- Tamaño compacto (~90MB)

### 4. Almacenamiento en ChromaDB
Los vectores se almacenan en una colección ChromaDB configurada con **similitud coseno** (`hnsw:space: cosine`). El almacenamiento es persistente en el directorio `chroma_db/`.

---

## Construcción del Prompt Aumentado

El prompt enviado al LLM tiene cuatro componentes concatenados:

```
[SYSTEM PROMPT]
Define el comportamiento del asistente, el idioma, el formato
de respuesta y las reglas estrictas anti-alucinación.
↓
[FEW-SHOT EXAMPLES]
Dos ejemplos completos (pregunta → respuesta) que muestran
el formato Markdown estructurado esperado.
↓
<CONTEXTO_RECUPERADO>
Los 4 chunks más relevantes recuperados de ChromaDB,
cada uno precedido de su fuente ([Fuente: manual_X.txt]).
</CONTEXTO_RECUPERADO>
↓
<PREGUNTA_USUARIO>
La consulta original del usuario sin modificar.
</PREGUNTA_USUARIO>
```

El System Prompt incluye una instrucción crítica: si el contexto no contiene información suficiente, el modelo debe responder `"No encuentro esa información en la base de conocimientos técnicos"` en lugar de generar contenido inventado.

---

## Interfaz Gráfica (Streamlit)

La GUI se construyó con **Streamlit** e incluye:

- **Chat interactivo** con historial de conversación persistente durante la sesión
- **Burbujas diferenciadas** para mensajes de usuario y del asistente
- **Panel de fuentes RAG** (expandible) que muestra qué manuales se consultaron para cada respuesta
- **Modo debug** (toggle en sidebar) que muestra el porcentaje de similitud coseno de cada chunk recuperado
- **Contador de consultas** realizadas en la sesión
- **Información del sistema** en el sidebar (modelo LLM, embeddings, Top-K configurado)
- **Botón para limpiar** la conversación

---

## Instrucciones de Instalación (Windows)

### Paso 1: Obtener una API Key de Gemini
Ir a https://aistudio.google.com/app/apikey y generar una clave.

### Paso 2: Clonar el repositorio
```bash
git clone https://github.com/sxmuxel/AsistenteIA.git
cd AsistenteIA
```

### Paso 3: Crear entorno virtual
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Paso 4: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 5: Configurar la API Key
Copia el archivo de ejemplo y pega tu clave dentro:
```bash
copy .env.example .env
```
Abre `.env` y reemplaza `tu_clave_aqui` con tu API Key real.

### Paso 6: Indexar la base de conocimientos *(solo la primera vez)*
```bash
python ingest.py
```

---

## Ejecución

### Interfaz Gráfica 
```bash
streamlit run app.py
```
Se abrirá en el navegador en `http://localhost:8501`

Para finalizar, escribir `salir`.

---

## Informe de Evaluación — 10 Preguntas de Prueba

Las siguientes pruebas evalúan dos métricas:
- **Fidelidad**: la respuesta está basada en el contexto recuperado, no inventada.
- **Relevancia**: la respuesta resuelve correctamente la pregunta del usuario.

El porcentaje de similitud y el chunk recuperado se obtienen activando el **modo debug** en el sidebar de la GUI.

