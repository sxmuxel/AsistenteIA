# Asistente de Soporte Técnico con IA Híbrida mediante RAG

## Descripción del Proyecto

Asistente de inteligencia artificial diseñado para actuar como un **analista de soporte técnico**, capaz de diagnosticar problemas comunes en computadoras con Windows y sugerir soluciones paso a paso basándose en una base de conocimientos real.

El sistema implementa un flujo **RAG (Retrieval-Augmented Generation)** híbrido con interfaz gráfica web: los documentos técnicos se vectorizan y almacenan localmente en ChromaDB, y ante cada consulta se recuperan los fragmentos más relevantes antes de generar la respuesta con el LLM. Esto garantiza respuestas fundamentadas en información verificada, sin alucinaciones.

---

## Tecnologías Utilizadas

| Componente | Herramienta | Propósito |
|---|---|---|
| **LLM** | `gemini-3.1-flash-lite` | Generación de respuestas en lenguaje natural |
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

### Fase de Ingesta (local)

1. Los documentos `.txt` almacenados en `knowledge_base/` son cargados al sistema.

2. Cada documento se divide en chunks de 600 caracteres y un overlap de 100 caracteres

3. A cada chunk se le generan embeddings usando el modelo `all-MiniLM-L6-v2` con vectores de 384 dimensiones

4. Los embeddings se almacenan en: `ChromaDB`, con una base persistente ubicada en `chroma_db/`

5. La búsqueda semántica utiliza similitud coseno


### Fase de Consulta (híbrido)

1. El usuario realiza una pregunta desde la interfaz realizada con Streamlit

2. La pregunta del usuario se convierte en embedding usando `all-MiniLM-L6-v2` 

3. Se realiza una búsqueda semántica en ChromaDB por similitud coseno y realiza la recuperación de los Top-4 chunks más relevantes

4. El sistema construye un prompt aumentado con:
   - System Prompt
   - Few-Shot Examples
   - Contexto recuperado
   - Pregunta del usuario

5. El prompt es enviado a `gemini-3.5-flash-lite` 

6. Finalmente, la GUI muestra la respuesta generada y los chunks recuperados

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
Se utiliza el modelo `all-MiniLM-L6-v2` de sentence-transformers para convertir cada chunk en un vector de **384 dimensiones**.

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

### Preguntas dentro del dominio (deben responder correctamente)
 
| # | Pregunta | Manual Top-1 | Chunk Top-1 recuperado | Similitud Top-1 (%) | 
|---|---|---|---|---|
| 1 | Mi PC está muy lenta y el disco aparece al 100% | manual_hardware.txt | ...al 80% si hay problemas de calor. - Esto limita el uso máximo de CPU para reducir la temperatura. | 50.7% | 
| 2 | El Wi-Fi muestra conectado pero no hay internet | manual_redes.txt | ...ed (al final de la página). ADVERTENCIA: Esto elimina todas las configuraciones de red guardadas. SOLUCIÓN: VERIFICA... | 66.1% | 
| 3 | La PC se reinicia sola con pantalla azul (BSOD) | manual_software.txt | ...N 2: PANTALLA AZUL DE LA MUERTE (BSOD) SÍNTOMAS: El sis... | 48.6% | 
| 4 | Una aplicación no se abre y da error al intentarlo | manual_software.txt | ...CCIÓN 3: ERRORES AL INSTALAR PROGRAMAS SÍNTOMAS: La ins... | 62.4% | 
| 5 | Tengo 6GB de RAM y el sistema va muy lento | manual_hardware.txt | ...ión a los disipadores del CPU y GPU. 5. Recomendado hacerlo cada 6-12 meses dependiendo del entorno.... | 55.9% |
| 6 | El USB no es reconocido por Windows | manual_hardware.txt | ...tivo aparece brevemente y luego desaparece. - La unidad USB no aparece en el explorador de archivos. SOLUCIÓN: REPARAR ... | 69.3% |
| 7 | La impresora está conectada pero no imprime | manual_hardware.txt | ...- Asignar una letra de unidad disponible.... | 57.7% |
 
### Preguntas fuera del dominio (deben rechazar sin alucinar)
 
| # | Pregunta | Chunk Top-1 recuperado | Similitud Top-1 | Rechazó correctamente |
|---|---|---|---|---|
| 8 | El equipo se sobrecalienta y se apaga solo | ...CIONES QUE NO ABREN O SE CIERRAN SOLAS | 50.0% | ✅ |
| 9 | ¿Cómo instalo Ubuntu en dual boot con Windows? | ...CAUSAS MÁS FRECUENTES: 1. Exceso de programas al inicio de Windows (startup). 2. Uso elevado de CPU o RAM por procesos e... | 46.1% | ✅ |
| 10 | ¿Cuál es la mejor GPU para gaming en 2026? | 67.7% | ...ión a los disipadores del CPU y GPU. 5. Recomendado hacerlo cada 6-12 meses dependiendo del entorno.... | ✅ |
 
### Resumen de Resultados
 
| Métrica | Valor |
|---|---|
| **Total de pruebas** | 10 |
| **Respuestas correctas** | 10 |
| **Alucinaciones detectadas** | 0 |
| **Rechazos correctos** (fuera del dominio) | 3/3 |
 
### Caso de Éxito Destacado
La consulta #2 fue la que mas fidelidad presentó en la lista de preguntas dentro del dominio, con un 69.3% de precisión respecto al Chunk Top-1.
 
### Caso de Error / Rechazo Analizado
La consulta sobre instalación de Ubuntu en dual boot recuperó chunks vagamente relacionados con gestión de disco y software de Windows, pero con similitudes bajas. El modelo reconoció que el contexto era insuficiente y devolvió el mensaje de rechazo en lugar de inventar un procedimiento de instalación — demostrando que el System Prompt anti-alucinación funciona correctamente.