# Analista de Soporte Técnico con IA Local mediante RAG

## Descripción del Proyecto
Asistente de inteligencia artificial diseñado para actuar como un **analista de soporte técnico**, capaz de diagnosticar problemas comunes en computadoras y sugerir soluciones paso a paso basándose en una base de conocimientos real.

El sistema implementa un flujo **RAG (Retrieval-Augmented Generation)** completamente local: los documentos técnicos se vectorizan y almacenan en una base de datos vectorial, y ante cada consulta del usuario se recuperan los fragmentos más relevantes antes de generar la respuesta con el LLM. Esto garantiza respuestas fundamentadas en información verificada, no en conocimiento genérico del modelo.

---

## Tecnologías Utilizadas

| Componente | Herramienta | Propósito |
|---|---|---|
| LLM local | Ollama + phi3 | Generación de respuestas en lenguaje natural |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Vectorización local de texto |
| Base vectorial | ChromaDB | Almacenamiento y búsqueda por similitud |
| Lenguaje | Python | Orquestación del pipeline |

---

## Base de Conocimientos
 
La base de conocimientos está compuesta por 4 manuales técnicos ubicados en `knowledge_base/`:
 
| Archivo | Contenido |
|---|---|
| `manual_rendimiento.txt` | Diagnóstico de lentitud, disco al 100%, optimización de inicio |
| `manual_redes.txt` | Problemas de Wi-Fi, Internet, Bluetooth, diagnóstico con ping/ipconfig |
| `manual_software.txt` | Aplicaciones que no abren, BSOD, errores de instalación, impresoras |
| `manual_hardware.txt` | Disco duro, RAM, temperatura, dispositivos USB |
 
---

# Arquitectura del Sistema RAG

## Fase de Ingesta (`ingest.py`)

Proceso de preparación de la base de conocimientos:

1. **Cargar documentos:** Lectura de archivos desde `knowledge_base/`.
2. **Dividir en chunks:** Fragmentación del texto en bloques de 600 caracteres con un overlap de 100 caracteres.
3. **Generar embeddings:** Conversión de los fragmentos mediante `sentence-transformers` (local).
4. **Almacenar en ChromaDB:** Persistencia de los vectores en el directorio `chroma_db/`.

## Fase de Consulta (`main.py`)

Flujo de ejecución para responder preguntas del usuario:

1. **Generar embedding de la pregunta:** Se utiliza el mismo modelo para vectorizar la consulta del usuario.
2. **Búsqueda de similitud:** Localización de los **Top-4 chunks** más relevantes en ChromaDB.
3. **Construcción del Prompt:** Se estructura el envío al modelo con los siguientes componentes:
    * **System Prompt:** Define el comportamiento del asistente.
    * **Few-Shot Examples:** Muestra el formato de salida esperado.
    * **Contexto recuperado:** Chunks relevantes obtenidos de la base de datos.
    * **Pregunta del usuario:** La consulta original.
4. **Generación con LLM:** Uso de **phi3 (Ollama)** para generar la respuesta basada en el contexto.
5. **Salida:** Entrega de la respuesta en formato **Markdown estructurado**.

---

## Instrucciones de ejecución (Windows)

### Paso 1: Instalar Ollama

Descargar e instalar Ollama desde:

https://ollama.com

Desde la terminal instalar el modelo:
```bash
ollama pull phi3
```

### Paso 2: Clonar repositorio

En la terminal:
```bash
git clone https://github.com/sxmuxel/AsistenteIA.git
cd AsistenteIA
```
### Paso 3: Crear entorno virtual

En la terminal:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Paso 4: Instalar dependencias requeridas

En la terminal:
```bash
pip install -r requirements.txt
```

### Paso 5: Indexar la base de conocimientos (solo la primera vez)
Este script carga los manuales, los divide en chunks, genera embeddings y los almacena en ChromaDB.
 
```bash
python ingest.py
```

---

## Ejecución

Asegúrese de que la aplicación Ollama esté ejecutándose en segundo plano antes de iniciar el asistente.

Para iniciar el asistente ejecutar:

```bash
python main.py
```

El sistema iniciará un chatbot en la terminal donde el usuario puede describir su problema técnico.

Para finalizar la conversación escribir:

```
salir
```

---

## Ejemplo de Interacción

Usuario:

`Mi pc está muy lenta, tengo 6GB de RAM`

Asistente:

<img width="975" height="579" alt="image" src="https://github.com/user-attachments/assets/d65a8ac0-dc66-46e5-8cff-01ad5874333b" />

