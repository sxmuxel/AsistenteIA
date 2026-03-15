# Analista de Soporte Técnico con IA Local

## Descripción del Proyecto
Asistente de inteligencia artificial diseñado para actuar como un **analista de soporte técnico**, capaz de diagnosticar problemas comunes en computadoras y sugerir soluciones paso a paso.

El sistema utiliza un **modelo de lenguaje ejecutado localmente** mediante Ollama, lo que permite realizar inferencias sin depender de servicios en la nube, garantizando privacidad de los datos y funcionamiento offline.

---

## Tecnologías Utilizadas

- Python
- Ollama (ejecución local de modelos LLM)
- Modelo Phi-3

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
cd Asistente IA
```
### Paso 3: Crear entorno virtual

En la terminal:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### Paso 4: Instalar dependencias

En la terminal:
```bash
pip install -r requirements.txt
```

---

## Ejecución

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

```
Mi computadora está muy lenta
```

Asistente:

<img width="1241" height="501" alt="Image" src="https://github.com/user-attachments/assets/71912080-ec40-4348-98cd-04155bea614b" />
