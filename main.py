# importar ollama
import ollama

# cargar system prompt
with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# cargar ejemplos few-shot
with open("examples/few_shot_examples.txt", "r", encoding="utf-8") as f:
    few_shot = f.read()

print("--- Asistente de Soporte Técnico IA ---")
print("Escribe tu problema técnico.")
print("Escribe 'salir' para terminar.\n")

while True:
    # leer pregunta del usuario
    pregunta = input("Usuario: ")

    if pregunta.lower() == "salir":
        print("Asistente: ¡Hasta luego!")
        break

    # construir prompt con delimitadores
    prompt_usuario = f"""
{few_shot}

<PREGUNTA_USUARIO>
{pregunta}
</PREGUNTA_USUARIO>
"""

    # enviar al modelo
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_usuario}
        ]
    )

    print("\nAsistente:\n")
    print(response["message"]["content"])
    print("\n---------------------------------\n")