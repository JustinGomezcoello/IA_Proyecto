from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # Cargar modelo y tokenizer desde Hugging Face
    print("Cargando modelo...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", device_map="auto")

    print("¡Modelo cargado! Pregunta lo que quieras.")
    
    while True:
        pregunta = input("Haz tu pregunta (o escribe 'salir' para terminar): ")
        if pregunta.lower() in ["salir", "exit"]:
            print("¡Adiós!")
            break

        inputs = tokenizer(pregunta, return_tensors="pt").to("cuda")  # Enviar a la GPU
        outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
        respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()
