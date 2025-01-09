import pyttsx3
import speech_recognition as sr
from transformers import LlamaTokenizer, LlamaForCausalLM

# Cargar el modelo LLaMa
print("Cargando el modelo LLaMa...")
tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-7b")
model = LlamaForCausalLM.from_pretrained("hf-internal-testing/llama-7b", device_map="auto")
print("Modelo LLaMa cargado.")

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

def reconocer_voz():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando tu pregunta...")
        try:
            audio = recognizer.listen(source, timeout=20, phrase_time_limit=20)
            texto = recognizer.recognize_google(audio, language="es-ES")
            print(f"Pregunta reconocida: {texto}")
            return texto
        except sr.UnknownValueError:
            print("No entendí lo que dijiste.")
            return None
        except sr.RequestError:
            print("Hubo un problema con el reconocimiento de voz.")
            return None

def generar_respuesta_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def texto_a_voz(texto):
    engine.stop()
    engine.say(texto)
    engine.runAndWait()

def main():
    while True:
        print("Di tu pregunta o di 'salir' para terminar.")
        pregunta = reconocer_voz()
        if pregunta is None:
            continue
        if "salir" in pregunta.lower():
            print("¡Adiós!")
            texto_a_voz("¡Adiós!")
            break

        respuesta = generar_respuesta_llama(pregunta)
        print(f"Respuesta: {respuesta}")
        texto_a_voz(respuesta)

if __name__ == "__main__":
    main()
