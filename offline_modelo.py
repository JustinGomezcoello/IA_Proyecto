import os
import pyttsx3
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

# Cargar modelo de QA desde Hugging Face
print("Cargando modelo de QA...")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
print("Modelo de QA cargado.")

# Contextos predefinidos
contextos = {
    "monumentos": """
    Ecuador tiene varios monumentos y estatuas representativas:
    - La Mitad del Mundo: Construida en 1979, simboliza la línea ecuatorial que divide el mundo en hemisferios norte y sur. Es un importante destino turístico en Quito.
    - La Virgen del Panecillo: Inaugurada en 1975, representa a la Virgen María y es un símbolo religioso y cultural en Quito. Está ubicada en el cerro El Panecillo y es una de las estatuas más grandes de Ecuador.
    - Monumento a los Héroes del 24 de Mayo: Situado en Guayaquil, fue construido en 1938 para honrar a los héroes de la independencia de Ecuador.
    - Estatua de Simón Bolívar: Ubicada en el Parque de la Libertad, representa la lucha por la independencia y la unión latinoamericana.
    - Monumento a Atahualpa: Este monumento celebra al último emperador inca y su conexión con la historia de Ecuador.
    """,
    "planetas": """
    Los planetas del sistema solar son:
    - Mercurio: Gris y el más cercano al Sol. Es el planeta con mayor variación de temperatura entre el día y la noche.
    - Venus: Amarillento, tiene una atmósfera densa de dióxido de carbono. Es el planeta más caliente debido a su efecto invernadero.
    - Tierra: Azul y verde, alberga vida. Tiene océanos que cubren más del 70% de su superficie.
    - Marte: Rojizo debido al óxido de hierro. Tiene el volcán más alto del sistema solar, el Monte Olimpo.
    - Júpiter: Marrón con bandas blancas, es el más grande del sistema solar y tiene la Gran Mancha Roja, una tormenta gigantesca.
    - Saturno: Amarillo pálido, famoso por sus anillos de hielo y polvo.
    - Urano: Azul verdoso debido al metano en su atmósfera. Gira "de lado" comparado con otros planetas.
    - Neptuno: Azul oscuro, el planeta más lejano del Sol. Tiene los vientos más rápidos del sistema solar.
    """
}

def seleccionar_contexto(pregunta):
    """Selecciona el contexto adecuado basado en la pregunta"""
    if "monumentos" in pregunta.lower() or "estatuas" in pregunta.lower() or "panecillo" in pregunta.lower():
        return contextos["monumentos"]
    elif "planetas" in pregunta.lower() or any(p in pregunta.lower() for p in ["mercurio", "venus", "tierra", "marte", "júpiter", "saturno", "urano", "neptuno"]):
        return contextos["planetas"]
    else:
        return None

def reconocer_voz():
    """Convierte la entrada de voz a texto"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Escuchando tu pregunta (habla claro y no te apresures)...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=30, phrase_time_limit=30)
            texto = recognizer.recognize_google(audio, language="es-ES")
            print(f"Pregunta reconocida: {texto}")
            return texto
        except sr.UnknownValueError:
            print("No entendí lo que dijiste. Por favor, intenta nuevamente.")
            return None
        except sr.RequestError:
            print("Hubo un problema con el reconocimiento de voz.")
            return None
        except sr.WaitTimeoutError:
            print("No se detectó entrada de voz. Por favor, intenta nuevamente.")
            return None

def generar_respuesta(pregunta):
    """Genera una respuesta basada en contextos o el modelo QA"""
    contexto = seleccionar_contexto(pregunta)
    if contexto:
        # Si hay un contexto relevante, usar el modelo QA para responder
        respuesta = qa_pipeline(question=pregunta, context=contexto)
        return respuesta["answer"]
    else:
        # Si no hay contexto, devolver un mensaje genérico
        return "Lo siento, no tengo información relevante para tu pregunta."

def texto_a_voz(texto):
    """Convierte texto a voz"""
    engine.stop()
    engine.say(texto)
    engine.runAndWait()

def main():
    while True:
        print("\nDi tu pregunta o di 'salir' para terminar.")
        pregunta = reconocer_voz()
        if pregunta is None:
            continue
        if "salir" in pregunta.lower():
            print("¡Adiós!")
            texto_a_voz("¡Adiós!")
            break

        respuesta = generar_respuesta(pregunta)
        print(f"Respuesta: {respuesta}")
        texto_a_voz(respuesta)

if __name__ == "__main__":
    main()
