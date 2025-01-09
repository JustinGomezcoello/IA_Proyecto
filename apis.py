import os
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
import openai

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configurar las credenciales de Azure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
TEXT_DEPLOYMENT_ID = os.getenv("TEXT_DEPLOYMENT_ID")

if not all([OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_TYPE, OPENAI_API_VERSION, TEXT_DEPLOYMENT_ID]):
    raise ValueError("Faltan variables de entorno para la configuración de Azure OpenAI.")

# Configurar el cliente de OpenAI para Azure
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE
openai.api_type = OPENAI_API_TYPE
openai.api_version = OPENAI_API_VERSION

# Inicializar el motor de texto a voz
engine = pyttsx3.init()

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
            audio = recognizer.listen(source, timeout=20, phrase_time_limit=20)
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

def generar_respuesta_openai(pregunta):
    """Genera una respuesta usando Azure OpenAI con la nueva API"""
    try:
        response = openai.ChatCompletion.create(
            engine=TEXT_DEPLOYMENT_ID,  # Nombre del modelo desplegado en Azure
            messages=[
                {"role": "system", "content": "Eres un asistente que responde preguntas de manera clara y concisa."},
                {"role": "user", "content": pregunta}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Lo siento, hubo un problema al generar la respuesta: {e}"


def generar_respuesta(pregunta):
    """Genera una respuesta usando contexto o Azure OpenAI"""
    contexto = seleccionar_contexto(pregunta)
    if contexto:
        # Usar el contexto para generar una respuesta
        prompt = f"Pregunta: {pregunta}\nContexto: {contexto}\nRespuesta:"
        return generar_respuesta_openai(prompt)
    else:
        # Usar Azure OpenAI para una respuesta general
        return generar_respuesta_openai(pregunta)

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
