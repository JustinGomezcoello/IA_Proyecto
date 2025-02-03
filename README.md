# IA_Proyecto

IA_Proyecto es un sistema basado en **inteligencia artificial** que incorpora modelos de **aprendizaje profundo** para diversas aplicaciones. Utiliza **PyTorch** como framework principal y permite la ejecución de modelos en línea y sin conexión.

## 📌 Características principales
- Implementación de modelos de **aprendizaje profundo**.
- Uso de **PyTorch** para el entrenamiento y prueba de modelos.
- Soporte para **modelos offline**.
- API para exponer funcionalidades de IA.

## 🚀 Tecnologías utilizadas
- **Python 3.8+** (Lenguaje principal).
- **PyTorch** (Framework de aprendizaje profundo).
- **Flask/FastAPI** (para la exposición de modelos vía API).
- **LLaMa** (Modelo de lenguaje avanzado).

## 📂 Estructura del proyecto
```
IA_Proyecto/
│── apis.py             # Endpoints para la API
│── app_basica.py       # Aplicación principal
│── LLaMa.py           # Implementación del modelo de lenguaje
│── offline_modelo.py   # Modelo ejecutable sin conexión
│── test_pytorch.py     # Pruebas con PyTorch
│── pyvenv.cfg          # Configuración del entorno virtual
```

## 🔧 Configuración
### 1️⃣ Prerrequisitos
- Tener instalado **Python 3.8+**.
- Instalar **PyTorch** y las dependencias necesarias:
```sh
pip install torch torchvision torchaudio flask fastapi
```



## 📬 Endpoints disponibles
Puedes probar los endpoints usando **Postman** o `curl`:
- **GET /modelo/status** → Verifica el estado del modelo.
- **POST /modelo/predict** → Realiza una predicción con datos de entrada.


---
🚀 **IA_Proyecto** permite la integración de modelos de IA de manera flexible y escalable. ¡Explora sus capacidades! 🤖

