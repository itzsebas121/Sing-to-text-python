# 🤟 Sign Language Recognition System - Real-time Translation to Text and Speech

Real-time sign language recognition and translation system using Deep Learning and Computer Vision.

## 📋 Project Description

This project implements a complete sign language recognition system that captures gestures via webcam, processes them using MediaPipe to extract keypoints from body, hands, and face, and classifies them using an LSTM neural network to translate them into text and speech.

## 🧠 Arquitectura de la Red Neuronal

### Tipo de Red
**LSTM (Long Short-Term Memory)** - Red Neuronal Recurrente especializada en secuencias temporales

### Arquitectura del Modelo

```
Input: (15 frames, 1662 keypoints)
    ↓
LSTM Layer 1: 64 unidades + Dropout (0.5) + L2 Regularization (0.01)
    ↓
LSTM Layer 2: 128 unidades + Dropout (0.5) + L2 Regularization (0.001)
    ↓
Dense Layer 1: 64 neuronas + ReLU + L2 Regularization (0.001)
    ↓
Dense Layer 2: 64 neuronas + ReLU + L2 Regularization (0.001)
    ↓
Output Layer: N clases + Softmax
```

**Parámetros clave:**
- **Frames por secuencia:** 15 frames normalizados
- **Keypoints totales:** 1,662 puntos por frame
  - Pose: 33 landmarks × 4 coordenadas (x, y, z, visibility) = 132
  - Rostro: 468 landmarks × 3 coordenadas (x, y, z) = 1,404
  - Mano izquierda: 21 landmarks × 3 coordenadas = 63
  - Mano derecha: 21 landmarks × 3 coordenadas = 63
- **Optimizador:** Adam
- **Función de pérdida:** Categorical Crossentropy
- **Métrica:** Accuracy

### Regularización
- **Dropout:** 50% para prevenir overfitting
- **L2 Regularization:** Aplicada en todas las capas LSTM y Dense
- **Early Stopping:** Paciencia de 10 épocas monitoreando accuracy

## 🔬 Algoritmos y Técnicas Utilizadas

### 1. Extracción de Características
- **MediaPipe Holistic:** Detección de 1,662 puntos clave en tiempo real
  - Pose estimation (33 puntos)
  - Face mesh (468 puntos)
  - Hand tracking bilateral (42 puntos total)

### 2. Preprocesamiento de Datos
- **Normalización temporal:** Interpolación/submuestreo a 15 frames fijos
  - Interpolación lineal para secuencias cortas
  - Submuestreo uniforme para secuencias largas
- **Padding:** Pre-padding con ceros para secuencias variables
- **Normalización de datos:** Conversión a float16 para eficiencia

### 3. Detección de Inicio/Fin de Seña
- **Algoritmo de ventana deslizante:**
  - Margen de frames: 1 frame
  - Delay de confirmación: 3 frames
  - Mínimo de frames: 5 frames
  - Detección basada en presencia de manos

### 4. Clasificación
- **Umbral de confianza:** 70-80% para aceptar predicción
- **Softmax:** Probabilidades normalizadas para cada clase
- **Argmax:** Selección de clase con mayor probabilidad

### 5. Post-procesamiento
- **Text-to-Speech:** Google TTS (gTTS) para síntesis de voz en español
- **Pygame:** Reproducción de audio generado

## 📁 Estructura del Proyecto

```
modelo_lstm_lsp/
├── 📄 Archivos Principales
│   ├── main.py                    # Interfaz GUI con PyQt5
│   ├── capture_samples.py         # Captura de muestras de entrenamiento
│   ├── normalize_samples.py       # Normalización de frames a 15 frames
│   ├── create_keypoints.py        # Extracción de keypoints con MediaPipe
│   ├── training_model.py          # Entrenamiento del modelo LSTM
│   ├── evaluate_model.py          # Evaluación y pruebas del modelo
│   └── confusion_matrix.py        # Generación de matriz de confusión
│
├── 🛠️ Archivos de Soporte
│   ├── model.py                   # Definición de arquitectura LSTM
│   ├── helpers.py                 # Funciones auxiliares
│   ├── constants.py               # Constantes y configuración
│   ├── text_to_speech.py          # Conversión texto a voz
│   ├── server.py                  # API Flask para procesamiento de videos
│   └── process_video.py           # Procesamiento de videos externos
│
├── 🎨 Interfaz
│   ├── mainwindow.ui              # Diseño de interfaz Qt (v1)
│   └── mainwindow_2.ui            # Diseño de interfaz Qt (v2)
│
├── 📂 Directorios de Datos
│   ├── frame_actions/             # Frames capturados por palabra
│   ├── data/                      # Datos procesados
│   │   ├── keypoints/             # Archivos .h5 con keypoints
│   │   └── data.json              # Metadatos
│   └── models/                    # Modelos entrenados
│       ├── actions_15.keras       # Modelo LSTM entrenado
│       └── words.json             # IDs de palabras reconocidas
│
└── 📋 Configuración
    ├── requirements.txt           # Dependencias del proyecto
    └── README.md                  # Este archivo
```

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 - 3.10 (recomendado 3.8 por compatibilidad con TensorFlow 2.10)
- Webcam funcional
- Windows/Linux/MacOS

### Instalación de Dependencias

```bash
# Clonar el repositorio
git clone https://github.com/itzsebas121/Sing-to-text-python
cd Sing-to-text-python

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Crear estructura de directorios necesaria
mkdir -p data/keypoints frame_actions  # Linux/Mac
# En Windows: mkdir data\keypoints, mkdir frame_actions
```

### Configuración Inicial Después de Clonar

> **⚠️ IMPORTANTE:** El repositorio NO incluye datos de entrenamiento ni modelos pre-entrenados debido a su gran tamaño. Después de clonar, tienes **dos opciones**:

#### Opción 1: Entrenar tu Propio Modelo (Recomendado para Aprender) 🎓

Esta opción te permite entender todo el proceso desde cero:

1. **Capturar tus propias muestras** para cada palabra que quieras reconocer
2. **Normalizar** las muestras capturadas
3. **Generar keypoints** de las muestras
4. **Entrenar** el modelo LSTM
5. **Evaluar** el modelo entrenado

```bash
# Sigue el flujo completo descrito en la sección "Guía de Uso"
python capture_samples.py      # Paso 1
python normalize_samples.py    # Paso 2
python create_keypoints.py     # Paso 3
python training_model.py       # Paso 4
python evaluate_model.py       # Paso 5
```

**Tiempo estimado:** 2-4 horas (dependiendo de cuántas palabras captures)

#### Opción 2: Descargar Modelo Pre-entrenado (Inicio Rápido) ⚡

Si solo quieres probar el sistema sin entrenar:

1. Descarga el modelo pre-entrenado y datos desde [enlace-a-releases] *(próximamente)*
2. Extrae los archivos en las carpetas correspondientes:
   - `models/actions_15.keras` - Modelo entrenado
   - `models/words.json` - Lista de palabras (ya incluido en el repo)
   - `data/keypoints/*.h5` - Keypoints de entrenamiento (opcional)
3. Ejecuta directamente:

```bash
python evaluate_model.py  # Prueba en tiempo real
# o
python main.py           # Interfaz GUI
```

**Nota:** El modelo pre-entrenado reconoce las palabras listadas en la sección "Palabras Reconocidas Actualmente".


### Dependencias Principales
```
tensorflow==2.10.1          # Framework de Deep Learning
keras==2.10.0               # API de alto nivel para TensorFlow
mediapipe==0.10.11          # Detección de pose y manos
opencv-contrib-python==4.9.0.80  # Procesamiento de imágenes
numpy==1.26.4               # Operaciones numéricas
pandas==2.2.2               # Manejo de datos
PyQt5==5.15.9               # Interfaz gráfica
gTTS==2.5.1                 # Text-to-Speech
pygame==2.5.2               # Reproducción de audio
Flask==3.0.2                # API REST (opcional)
tables==3.9.2               # Manejo de archivos HDF5
protobuf==3.20.3            # Serialización de datos
```

## 📖 Guía de Uso

### Flujo de Trabajo Completo

#### 1️⃣ Captura de Muestras
```bash
python capture_samples.py
```
- Modifica la variable `word_name` en el script para la palabra a capturar
- Realiza la seña frente a la cámara múltiples veces (recomendado: 50-100 muestras)
- Las muestras se guardan en `frame_actions/<palabra>/sample_<timestamp>/`
- Presiona 'q' para salir

**Consejos:**
- Varía la velocidad de ejecución de la seña
- Cambia ligeramente la posición y ángulo
- Usa diferentes iluminaciones
- Captura con ambas manos si aplica

#### 2️⃣ Normalización de Frames
```bash
python normalize_samples.py
```
- Normaliza todas las muestras a exactamente 15 frames
- Usa interpolación para secuencias cortas
- Usa submuestreo para secuencias largas
- Sobrescribe los frames originales

#### 3️⃣ Generación de Keypoints
```bash
python create_keypoints.py
```
- Extrae los 1,662 keypoints de cada frame usando MediaPipe
- Genera archivos `.h5` en `data/keypoints/<palabra>.h5`
- Procesa todas las palabras en `frame_actions/` por defecto
- Muestra progreso en tiempo real

#### 4️⃣ Entrenamiento del Modelo
```bash
python training_model.py
```
- Entrena la red LSTM con todas las palabras disponibles
- Parámetros por defecto: 500 épocas máximo, early stopping con paciencia 10
- División: 95% entrenamiento, 5% validación
- Guarda el modelo en `models/actions_15.keras`
- Muestra resumen del modelo y métricas

**Personalización:**
```python
# En training_model.py
training_model(MODEL_PATH, epochs=1000)  # Cambiar número de épocas
```

#### 5️⃣ Evaluación del Modelo
```bash
python evaluate_model.py
```
- Prueba el modelo en tiempo real con la cámara
- Muestra predicciones con porcentaje de confianza
- Reproduce audio de las palabras reconocidas
- Umbral de confianza: 80% por defecto
- Presiona 'q' para salir

**Parámetros ajustables:**
```python
evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3)
# src: None para cámara, o ruta de video
# threshold: umbral de confianza (0.0-1.0)
```

#### 6️⃣ Interfaz Gráfica (GUI)
```bash
python main.py
```
- Interfaz PyQt5 con visualización en tiempo real
- Muestra keypoints sobre el video
- Traduce señas a texto y voz automáticamente
- Acumula palabras reconocidas en la interfaz

#### 7️⃣ Matriz de Confusión (Opcional)
```bash
python confusion_matrix.py
```
- Genera matriz de confusión para evaluar el modelo
- Visualiza errores de clasificación entre clases
- Útil para identificar señas que se confunden

#### 8️⃣ API REST (Opcional)
```bash
python server.py
```
- Inicia servidor Flask en `http://0.0.0.0:5000`
- Endpoint: `POST /upload_video` - Procesa videos y retorna traducción
- Útil para integración con aplicaciones móviles/web

## ⚙️ Configuración Avanzada

### Modificar Parámetros del Modelo
Edita `constants.py`:
```python
MIN_LENGTH_FRAMES = 5      # Mínimo de frames para detectar seña
LENGTH_KEYPOINTS = 1662    # Total de keypoints (NO MODIFICAR)
MODEL_FRAMES = 15          # Frames por secuencia (requiere reentrenamiento)
```

### Agregar Nuevas Palabras
1. Edita `constants.py` y agrega la palabra a `words_text`:
```python
words_text = {
    "hola": "HOLA",
    "adios": "ADIOS",
    "nueva_palabra": "NUEVA PALABRA",  # Agregar aquí
}
```
2. Captura muestras con `capture_samples.py`
3. Normaliza con `normalize_samples.py`
4. Genera keypoints con `create_keypoints.py`
5. Reentrena el modelo con `training_model.py`

### Ajustar Arquitectura del Modelo
Edita `model.py`:
```python
def get_model(max_length_frames, output_length: int):
    model = Sequential()
```

### Error: Incompatibilidad de TensorFlow
- Asegúrate de usar Python 3.8-3.10
- TensorFlow 2.10 requiere protobuf 3.20.x

### Cámara no detectada
- Cambia el índice de cámara en los scripts:
```python
video = cv2.VideoCapture(0)  # Prueba con 0, 1, 2, etc.
```

### Modelo no reconoce señas
- Verifica que el umbral no sea muy alto (reduce de 0.8 a 0.6)
- Asegúrate de tener suficientes muestras de entrenamiento (>50 por palabra)
- Revisa que la iluminación sea adecuada
- Confirma que las manos sean visibles en el frame

### Error al cargar modelo .keras
- Verifica que el archivo exista en `models/actions_15.keras`
- Reentrena el modelo si es necesario

## 📈 Mejoras Futuras

- [ ] Implementar atención (Attention Mechanism) en LSTM
- [ ] Agregar más palabras al vocabulario
- [ ] Implementar traducción de frases completas
- [ ] Optimizar para dispositivos móviles (TensorFlow Lite)
- [ ] Mejorar detección con transformers
- [ ] Agregar soporte para otras lenguas de señas
- [ ] Implementar data augmentation para mejorar generalización

## 🎥 Video Tutorial

Explicación detallada del código: [https://youtu.be/3EK0TxfoAMk](https://youtu.be/3EK0TxfoAMk)

*Nota: Próximamente video con las mejoras implementadas*

## 📄 Licencia

Este proyecto es de código abierto y está disponible para fines educativos y de investigación.

## 👨‍💻 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas, sugerencias o colaboraciones, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ para la comunidad sorda**
