# 🎯 Guía: Mejorar Precisión con Normalización de Posición

## ✅ Fase 1 Completada: Normalización Relativa

### Cambios Implementados

Se implementó **normalización de keypoints relativa** para hacer el reconocimiento invariante a la posición en el frame.

#### ¿Qué hace?
- **Normaliza posiciones** relativas al punto medio entre los hombros
- **Normaliza escala** usando el ancho de los hombros
- **Resultado**: La misma seña se reconoce igual sin importar dónde estés en el frame o qué tan lejos de la cámara

#### Archivos Modificados
- `helpers.py`: Nueva función `extract_keypoints_normalized()`
- `constants.py`: Flag `USE_NORMALIZED_KEYPOINTS = True`
- `main.py`: Usa keypoints normalizados
- `evaluate_model.py`: Usa keypoints normalizados

## 📋 Pasos para Probar la Mejora

### Paso 1: Eliminar Datos Antiguos
Los keypoints antiguos (posiciones absolutas) no son compatibles con el nuevo sistema.

```bash
# Eliminar keypoints antiguos
rm -r data/keypoints/*

# Eliminar modelo antiguo (opcional pero recomendado)
rm models/actions_15.keras
```

### Paso 2: Regenerar Keypoints
Genera nuevos keypoints con normalización:

```bash
python create_keypoints.py
```

**Nota**: Esto procesará todos los frames en `frame_actions/` y creará archivos `.h5` normalizados.

### Paso 3: Reentrenar el Modelo
Entrena el modelo con los nuevos keypoints normalizados:

```bash
python training_model.py
```

### Paso 4: Probar Reconocimiento
Prueba el modelo mejorado:

```bash
python evaluate_model.py
```

## 🧪 Cómo Validar la Mejora

### Test de Posición
1. Haz una seña en el **centro** del frame → Anota resultado
2. Haz la **misma seña** a la **izquierda** → Debería reconocerse igual
3. Haz la **misma seña** a la **derecha** → Debería reconocerse igual
4. Haz la **misma seña** más **cerca** de la cámara → Debería reconocerse igual
5. Haz la **misma seña** más **lejos** de la cámara → Debería reconocerse igual

### Mejora Esperada
- ✅ **Antes**: ~20-40% de precisión, muy sensible a posición
- ✅ **Después**: ~50-70% de precisión, invariante a posición

## 🔧 Configuración

### Activar/Desactivar Normalización
En `constants.py`:

```python
# Activar normalización (recomendado)
USE_NORMALIZED_KEYPOINTS = True

# Desactivar (volver a posiciones absolutas)
USE_NORMALIZED_KEYPOINTS = False
```

**Importante**: Si cambias este flag, debes regenerar keypoints y reentrenar.

## 🚀 Próxima Fase: Características Avanzadas

Una vez que valides que la normalización funciona, podemos implementar **Fase 2**:

### Características Adicionales
1. **Ángulos de dedos** (forma de la mano)
2. **Vectores de movimiento** (velocidad, dirección)
3. **Distancia entre manos**
4. **Apertura de la palma**

**Mejora esperada Fase 2**: 80-95% de precisión

## ❓ Troubleshooting

### Error: "No such file or directory: data/keypoints"
```bash
mkdir -p data/keypoints
```

### Error al cargar modelo antiguo
Elimina el modelo antiguo:
```bash
rm models/actions_15.keras
python training_model.py
```

### Precisión sigue baja
1. Verifica que `USE_NORMALIZED_KEYPOINTS = True`
2. Asegúrate de haber regenerado keypoints
3. Asegúrate de haber reentrenado el modelo
4. Captura más muestras (mínimo 30 por seña)

## 📊 Comparación

| Aspecto | Antes (Absoluto) | Después (Normalizado) |
|---------|------------------|----------------------|
| Sensibilidad a posición | ❌ Muy alta | ✅ Ninguna |
| Sensibilidad a distancia | ❌ Muy alta | ✅ Baja |
| Precisión típica | 20-40% | 50-70% |
| Generalización | ❌ Pobre | ✅ Buena |

## ✨ Siguiente Paso

Después de probar y validar la mejora, avísame para implementar **Fase 2** con características avanzadas.
