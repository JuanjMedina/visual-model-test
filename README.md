# 🎯 TACO-YOLO: Sistema de Entrenamiento

Este sistema automatiza el proceso completo de entrenamiento de modelos YOLO usando el dataset TACO para detección de basura.

## 📋 Requisitos

### Dependencias principales

```bash
pip install -r ../requirements_yolo.txt
```

### Hardware recomendado

- **GPU**: NVIDIA con CUDA (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB+
- **Almacenamiento**: Mínimo 10GB de espacio libre

## 🚀 Inicio Rápido

### 1. Entrenamiento completo automatizado

```bash
# Entrenamiento básico con modelo nano
python run_complete_training.py

# Entrenamiento con modelo small y 200 épocas
python run_complete_training.py --model_size s --epochs 200

# Entrenamiento con seguimiento W&B
python run_complete_training.py --use_wandb
```

### 2. Solo preparar dataset (sin entrenar)

```bash
python run_complete_training.py --skip_training
```

### 3. Entrenar con dataset existente

```bash
python run_complete_training.py --skip_conversion
```

## 📁 Estructura del Proyecto

```
yolo_training/
├── run_complete_training.py  # Script maestro
├── coco_to_yolo.py          # Conversión COCO → YOLO
├── dataset_split.py         # División train/val/test
├── train_yolo.py           # Entrenamiento YOLO
├── inference.py            # Inferencia y evaluación
├── taco_dataset.yaml       # Configuración del dataset
├── requirements_yolo.txt   # Dependencias
└── README.md              # Esta documentación

# Directorios generados
yolo_dataset/              # Dataset en formato YOLO
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

runs/                      # Resultados de entrenamiento
├── train/
├── val/
└── predict/
```

## 🛠️ Uso Detallado

### Conversión de Dataset

```bash
# Convertir manualmente COCO a YOLO
python coco_to_yolo.py \
  --annotations ../data/annotations.json \
  --images ../data \
  --output yolo_dataset \
  --mapping ../detector/taco_config/map_17.csv
```

### División del Dataset

```bash
# Dividir dataset en 80% train, 10% val, 10% test
python dataset_split.py \
  --dataset yolo_dataset \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

### Entrenamiento Personalizado

```bash
# Entrenamiento básico
python train_yolo.py --model_size n --epochs 100

# Entrenamiento avanzado
python train_yolo.py \
  --model_size s \
  --epochs 200 \
  --batch_size 32 \
  --img_size 640 \
  --lr0 0.01 \
  --patience 50 \
  --use_wandb
```

### Inferencia y Evaluación

```bash
# Predicción en imagen individual
python inference.py \
  --model runs/train/yolon_*/weights/best.pt \
  --mode single \
  --image ruta/a/imagen.jpg

# Predicción en lote con análisis
python inference.py \
  --model runs/train/yolon_*/weights/best.pt \
  --mode batch \
  --images_dir ruta/a/imagenes/ \
  --analyze \
  --output_json resultados.json

# Evaluación en dataset de prueba
python inference.py \
  --model runs/train/yolon_*/weights/best.pt \
  --mode evaluate \
  --test_data taco_dataset.yaml
```

## 📊 Clases del Dataset

El sistema usa el mapeo de 17 clases principales de TACO:

| ID  | Clase              | Descripción              |
| --- | ------------------ | ------------------------ |
| 0   | Aluminium foil     | Papel de aluminio        |
| 1   | Can                | Latas                    |
| 2   | Carton             | Cartón                   |
| 3   | Cup                | Vasos                    |
| 4   | Glass bottle       | Botellas de vidrio       |
| 5   | Metal bottle cap   | Tapas metálicas          |
| 6   | Other              | Otros                    |
| 7   | Paper              | Papel                    |
| 8   | Plastic bottle     | Botellas de plástico     |
| 9   | Plastic bottle cap | Tapas de plástico        |
| 10  | Plastic container  | Contenedores de plástico |
| 11  | Plastic film       | Películas de plástico    |
| 12  | Plastic lid        | Tapas de plástico        |
| 13  | Pop tab            | Anillas de latas         |
| 14  | Straw              | Pajillas                 |
| 15  | Styrofoam piece    | Espuma de poliestireno   |
| 16  | Wrapper            | Envolturas               |

## ⚙️ Configuración Avanzada

### Hiperparámetros

```python
# Optimización
--lr0 0.01              # Learning rate inicial
--momentum 0.937        # Momentum
--weight_decay 0.0005   # Regularización L2
--patience 30           # Early stopping

# Augmentación
--augment True          # Activar augmentación
--mosaic 1.0           # Probabilidad de mosaic
--translate 0.1        # Translación
--scale 0.5            # Escalado
--fliplr 0.5           # Flip horizontal
```

### Modelos Disponibles

| Modelo  | Tamaño | Parámetros | Velocidad | Precisión  |
| ------- | ------ | ---------- | --------- | ---------- |
| YOLOv8n | Nano   | 3.2M       | ⚡⚡⚡    | ⭐⭐       |
| YOLOv8s | Small  | 11.2M      | ⚡⚡      | ⭐⭐⭐     |
| YOLOv8m | Medium | 25.9M      | ⚡        | ⭐⭐⭐⭐   |
| YOLOv8l | Large  | 43.7M      | 💨        | ⭐⭐⭐⭐⭐ |
| YOLOv8x | XLarge | 68.2M      | 🐌        | ⭐⭐⭐⭐⭐ |

## 📈 Métricas de Evaluación

- **mAP@0.5**: Precisión media a IoU 0.5
- **mAP@0.5:0.95**: Precisión media promedio
- **Precision**: Precisión por clase
- **Recall**: Exhaustividad por clase
- **F1-Score**: Media armónica de precisión y recall

## 🔧 Solución de Problemas

### Error: CUDA out of memory

```bash
# Reducir batch size
--batch_size 8

# Reducir tamaño de imagen
--img_size 416

# Usar modelo más pequeño
--model_size n
```

### Error: Archivos no encontrados

```bash
# Verificar rutas
ls ../data/annotations.json
ls ../data/batch_*

# Usar rutas absolutas
--annotations /ruta/completa/annotations.json
```

### Rendimiento lento

```bash
# Usar menos workers
--workers 4

# Desactivar augmentación
--augment False

# Usar GPU
--device cuda
```

## 📚 Ejemplos de Uso

### Entrenamiento rápido para pruebas

```bash
python run_complete_training.py \
  --model_size n \
  --epochs 10 \
  --batch_size 8 \
  --img_size 416
```

### Entrenamiento de producción

```bash
python run_complete_training.py \
  --model_size m \
  --epochs 300 \
  --batch_size 16 \
  --img_size 640 \
  --use_wandb \
  --patience 50
```

### Solo preparar datos

```bash
python run_complete_training.py \
  --skip_training \
  --train_ratio 0.7 \
  --val_ratio 0.2 \
  --test_ratio 0.1
```

## 📊 Seguimiento con Weights & Biases

```bash
# Configurar W&B
wandb login

# Entrenar con seguimiento
python run_complete_training.py --use_wandb
```

Visualiza métricas en tiempo real:

- Pérdida de entrenamiento y validación
- Métricas de precisión (mAP, precision, recall)
- Imágenes de validación con predicciones
- Gráficos de distribución de clases

## 🎯 Mejores Prácticas

1. **Empezar con modelo pequeño**: Usa YOLOv8n para pruebas rápidas
2. **Validar datos**: Verifica que las conversiones sean correctas
3. **Monitorear overfitting**: Usa early stopping y validation loss
4. **Experimentar con augmentación**: Ajusta según el rendimiento
5. **Evaluar en test set**: Usa datos nunca vistos para evaluación final

## 🔄 Flujo de Trabajo Recomendado

1. **Preparación**: `run_complete_training.py --skip_training`
2. **Entrenamiento rápido**: Modelo nano, pocas épocas
3. **Evaluación inicial**: Verificar que funciona
4. **Entrenamiento completo**: Modelo más grande, más épocas
5. **Evaluación final**: Test set y análisis de resultados
6. **Optimización**: Ajustar hiperparámetros si es necesario

## 📞 Soporte

Para problemas o dudas:

1. Revisar logs de error
2. Verificar configuración de GPU/CUDA
3. Consultar documentación de Ultralytics
4. Verificar versiones de dependencias
