#!/usr/bin/env python3
"""
Script para inferencia y evaluación de modelos YOLO entrenados con dataset TACO
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_model(model_path):
    """Cargar modelo YOLO entrenado"""
    print(f"🔄 Cargando modelo: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return None
    
    model = YOLO(model_path)
    print(f"✅ Modelo cargado exitosamente")
    return model

def predict_single_image(model, image_path, conf_threshold=0.25, save_dir=None):
    """Realizar predicción en una sola imagen"""
    if not Path(image_path).exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return None
    
    # Realizar predicción
    results = model(image_path, conf=conf_threshold)
    
    # Procesar resultados
    result = results[0]
    
    # Información de la predicción
    prediction_info = {
        'image_path': image_path,
        'detections': [],
        'confidence_scores': [],
        'class_names': [],
        'bounding_boxes': []
    }
    
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            class_name = model.names[int(cls)]
            prediction_info['detections'].append({
                'bbox': box.tolist(),
                'confidence': float(conf),
                'class': class_name,
                'class_id': int(cls)
            })
            prediction_info['confidence_scores'].append(float(conf))
            prediction_info['class_names'].append(class_name)
            prediction_info['bounding_boxes'].append(box.tolist())
    
    # Guardar imagen con predicciones si se especifica
    if save_dir:
        save_path = Path(save_dir) / f"prediction_{Path(image_path).name}"
        annotated_image = result.plot()
        cv2.imwrite(str(save_path), annotated_image)
        print(f"💾 Imagen guardada: {save_path}")
    
    return prediction_info

def predict_batch(model, images_dir, conf_threshold=0.25, save_dir=None):
    """Realizar predicciones en lote"""
    print(f"🔄 Procesando imágenes en: {images_dir}")
    
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"❌ Directorio no encontrado: {images_dir}")
        return []
    
    # Obtener lista de imágenes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))
    
    print(f"📸 Encontradas {len(image_files)} imágenes")
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Procesar cada imagen
    all_predictions = []
    
    for i, image_file in enumerate(image_files):
        print(f"🔍 Procesando {i+1}/{len(image_files)}: {image_file.name}")
        
        prediction = predict_single_image(model, str(image_file), conf_threshold, save_dir)
        if prediction:
            all_predictions.append(prediction)
    
    return all_predictions

def evaluate_model(model, test_dataset_path, conf_threshold=0.25):
    """Evaluar modelo en dataset de prueba"""
    print(f"📊 Evaluando modelo en: {test_dataset_path}")
    
    # Evaluar usando ultralytics
    results = model.val(data=test_dataset_path, conf=conf_threshold)
    
    # Extraer métricas
    metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'f1_score': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) if (results.box.mp + results.box.mr) > 0 else 0
    }
    
    print("\n📈 Métricas de evaluación:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics, results

def analyze_predictions(predictions, output_dir=None):
    """Analizar predicciones y generar estadísticas"""
    print("📊 Analizando predicciones...")
    
    if not predictions:
        print("❌ No hay predicciones para analizar")
        return
    
    # Recopilar estadísticas
    total_detections = sum(len(pred['detections']) for pred in predictions)
    images_with_detections = sum(1 for pred in predictions if len(pred['detections']) > 0)
    
    # Contar clases
    class_counts = {}
    confidence_scores = []
    
    for pred in predictions:
        for detection in pred['detections']:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(detection['confidence'])
    
    # Mostrar estadísticas
    print(f"\n📋 Estadísticas generales:")
    print(f"  Total de imágenes: {len(predictions)}")
    print(f"  Imágenes con detecciones: {images_with_detections}")
    print(f"  Total de detecciones: {total_detections}")
    print(f"  Promedio de detecciones por imagen: {total_detections/len(predictions):.2f}")
    
    if confidence_scores:
        print(f"  Confianza promedio: {np.mean(confidence_scores):.3f}")
        print(f"  Confianza mínima: {np.min(confidence_scores):.3f}")
        print(f"  Confianza máxima: {np.max(confidence_scores):.3f}")
    
    print(f"\n🏷️  Detecciones por clase:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    # Crear gráficos si se especifica directorio de salida
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Gráfico de distribución de clases
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts)
        plt.title('Distribución de Clases Detectadas')
        plt.xlabel('Clase')
        plt.ylabel('Número de Detecciones')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Histograma de confianza
        if confidence_scores:
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribución de Puntuaciones de Confianza')
            plt.xlabel('Confianza')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Gráficos guardados en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Inferencia y evaluación de modelos YOLO-TACO')
    
    # Parámetros principales
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al modelo YOLO entrenado (.pt)')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate'], default='single',
                        help='Modo de operación')
    
    # Parámetros para inferencia
    parser.add_argument('--image', type=str,
                        help='Ruta a imagen individual (modo single)')
    parser.add_argument('--images_dir', type=str,
                        help='Directorio con imágenes (modo batch)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Umbral de confianza para detecciones')
    parser.add_argument('--save_dir', type=str, default='runs/predict',
                        help='Directorio para guardar resultados')
    
    # Parámetros para evaluación
    parser.add_argument('--test_data', type=str,
                        help='Archivo YAML de configuración para evaluación')
    
    # Opciones adicionales
    parser.add_argument('--analyze', action='store_true',
                        help='Realizar análisis estadístico de predicciones')
    parser.add_argument('--output_json', type=str,
                        help='Guardar predicciones en archivo JSON')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model = load_model(args.model)
    if model is None:
        return 1
    
    # Crear directorio de salida
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Modo: {args.mode}")
    print(f"📁 Directorio de salida: {args.save_dir}")
    print(f"🎚️  Umbral de confianza: {args.conf_threshold}")
    
    # Ejecutar según el modo
    if args.mode == 'single':
        if not args.image:
            print("❌ Error: Se requiere --image para modo single")
            return 1
        
        print("\n🔍 PREDICCIÓN EN IMAGEN INDIVIDUAL")
        print("="*40)
        
        prediction = predict_single_image(model, args.image, args.conf_threshold, args.save_dir)
        
        if prediction:
            print(f"✅ Predicción completada")
            print(f"📸 Imagen: {prediction['image_path']}")
            print(f"🔍 Detecciones: {len(prediction['detections'])}")
            
            for i, detection in enumerate(prediction['detections']):
                print(f"  {i+1}. {detection['class']} (confianza: {detection['confidence']:.3f})")
            
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(prediction, f, indent=2)
                print(f"💾 Predicción guardada en: {args.output_json}")
    
    elif args.mode == 'batch':
        if not args.images_dir:
            print("❌ Error: Se requiere --images_dir para modo batch")
            return 1
        
        print("\n🔍 PREDICCIÓN EN LOTE")
        print("="*40)
        
        predictions = predict_batch(model, args.images_dir, args.conf_threshold, args.save_dir)
        
        if predictions:
            print(f"✅ Predicciones completadas: {len(predictions)} imágenes")
            
            if args.analyze:
                analyze_predictions(predictions, args.save_dir)
            
            if args.output_json:
                with open(args.output_json, 'w') as f:
                    json.dump(predictions, f, indent=2)
                print(f"💾 Predicciones guardadas en: {args.output_json}")
    
    elif args.mode == 'evaluate':
        if not args.test_data:
            print("❌ Error: Se requiere --test_data para modo evaluate")
            return 1
        
        print("\n📊 EVALUACIÓN DEL MODELO")
        print("="*40)
        
        metrics, results = evaluate_model(model, args.test_data, args.conf_threshold)
        
        # Guardar métricas
        metrics_file = Path(args.save_dir) / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Métricas guardadas en: {metrics_file}")
    
    print("\n🎉 ¡Proceso completado exitosamente!")
    
    return 0

if __name__ == "__main__":
    main() 