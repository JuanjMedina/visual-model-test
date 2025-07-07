#!/usr/bin/env python3
"""
Script maestro para automatizar todo el proceso de entrenamiento YOLO con dataset TACO
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import yaml

def run_command(cmd, description=""):
    """Ejecutar comando del sistema con manejo de errores"""
    print(f"\n🔄 {description}")
    print(f"Ejecutando: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado exitosamente")
        if result.stdout:
            print(f"Salida: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"Código de error: {e.returncode}")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Verificar que todas las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'matplotlib',
        'pandas',
        'numpy',
        'pyyaml',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} instalado")
        except ImportError:
            print(f"❌ {package} NO encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Paquetes faltantes: {missing_packages}")
        print("Instalando dependencias...")
        
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Instalando dependencias"):
            return False
    
    return True

def setup_directories(base_dir):
    """Crear estructura de directorios necesaria"""
    print("📁 Configurando directorios...")
    
    directories = [
        'yolo_dataset',
        'yolo_dataset/images',
        'yolo_dataset/labels',
        'yolo_dataset/train',
        'yolo_dataset/val',
        'yolo_dataset/test',
        'yolo_dataset/train/images',
        'yolo_dataset/train/labels',
        'yolo_dataset/val/images',
        'yolo_dataset/val/labels',
        'yolo_dataset/test/images',
        'yolo_dataset/test/labels',
        'runs',
        'runs/train',
        'runs/val',
        'runs/predict'
    ]
    
    for dir_name in directories:
        dir_path = Path(base_dir) / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 {dir_path}")
    
    return True

def convert_dataset(annotations_file, images_dir, output_dir, mapping_file):
    """Convertir dataset COCO a YOLO"""
    print("🔄 Convirtiendo dataset COCO a YOLO...")
    
    script_path = Path(__file__).parent / "coco_to_yolo.py"
    cmd = f"python {script_path} --annotations {annotations_file} --images {images_dir} --output {output_dir} --mapping {mapping_file}"
    
    return run_command(cmd, "Conversión COCO a YOLO")

def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Dividir dataset en train/val/test"""
    print("🔄 Dividiendo dataset en train/val/test...")
    
    script_path = Path(__file__).parent / "dataset_split.py"
    cmd = f"python {script_path} --dataset {dataset_dir} --train_ratio {train_ratio} --val_ratio {val_ratio} --test_ratio {test_ratio}"
    
    return run_command(cmd, "División del dataset")

def update_yaml_config(yaml_file, dataset_path):
    """Actualizar archivo YAML con rutas correctas"""
    print("📝 Actualizando configuración YAML...")
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Actualizar rutas
    config['path'] = str(Path(dataset_path).absolute())
    config['train'] = 'train/images'
    config['val'] = 'val/images'
    config['test'] = 'test/images'
    
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuración actualizada en {yaml_file}")
    return True

def train_model(model_size, dataset_yaml, epochs, batch_size, img_size, device, save_dir, use_wandb=False):
    """Entrenar modelo YOLO"""
    print("🚀 Iniciando entrenamiento del modelo...")
    
    script_path = Path(__file__).parent / "train_yolo.py"
    cmd = f"python {script_path} --model_size {model_size} --dataset {dataset_yaml} --epochs {epochs} --batch_size {batch_size} --img_size {img_size} --device {device} --save_dir {save_dir}"
    
    if use_wandb:
        cmd += " --use_wandb"
    
    return run_command(cmd, "Entrenamiento del modelo")

def validate_files(annotations_file, images_dir, mapping_file):
    """Validar que los archivos necesarios existan"""
    print("🔍 Validando archivos necesarios...")
    
    files_to_check = [
        (annotations_file, "Archivo de anotaciones"),
        (images_dir, "Directorio de imágenes"),
        (mapping_file, "Archivo de mapeo de clases")
    ]
    
    for file_path, description in files_to_check:
        if not Path(file_path).exists():
            print(f"❌ {description} no encontrado: {file_path}")
            return False
        print(f"✅ {description}: {file_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Script maestro para entrenamiento YOLO con dataset TACO')
    
    # Rutas de archivos
    parser.add_argument('--annotations', type=str, default='../data/annotations.json',
                        help='Archivo JSON de anotaciones COCO')
    parser.add_argument('--images', type=str, default='../data',
                        help='Directorio con imágenes (contiene batch_*)')
    parser.add_argument('--mapping', type=str, default='../detector/taco_config/map_17.csv',
                        help='Archivo CSV de mapeo de clases')
    parser.add_argument('--output', type=str, default='yolo_dataset',
                        help='Directorio de salida para dataset YOLO')
    
    # Parámetros de división del dataset
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proporción para entrenamiento')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Proporción para validación')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Proporción para prueba')
    
    # Parámetros de entrenamiento
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Tamaño del modelo YOLO')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño del batch')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Tamaño de imagen')
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='runs/train',
                        help='Directorio para guardar resultados')
    
    # Opciones adicionales
    parser.add_argument('--use_wandb', action='store_true',
                        help='Usar Weights & Biases')
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Omitir conversión (usar dataset existente)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Solo preparar dataset, no entrenar')
    parser.add_argument('--install_deps', action='store_true',
                        help='Instalar dependencias automáticamente')
    
    args = parser.parse_args()
    
    print("🎯 TACO-YOLO: Entrenamiento Automatizado")
    print("=" * 50)
    
    # Instalar dependencias si se requiere
    if args.install_deps:
        if not check_dependencies():
            print("❌ Error instalando dependencias")
            return 1
    
    # Validar archivos de entrada
    if not args.skip_conversion:
        if not validate_files(args.annotations, args.images, args.mapping):
            print("❌ Error: Archivos necesarios no encontrados")
            return 1
    
    # Configurar directorios
    if not setup_directories('.'):
        print("❌ Error configurando directorios")
        return 1
    
    # Paso 1: Convertir dataset COCO a YOLO
    if not args.skip_conversion:
        print("\n" + "="*50)
        print("PASO 1: CONVERSIÓN COCO A YOLO")
        print("="*50)
        
        if not convert_dataset(args.annotations, args.images, args.output, args.mapping):
            print("❌ Error en conversión del dataset")
            return 1
    
    # Paso 2: Dividir dataset
    print("\n" + "="*50)
    print("PASO 2: DIVISIÓN DEL DATASET")
    print("="*50)
    
    if not split_dataset(args.output, args.train_ratio, args.val_ratio, args.test_ratio):
        print("❌ Error dividiendo el dataset")
        return 1
    
    # Paso 3: Actualizar configuración YAML
    print("\n" + "="*50)
    print("PASO 3: CONFIGURACIÓN YAML")
    print("="*50)
    
    yaml_file = "taco_dataset.yaml"
    if not update_yaml_config(yaml_file, args.output):
        print("❌ Error actualizando configuración YAML")
        return 1
    
    # Paso 4: Entrenar modelo (opcional)
    if not args.skip_training:
        print("\n" + "="*50)
        print("PASO 4: ENTRENAMIENTO DEL MODELO")
        print("="*50)
        
        start_time = time.time()
        
        if not train_model(
            args.model_size, 
            yaml_file, 
            args.epochs, 
            args.batch_size, 
            args.img_size, 
            args.device, 
            args.save_dir, 
            args.use_wandb
        ):
            print("❌ Error en entrenamiento del modelo")
            return 1
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n⏱️  Tiempo total de entrenamiento: {training_time/3600:.2f} horas")
    
    print("\n" + "="*50)
    print("🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("="*50)
    
    print("\n📋 Resumen:")
    print(f"• Dataset convertido y dividido en: {args.output}")
    print(f"• Configuración YAML: {yaml_file}")
    if not args.skip_training:
        print(f"• Modelos entrenados en: {args.save_dir}")
    
    print("\n🚀 Próximos pasos:")
    print("1. Revisar las métricas de entrenamiento")
    print("2. Evaluar el modelo en el conjunto de prueba")
    print("3. Realizar inferencia en nuevas imágenes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 