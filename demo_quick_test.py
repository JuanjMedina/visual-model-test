#!/usr/bin/env python3
"""
Demo rápido del sistema TACO-YOLO
Ejecuta una prueba completa con configuración mínima para verificar funcionamiento
"""

import os
import sys
import argparse
from pathlib import Path
import time
import subprocess

def print_banner():
    """Mostrar banner del demo"""
    print("=" * 60)
    print("🎯 TACO-YOLO: DEMO RÁPIDO")
    print("=" * 60)
    print("Este demo ejecuta una prueba completa del sistema con:")
    print("• Conversión COCO → YOLO")
    print("• División del dataset")
    print("• Entrenamiento rápido (5 épocas)")
    print("• Evaluación básica")
    print("=" * 60)

def check_prerequisites():
    """Verificar prerequisitos básicos"""
    print("🔍 Verificando prerequisitos...")
    
    # Verificar archivos necesarios
    required_files = [
        '../data/annotations.json',
        '../detector/taco_config/map_17.csv'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Archivo requerido no encontrado: {file_path}")
            return False
        print(f"✅ {file_path}")
    
    # Verificar que hay al menos un directorio batch_
    data_dir = Path('../data')
    batch_dirs = list(data_dir.glob('batch_*'))
    if not batch_dirs:
        print("❌ No se encontraron directorios batch_* en ../data")
        return False
    
    print(f"✅ Encontrados {len(batch_dirs)} directorios batch_*")
    return True

def install_minimal_dependencies():
    """Instalar dependencias mínimas"""
    print("📦 Instalando dependencias mínimas...")
    
    minimal_deps = [
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'matplotlib',
        'pyyaml'
    ]
    
    try:
        for dep in minimal_deps:
            try:
                __import__(dep)
                print(f"✅ {dep} ya instalado")
            except ImportError:
                print(f"📦 Instalando {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                print(f"✅ {dep} instalado")
        return True
    except Exception as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def run_demo():
    """Ejecutar demo completo"""
    print("\n🚀 Iniciando demo...")
    
    # Parámetros del demo
    demo_params = {
        'model_size': 'n',        # Modelo más pequeño
        'epochs': 5,              # Pocas épocas para prueba rápida
        'batch_size': 8,          # Batch pequeño
        'img_size': 416,          # Tamaño de imagen reducido
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    }
    
    print("⚙️ Configuración del demo:")
    for key, value in demo_params.items():
        print(f"  {key}: {value}")
    
    # Comando para ejecutar el entrenamiento completo
    cmd = [
        sys.executable,
        'run_complete_training.py',
        '--model_size', demo_params['model_size'],
        '--epochs', str(demo_params['epochs']),
        '--batch_size', str(demo_params['batch_size']),
        '--img_size', str(demo_params['img_size']),
        '--train_ratio', str(demo_params['train_ratio']),
        '--val_ratio', str(demo_params['val_ratio']),
        '--test_ratio', str(demo_params['test_ratio']),
        '--install_deps'
    ]
    
    print(f"\n🔄 Ejecutando comando:")
    print(' '.join(cmd))
    
    start_time = time.time()
    
    try:
        # Ejecutar el comando
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Demo completado exitosamente!")
        print(f"⏱️ Tiempo total: {duration/60:.2f} minutos")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando el demo:")
        print(f"Código de error: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Demo interrumpido por el usuario")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return False

def show_results():
    """Mostrar resultados del demo"""
    print("\n📊 Revisando resultados...")
    
    # Verificar directorios creados
    dirs_to_check = [
        'yolo_dataset',
        'yolo_dataset/train',
        'yolo_dataset/val',
        'yolo_dataset/test',
        'runs'
    ]
    
    for dir_path in dirs_to_check:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} creado")
        else:
            print(f"❌ {dir_path} no encontrado")
    
    # Buscar modelos entrenados
    runs_dir = Path('runs/train')
    if runs_dir.exists():
        model_dirs = list(runs_dir.glob('yolo*'))
        if model_dirs:
            print(f"✅ {len(model_dirs)} experimento(s) de entrenamiento encontrado(s)")
            for model_dir in model_dirs:
                weights_dir = model_dir / 'weights'
                if weights_dir.exists():
                    best_model = weights_dir / 'best.pt'
                    last_model = weights_dir / 'last.pt'
                    if best_model.exists():
                        print(f"  📁 Mejor modelo: {best_model}")
                    if last_model.exists():
                        print(f"  📁 Último modelo: {last_model}")
        else:
            print("❌ No se encontraron modelos entrenados")

def main():
    parser = argparse.ArgumentParser(description='Demo rápido del sistema TACO-YOLO')
    parser.add_argument('--skip_install', action='store_true',
                        help='Omitir instalación de dependencias')
    parser.add_argument('--only_prep', action='store_true',
                        help='Solo preparar dataset, no entrenar')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Verificar prerequisitos
    if not check_prerequisites():
        print("\n❌ Faltan prerequisitos necesarios")
        return 1
    
    # Instalar dependencias si es necesario
    if not args.skip_install:
        if not install_minimal_dependencies():
            print("\n❌ Error instalando dependencias")
            return 1
    
    # Ejecutar demo
    if args.only_prep:
        print("\n🔄 Modo: Solo preparación de datos")
        # Aquí podrías agregar lógica para solo preparar datos
    else:
        if not run_demo():
            print("\n❌ El demo no se completó exitosamente")
            return 1
    
    # Mostrar resultados
    show_results()
    
    print("\n" + "=" * 60)
    print("🎉 ¡DEMO COMPLETADO!")
    print("=" * 60)
    print("\n📋 Próximos pasos:")
    print("1. Revisar resultados en la carpeta 'runs/'")
    print("2. Probar inferencia con el modelo entrenado")
    print("3. Experimentar con diferentes configuraciones")
    print("4. Ejecutar entrenamiento completo para mejores resultados")
    
    print("\n💡 Comandos útiles:")
    print("• Entrenamiento completo: python run_complete_training.py")
    print("• Inferencia: python inference.py --model runs/train/yolo*/weights/best.pt --mode single --image imagen.jpg")
    print("• Evaluación: python inference.py --model runs/train/yolo*/weights/best.pt --mode evaluate --test_data taco_dataset.yaml")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 