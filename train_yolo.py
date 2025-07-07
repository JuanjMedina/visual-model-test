import os
import argparse
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Importar wandb de forma opcional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb no está instalado. Entrenamiento sin seguimiento W&B.")

def setup_wandb(project_name="TACO-YOLO", run_name=None):
    """Configurar Weights & Biases para seguimiento de experimentos"""
    if not WANDB_AVAILABLE:
        print("⚠️ wandb no disponible. Omitiendo configuración.")
        return None
        
    if run_name is None:
        run_name = f"yolo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            save_code=True
        )
        return run_name
    except Exception as e:
        print(f"⚠️ Error configurando wandb: {e}")
        return None

def train_yolo_model(
    model_size='n',
    dataset_yaml='taco_dataset.yaml',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='auto',
    save_dir='runs/train',
    use_wandb=False,
    pretrained=True,
    resume=False,
    augmentation=True,
    workers=8,
    patience=30,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box_loss_gain=0.05,
    cls_loss_gain=0.5,
    dfl_loss_gain=1.5,
    freeze_layers=None
):
    """Entrenar modelo YOLO con configuración personalizada"""
    
    # Configurar dispositivo
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Utilizando dispositivo: {device}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar modelo
    model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
    print(f"Cargando modelo: {model_name}")
    
    model = YOLO(model_name)
    
    # Configurar Weights & Biases si se requiere
    if use_wandb and WANDB_AVAILABLE:
        run_name = setup_wandb()
        if run_name:
            print(f"Seguimiento W&B habilitado: {run_name}")
        else:
            print("⚠️ No se pudo configurar W&B, continuando sin seguimiento")
            use_wandb = False
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠️ W&B solicitado pero no disponible, continuando sin seguimiento")
        use_wandb = False
    
    # Congelar capas si se especifica
    if freeze_layers:
        print(f"Congelando primeras {freeze_layers} capas")
        for i, param in enumerate(model.model.parameters()):
            if i < freeze_layers:
                param.requires_grad = False
    
    # Parámetros de entrenamiento (solo los válidos)
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': save_dir,
        'name': f'yolo{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'resume': resume,
        'workers': workers,
        'patience': patience,
        'save_period': 10,  # Guardar cada 10 épocas
        'save': True,
        'plots': True,
        'val': True,
        'verbose': True,
        'seed': 42,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'dropout': 0.0,
        'cache': False,
        'copy_paste': 0.0,
        'mixup': 0.0,
        'mosaic': 1.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'degrees': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'lr0': lr0,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,
        'warmup_bias_lr': warmup_bias_lr,
    }

    
    print("Configuración de entrenamiento:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Iniciar entrenamiento
    print("\n🚀 Iniciando entrenamiento...")
    results = model.train(**train_args)
    
    # Validar modelo
    print("\n📊 Validando modelo...")
    val_results = model.val()
    
    # Guardar métricas
    metrics = {
        'map50': val_results.box.map50,
        'map50_95': val_results.box.map,
        'precision': val_results.box.mp,
        'recall': val_results.box.mr,
        'train_loss': results.results_dict.get('train/box_loss', 0),
        'val_loss': results.results_dict.get('val/box_loss', 0),
    }
    
    print("\n✅ Métricas finales:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Cerrar W&B si se está usando
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.finish()
        except:
            pass
    
    return model, results, metrics

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo YOLO con dataset TACO')
    
    # Parámetros del modelo
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Tamaño del modelo YOLO (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--dataset', type=str, default='taco_dataset.yaml',
                        help='Archivo YAML de configuración del dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Tamaño del batch')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Tamaño de imagen de entrada')
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo para entrenamiento (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='runs/train',
                        help='Directorio para guardar resultados')
    
    # Parámetros de optimización
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Learning rate inicial')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='Momentum del optimizador')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=30,
                        help='Paciencia para early stopping')
    parser.add_argument('--workers', type=int, default=8,
                        help='Número de workers para carga de datos')
    
    # Parámetros adicionales
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Usar modelo preentrenado')
    parser.add_argument('--resume', action='store_true',
                        help='Resumir entrenamiento desde checkpoint')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Usar Weights & Biases para seguimiento')
    parser.add_argument('--freeze_layers', type=int, default=None,
                        help='Número de capas a congelar')
    
    args = parser.parse_args()
    
    # Verificar que el archivo de dataset existe
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Archivo de dataset no encontrado: {args.dataset}")
        return
    
    print("🎯 Configuración de entrenamiento YOLO-TACO")
    print(f"Modelo: YOLOv8{args.model_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tamaño de imagen: {args.img_size}")
    print(f"Dispositivo: {args.device}")
    print("-" * 50)
    
    # Entrenar modelo
    model, results, metrics = train_yolo_model(
        model_size=args.model_size,
        dataset_yaml=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        pretrained=args.pretrained,
        resume=args.resume,
        workers=args.workers,
        patience=args.patience,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        freeze_layers=args.freeze_layers
    )
    
    print("\n🎉 ¡Entrenamiento completado exitosamente!")
    print(f"Modelo guardado en: {args.save_dir}")
    
    # Mostrar mejores métricas
    print("\n📈 Mejores métricas:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main() 