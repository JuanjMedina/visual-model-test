import os
import shutil
import argparse
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def split_yolo_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Dividir dataset YOLO en train/val/test"""
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # Verificar que existan los directorios
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Directorios 'images' y 'labels' no encontrados")
    
    # Obtener lista de imágenes
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    # Verificar que cada imagen tenga su archivo de etiqueta correspondiente
    valid_files = []
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + '.txt')
        if label_file.exists():
            valid_files.append(img_file.name)
    
    print(f"Archivos válidos encontrados: {len(valid_files)}")
    
    # Dividir los archivos
    random.shuffle(valid_files)
    
    # Calcular tamaños de los conjuntos
    total_files = len(valid_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size
    
    train_files = valid_files[:train_size]
    val_files = valid_files[train_size:train_size + val_size]
    test_files = valid_files[train_size + val_size:]
    
    print(f"División del dataset:")
    print(f"- Entrenamiento: {len(train_files)} archivos")
    print(f"- Validación: {len(val_files)} archivos")
    print(f"- Prueba: {len(test_files)} archivos")
    
    # Crear directorios de salida
    splits = ['train', 'val', 'test']
    for split in splits:
        (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Mover archivos a sus respectivos directorios
    file_sets = [
        (train_files, 'train'),
        (val_files, 'val'),
        (test_files, 'test')
    ]
    
    for files, split in file_sets:
        for filename in files:
            # Copiar imagen
            src_img = images_dir / filename
            dst_img = dataset_path / split / 'images' / filename
            shutil.copy2(src_img, dst_img)
            
            # Copiar etiqueta
            label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
            src_label = labels_dir / label_filename
            dst_label = dataset_path / split / 'labels' / label_filename
            shutil.copy2(src_label, dst_label)
    
    print("División completada exitosamente!")
    return len(train_files), len(val_files), len(test_files)

def main():
    parser = argparse.ArgumentParser(description='Dividir dataset YOLO en train/val/test')
    parser.add_argument('--dataset', required=True, help='Directorio del dataset YOLO')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Proporción para entrenamiento')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Proporción para validación')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Proporción para prueba')
    
    args = parser.parse_args()
    
    # Verificar que las proporciones sumen 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        raise ValueError("Las proporciones deben sumar 1.0")
    
    split_yolo_dataset(args.dataset, args.train_ratio, args.val_ratio, args.test_ratio)

if __name__ == "__main__":
    main() 