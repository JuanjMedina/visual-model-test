import json
import os
import argparse
from pathlib import Path
import csv
import shutil
from PIL import Image
import numpy as np

def load_class_mapping(mapping_file):
    """Cargar mapeo de clases desde archivo CSV"""
    class_mapping = {}
    class_names = []
    
    with open(mapping_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                original_class = row[0].strip()
                mapped_class = row[1].strip()
                
                if mapped_class not in class_names:
                    class_names.append(mapped_class)
                
                class_mapping[original_class] = class_names.index(mapped_class)
    
    return class_mapping, class_names

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """Convertir bounding box de formato COCO a YOLO"""
    x, y, width, height = bbox
    
    # Normalizar coordenadas
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    return [x_center, y_center, norm_width, norm_height]

def coco_to_yolo(annotations_file, images_dir, output_dir, class_mapping_file):
    """Convertir dataset COCO a formato YOLO"""
    
    # Cargar mapeo de clases
    class_mapping, class_names = load_class_mapping(class_mapping_file)
    
    # Cargar anotaciones COCO
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Crear directorios de salida
    output_path = Path(output_dir)
    labels_dir = output_path / 'labels'
    images_out_dir = output_path / 'images'
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear diccionario de categorías COCO
    categories = {}
    for cat in coco_data['categories']:
        categories[cat['id']] = cat['name']
    
    # Procesar cada imagen
    converted_images = 0
    skipped_images = 0
    
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Buscar archivo de imagen en todos los subdirectorios
        img_path = None
        
        # Crear variaciones del nombre de archivo para diferentes extensiones
        base_name = Path(img_filename).stem
        possible_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
        possible_filenames = [base_name + ext for ext in possible_extensions]
        
        # Buscar en directorios batch_*
        for batch_dir in Path(images_dir).glob('batch_*'):
            for filename in possible_filenames:
                potential_path = batch_dir / filename
                if potential_path.exists():
                    img_path = potential_path
                    break
            if img_path:
                break
        
        # Si no se encuentra, buscar recursivamente en todos los subdirectorios
        if not img_path:
            for root, dirs, files in os.walk(images_dir):
                # Buscar coincidencia insensible a mayúsculas/minúsculas
                files_lower = [f.lower() for f in files]
                img_filename_lower = img_filename.lower()
                
                if img_filename_lower in files_lower:
                    # Encontrar el archivo original con la capitalización correcta
                    original_filename = files[files_lower.index(img_filename_lower)]
                    img_path = Path(root) / original_filename
                    break
                
                # También buscar por variaciones de extensión
                for filename in possible_filenames:
                    if filename.lower() in files_lower:
                        original_filename = files[files_lower.index(filename.lower())]
                        img_path = Path(root) / original_filename
                        break
                
                if img_path:
                    break
        
        if not img_path:
            # Solo mostrar advertencia cada 100 imágenes para no saturar la salida
            if skipped_images % 100 == 0:
                print(f"Imagen no encontrada: {img_filename} (omitidas hasta ahora: {skipped_images})")
            skipped_images += 1
            continue
        
        # Copiar imagen al directorio de salida (usar el nombre original del archivo encontrado)
        actual_filename = img_path.name
        # Normalizar la extensión para consistencia (usar .jpg como estándar)
        normalized_filename = Path(img_filename).stem + '.jpg'
        shutil.copy2(img_path, images_out_dir / normalized_filename)
        
        # Obtener anotaciones para esta imagen
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        # Crear archivo de etiquetas YOLO (usar el nombre normalizado)
        label_filename = normalized_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = labels_dir / label_filename
        
        yolo_annotations = []
        
        for ann in img_annotations:
            category_id = ann['category_id']
            category_name = categories.get(category_id, '')
            
            # Verificar si la categoría está en el mapeo
            if category_name in class_mapping:
                class_id = class_mapping[category_name]
                bbox = ann['bbox']
                
                # Convertir bbox a formato YOLO
                yolo_bbox = convert_bbox_coco_to_yolo(bbox, img_width, img_height)
                
                # Crear línea de anotación YOLO
                yolo_line = f"{class_id} {' '.join(map(str, yolo_bbox))}"
                yolo_annotations.append(yolo_line)
        
        # Escribir archivo de etiquetas
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        converted_images += 1
        
        if converted_images % 100 == 0:
            print(f"Procesadas {converted_images} imágenes...")
    
    print(f"Conversión completada:")
    print(f"- Imágenes convertidas: {converted_images}")
    print(f"- Imágenes omitidas: {skipped_images}")
    
    # Guardar archivo de clases
    classes_file = output_path / 'classes.txt'
    with open(classes_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Clases guardadas en: {classes_file}")
    print(f"Clases encontradas: {class_names}")
    
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Convertir dataset COCO a formato YOLO')
    parser.add_argument('--annotations', required=True, help='Archivo JSON de anotaciones COCO')
    parser.add_argument('--images', required=True, help='Directorio con imágenes')
    parser.add_argument('--output', required=True, help='Directorio de salida')
    parser.add_argument('--mapping', required=True, help='Archivo CSV de mapeo de clases')
    
    args = parser.parse_args()
    
    print("Iniciando conversión COCO a YOLO...")
    print(f"Archivo de anotaciones: {args.annotations}")
    print(f"Directorio de imágenes: {args.images}")
    print(f"Directorio de salida: {args.output}")
    print(f"Archivo de mapeo: {args.mapping}")
    
    class_names = coco_to_yolo(args.annotations, args.images, args.output, args.mapping)
    
    print("¡Conversión completada exitosamente!")

if __name__ == "__main__":
    main() 