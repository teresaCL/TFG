"""
Script para convertir los archivos con restricciones a formato json requerido por el algoritmo S-MDEClust 
Uso: python to_json.py <input_dir> <output_dir>
    - input_dir: directorio de entrada. Debe de tener un subdirectorio para cada dataset que contenga los archivos de restricciones. El formato de cada linea del archivo de restricciones es: TIPO IDX1 IDX2
    - output_dir: directrio de salida 
"""
import os
import sys
import json

# Procesa un archivo de texto con restricciones ML/CL y guarda su representación estructurada en un archivo JSON
def process_file(input_file, output_path):
    # Obtener el nombre del archivo sin su extensión
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # El nombre del archivo de salida será el mismo que el de entrada, pero con .json
    output_file = os.path.join(output_path, f"{base_name}.json")

    ml_data = []
    cl_data = []

    # Abrir y leer el archivo de entrada
    with open(input_file, 'r') as file:
        for line in file:
            # Separar la línea por espacios
            parts = line.split()

            # Obtener los valores según la etiqueta
            if parts[0] == 'ML':
                ml_data.append([int(parts[1]), int(parts[2])])
            elif parts[0] == 'CL':
                cl_data.append([int(parts[1]), int(parts[2])])

    # Crear el diccionario
    data = {
        "ml": ml_data,
        "cl": cl_data,
        "sml": [],
        "scl": [],
        "sml_proba": [],
        "scl_proba": []
    }

    # Guardar el resultado en un archivo JSON
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


# Comprobar y recuperar argumentos
if len(sys.argv) < 3:
        print("Uso: python to_json.py <input_dir> <output_dir>")
        sys.exit(1)

input_base_path = sys.argv[1]
output_base_path = sys.argv[2]

if not os.path.isdir(input_base_path):
    print(f"[Error]: El directorio de entrada no existe: {input_base_path}")
    sys.exit(1)

# Crear directorio de salida si no existe
os.makedirs(output_base_path, exist_ok=True)

# Nombres de las carpetas con losa archivos de restricciones para cada dataset
datasets = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]

# Convertir todos los archivos de restricciones de cada conjunto de datos
for dataset in datasets:
    input_dir = os.path.join(input_base_path, dataset)
    output_dir = os.path.join(output_base_path, dataset)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file)
        process_file(input_path, output_dir)
        print(f"Guardado: {dataset}/{file.replace('.txt', '.json')}")