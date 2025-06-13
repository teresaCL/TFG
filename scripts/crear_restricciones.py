import os
import json
import random
import numpy as np
from itertools import combinations
import pandas as pd

# Función para generar las restricciones
def generate_constraints(data, n_ml, n_cl):
    labels = data[:, -1]
    
    # Agrupar índices por etiqueta
    label_groups_dict = {}  # las claves son las etiquetas y los valores son listas de índices de instancias con esa etiqueta.
    for idx, label in enumerate(labels):
        if label not in label_groups_dict:
            label_groups_dict[label] = []
        label_groups_dict[label].append(idx)
    
    # Listas con pares con la misma etiqueta y con diferente etiqueta
    same_label_pairs = []
    diff_label_pairs = []

    # Posibles etiquetas
    label_keys = list(label_groups_dict.keys())

    # Generar pares de la misma clase
    for label in label_keys:
        idx_group = label_groups_dict[label]    # índices de las instancias con etiqueta label
        same_label_pairs.extend(combinations(idx_group, 2))

    # Generar pares de clases distintas
    for i in range(len(label_keys)):
        for j in range(i + 1, len(label_keys)):
            idx_group1, idx_group2 = label_groups_dict[label_keys[i]], label_groups_dict[label_keys[j]]
            diff_label_pairs.extend((a, b) for a in idx_group1 for b in idx_group2)

    # Seleccionar restricciones aleatorias sin reemplazo
    must_link = random.sample(same_label_pairs, min(n_ml, len(same_label_pairs))) if len(same_label_pairs) > 0 else []
    cannot_link = random.sample(diff_label_pairs, min(n_cl, len(diff_label_pairs))) if len(diff_label_pairs) > 0 else []

    if len(same_label_pairs) < n_ml:
        print(f"Se encontraron solo {len(same_label_pairs)} pares con la misma etiqueta, menor que {n_ml} restricciones ML solicitadas.")

    if len(diff_label_pairs) < n_cl:
        print(f"Se encontraron solo {len(diff_label_pairs)} pares con distinta etiqueta, menor que {n_cl} restricciones CL solicitadas.")

    return must_link, cannot_link

# Función para crear el archivo json de restricciones
def generate_json(constraint_file_path, must_link, cannot_link):
    constraints = {"ml": must_link, "cl": cannot_link, "sml": [], "scl": [], "sml_proba": [], "scl_proba": []}
    with open(constraint_file_path, 'w') as json_file:
            json.dump(constraints, json_file)


############################################################################################################
# Número de archivos de restricciones a generar por cada tipo de configuración
num_file_constraints = 5

# Datasets para los que se van a generar las restricciones
datasets = [
    ("toxicity", 50, 100),
    ("movement-libras", 100, 150)
]

# Directorio de datos restricciones
data_folder = './Data'
constraint_folder = './Data/constraint_sets'
os.makedirs(data_folder, exist_ok=True)
os.makedirs(constraint_folder, exist_ok=True)

# Iterar sobre los archivos de datos y la lista de datasets
for dataset_name, num1, num2 in datasets:
    dataset_file = os.path.join(data_folder, f"{dataset_name}.txt")
    
    # Comprobar si el archivo existe en el directorio
    if os.path.exists(dataset_file):
        print(f"Procesando {dataset_file}...")
        dataset = pd.read_csv(dataset_file, header=None)
        data = dataset.values

        # Crear la carpeta para las restricciones si no existe
        dataset_constraint_folder = os.path.join(constraint_folder, dataset_name)
        if not os.path.exists(dataset_constraint_folder):
            os.makedirs(dataset_constraint_folder)

        # Generar archivos de restricciones. Número total de restricciones: num1. Tipo: solo ML
        for i in range(num_file_constraints):
            n_ml = num1
            n_cl = 0
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

        # Generar archivos de restricciones. Número total de restricciones: num1. Tipo: solo CL
        for i in range(num_file_constraints):
            n_ml = 0
            n_cl = num1
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

        # Generar archivos de restricciones. Número total de restricciones: num1. Tipo: mismo número CL y ML
        for i in range(num_file_constraints):
            n_ml = num1//2
            n_cl = num1 - n_ml
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

        # Generar archivos de restricciones. Número total de restricciones: num2. Tipo: solo ML
        for i in range(num_file_constraints):
            n_ml = num2
            n_cl = 0
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

        # Generar archivos de restricciones. Número total de restricciones: num2. Tipo: solo CL
        for i in range(num_file_constraints):
            n_ml = 0
            n_cl = num2
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

        # Generar archivos de restricciones. Número total de restricciones: num2. Tipo: mismo número CL y ML
        for i in range(num_file_constraints):
            n_ml = num2//2
            n_cl = num2 - n_ml
            must_link, cannot_link = generate_constraints(data, n_ml, n_cl)
            json_filename = os.path.join(dataset_constraint_folder, f"ml_{n_ml}_cl_{n_cl}_{i}.json")
            generate_json(json_filename, must_link, cannot_link)

    else:
        print(f"No se encontró el archivo {dataset_file}")
                      
