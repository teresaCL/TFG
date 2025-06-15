"""
Script para convertir a csv todos los archivos txt con resultados situados en un determinado directorio de entrada
"""
import os
import csv

# Carpetas
INPUT_FOLDER = '../Results/mediastxt'
OUTPUT_FOLDER = '../Results/mediascsv'

# Lista de datasets
EXPECTED_DATASETS = ["iris", "wine", "connectionist", "seeds", "heart", "vertebral", "computers","gene", "movement-libras", "toxicity", "ECG5000", "ecoli", "glass", "accent"]

# Función que extrae los valores no vacíos de una línea separada por barras verticales 
def parse_line(line):
    parts = [x.strip() for x in line.strip().split('|') if x.strip()]
    return parts

# Función que procesa un archivo .txt con resultados tabulados y devuelve cabecera y datos organizados por dataset.
def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}   # datos para cada dataset
    headers = []    # cabecera

    for line in lines:
        if line.strip().startswith('||'):
            parts = parse_line(line)
            # Añadir cabecera o los datos del dataset
            if not headers:
                headers = parts
            else:
                dataset_name = parts[0] # primera columna es el dataset
                data[dataset_name] = parts  # guardar fila en el diccionaria data, bajo la clave dataset_name

    # Añadir datasets que faltan con valores vacíos
    for dataset in EXPECTED_DATASETS:
        if dataset not in data:
            data[dataset] = [dataset] + [''] * (len(headers) - 1)

    return headers, data

# Formatea los valores de la columna "time(s)" con un decimal para cada dataset esperado
def format_data(headers, data_dict):
    # Buscar índice de la columna time(s)
    time_idx = headers.index("time(s)")

    formatted_rows = []

    for dataset in EXPECTED_DATASETS:
        row = data_dict[dataset]    # extraer información del dataset
        try:
            row[time_idx] = f"{float(row[time_idx]):.1f}"   # convertir el valor a float y formatearlo con un decimal
        except ValueError:
            pass    # deja el valor como está si no se puede convertir
        formatted_rows.append(row)

    return formatted_rows

# Guarda los encabezados y los datos formateados en un archivo CSV en la ruta especificada
def save_to_csv(headers, data_rows, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in data_rows:
            writer.writerow(row)

# Convertir a cvs todos los archivos txt en la carpeta de entrada
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith('.txt'):
        txt_path = os.path.join(INPUT_FOLDER, filename)
        csv_path = os.path.join(OUTPUT_FOLDER, filename.replace('.txt', '.csv'))

        headers, data = process_file(txt_path)
        formatted_data = format_data(headers, data)
        save_to_csv(headers, formatted_data, csv_path)
        print(f"Convertido: {filename} --> {os.path.basename(csv_path)}")
