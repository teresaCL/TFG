"""
El script debe estar ubicado en el directorio raíz del proyecto.
"""
import time
import subprocess
import os
import sys
from datetime import datetime
import pickle as pkl

# Título de la carpeta con resultados
title = "SHADE"

# Crear carpeta si no existe
date = datetime.now().strftime('%Y-%m-%d')
ruta_origen = f"./Results/{date}-{title}"
os.makedirs(ruta_origen, exist_ok=True)

# Listado de datasets y número de clusters   
datasets = [("iris",3), ("wine",3), ("connectionist",2), ("seeds",3), ("heart",2), ("vertebral",2), ("computers",2), ("gene",5), ("movement-libras",15), ("toxicity",2), ("ECG5000",5), ("ecoli",8), ("glass",6), ("accent",6)]    

# Crear y escribir cabecera en el archivo de resumen de los datasets "medias_por_dataset.txt"
nombre_archivo_resumen_dataset = f"{ruta_origen}/medias_por_dataset.txt"
cabecera_resumen_dataset = '||' + str("dataset").rjust(15) + ' |' + str("score").rjust(15) + ' |' + str("n_iter").rjust(15) + ' |' + str("n_ls").rjust(15) + ' |' + str("n_iter_ls").rjust(15) + ' |' + str("time(s)").rjust(15) + ' |' + str("% pop collapsed").rjust(15) + ' ||\n'
with open(nombre_archivo_resumen_dataset, "a") as archivo:
    archivo.write(cabecera_resumen_dataset)

# Crear y escribir cabecera en el archivo de resumen de ejecuciones con mismas restricciones "medias_mismas_restr.txt"
nombre_archivo_resumen_mismas_restr = f"{ruta_origen}/medias_mismas_restr.txt"
cabecera_resumen_mismas_restr = '||' + str("dataset").rjust(15) + ' |' + str("restricciones").rjust(15) + ' |' + str("score").rjust(15) + ' |' + str("n_iter").rjust(15) + ' |' + str("n_ls").rjust(15) + ' |' + str("n_iter_ls").rjust(15) + ' |' + str("time(s)").rjust(15) + ' |' + str("% pop collapsed").rjust(15) + ' ||\n'
with open(nombre_archivo_resumen_mismas_restr, "a") as archivo:
    archivo.write(cabecera_resumen_mismas_restr)

# Para cada dataset
for data, k in datasets:
    print(data)

    # Reiniciar listas
    list_score = []
    list_n_iter = []
    list_n_ls = []
    list_n_iter_ls = []
    list_elapsed_time = []
    list_is_pop_collapsed = []

    # Recuperar los archivos de restricciones del dataset y ordenarlos para que estén los archivos del mismo número y tipo de restricciones seguidos (del 0 al 4)
    archivos_restricciones = os.listdir(f"./Data/constraint_sets/{data}")
    archivos_restricciones.sort()

    # Ejecutar el algoritmo con todos los archivos de restricciones
    for file in archivos_restricciones:
        print(f"\t{file}")

        # Argumentos para ejecutar el algoritmo
        programa = [
            "python", "main_smdeclust.py",
            "--dataset", f"Data/{data}.txt",
            "--constraints", f"Data/constraint_sets/{data}/{file}",
            "--seed", "42",
            "--mutation",
            "--P", "20",
            "--K", f"{k}",
            "--assignment", "greedy",
            "--title", f"{title}",
            "--solis", "no",
            "--crossover", "original",
            "--apply_LS_all",
            "--F", "mdeclust"
        ]

        # Ejecutar el algoritmo
        result = subprocess.run(programa, capture_output=True, text=True)

        # Comprobar si hubo error
        if result.returncode != 0:
            print(f"Error al ejecutar {programa}:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        # Leer las variables del archivo pickle de resultados
        file_resultados = f"{ruta_origen}/{data}/{file.split('/')[-1].split('.')[0]}/{file.split('/')[-1].split('.')[0]}.pkl"
        with open(file_resultados, "rb") as f:
            results = pkl.load(f)
        
        list_score.append(results["score"])
        list_n_iter.append(results["n_iter"])
        list_n_ls.append(results["n_ls"])
        list_n_iter_ls.append(results["n_iter_ls"])
        list_elapsed_time.append(results["elapsed_time"])
        list_is_pop_collapsed.append(results["is_pop_collapsed"])

        # Guardar el tiempo medio de las 5 ejecuciones con la misma configuración de restricciones (mismo número y tipo)
        if file.endswith("4.json"):
            media_mismas_restr_score = sum(list_score[-5:])/5
            media_mismas_restr_n_iter = sum(list_n_iter[-5:])/5
            media_mismas_restr_n_ls = sum(list_n_ls[-5:])/5
            media_mismas_restr_n_iter_ls = sum(list_n_iter_ls[-5:])/5
            media_mismas_restr_elapsed_time = sum(list_elapsed_time[-5:])/5
            porcent_mismas_restr_collapsed = (sum(list_is_pop_collapsed[-5:])/5) * 100

            resumen_mismas_restr = '||' + str(f"{data}").rjust(15) + ' |' + str(f"{file.split('/')[-1].split('.')[0][:-2]}").rjust(15) + ' |' + str(round(media_mismas_restr_score,3)).rjust(15) + ' |' + str(round(media_mismas_restr_n_iter,3)).rjust(15) + ' |' + str(round(media_mismas_restr_n_ls,3)).rjust(15) + ' |' + str(round(media_mismas_restr_n_iter_ls,3)).rjust(15) + ' |' + str(round(media_mismas_restr_elapsed_time, 3)).rjust(15) + ' |' + str(round(porcent_mismas_restr_collapsed,3)).rjust(15) + ' ||\n'
            with open(nombre_archivo_resumen_mismas_restr, "a") as archivo:
                archivo.write(resumen_mismas_restr)
            
    # Guardar el tiempo medio de todas las ejecuciones con el mismo dataset
    media_score = sum(list_score)/len(list_score)
    media_n_iter = sum(list_n_iter)/len(list_n_iter)
    media_n_ls = sum(list_n_ls)/len(list_n_ls)
    media_n_iter_ls = sum(list_n_iter_ls)/len(list_n_iter_ls)
    media_elapsed_time = sum(list_elapsed_time)/len(list_elapsed_time)
    porcent_collapsed = (sum(list_is_pop_collapsed)/len(list_is_pop_collapsed)) * 100

    resumen_dataset = '||' + str(f"{data}").rjust(15) + ' |' + str(round(media_score,3)).rjust(15) + ' |' + str(round(media_n_iter,3)).rjust(15) + ' |' + str(round(media_n_ls,3)).rjust(15) + ' |' + str(round(media_n_iter_ls,3)).rjust(15) + ' |' + str(round(media_elapsed_time, 3)).rjust(15) + ' |' + str(round(porcent_collapsed,3)).rjust(15) + ' ||\n'
    with open(nombre_archivo_resumen_dataset, "a") as archivo:
        archivo.write(resumen_dataset)
