"""
Script para crear el código LaTex de las tablas con resultados
- Cada archivo de resultados en la carpeta ../Results/mediascsv debe empezar por un número identificativo seguido de un guión
- Uso: python crearTablas.py num1 nombre1 [num2 nombre2]
    num1 (y num2) son los números identificativos de los resultados que se quieren incluir en la tabla
    nombre1 (y nombre2) son los nombres que aparecerán en la tabla para hacer referencia a los resultados
"""
import csv
import os
import sys

# Lista con los datasets
datasets = ["iris", "wine", "connectionist", "seeds", "heart", "vertebral", "computers","gene", "movement-libras", "toxicity", "ECG5000", "ecoli", "glass", "accent"]

# Función que devuelve el nombre del dataset tal y como aparecerá en la tabla final
def formatear_nombre_dataset(name):
    name_lower = name.lower()
    if name_lower == "movement-libras":
        return "Movement Libras"
    elif name == "ECG5000":
        return "ECG5000"
    else:
        return name.capitalize() # primera letra en mayúscula

# Función que formatea un número en notación científica si es muy grande, o en decimal con los decimales dados si no
def formatear_numero(num, decimales=3):
    if abs(num) >= 4e7:
        return f"{num:.{decimales}e}"
    else:
        return f"{num:.{decimales}f}"

# Función que devuelve una diferencia con signo, color (rojo si es mayor, azul si es menor) y formato para mostrar en LaTeX
def formatear_diferencia(valor_ref, valor_cmp, decimales=3):
    diff = valor_cmp - valor_ref
    sign = "+" if diff >= 0 else "-"
    color = "black"
    if diff > 0:
        color = "red"
    elif diff < 0:
        color = "blue"
    valor = f"{sign}{formatear_numero(abs(diff), decimales)}"
    return f"{{\\color{{{color}}}{valor}}}"

# Función que devuelve un número con signo, color (rojo si es mayor, azul si es menor) y formato para mostrar en LaTeX
def formatear_promedio(valor, decimales=3):
    signo = "+" if valor > 0 else ""
    color = "red" if valor > 0 else ("blue" if valor < 0 else "black")
    valor_formateado = f"{signo}{formatear_numero(valor, decimales)}"
    return f"{{\\color{{{color}}}{valor_formateado}}}"

# Función que carga un archivo CSV y devuelve un diccionario donde cada clave es el nombre del dataset y cada valor es una fila completa del CSV asociada a ese dataset
def cargar_datos_csv(path):
    datos = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset = row["dataset"]
            datos[dataset] = row
    return datos

# Función que busca en una carpeta el archivo CSV cuyo nombre comienza con un número identificador seguido de guion y devuelve su ruta completa
def encontrar_archivo_por_num(carpeta_csv, num):
    for fname in os.listdir(carpeta_csv):
        if fname.startswith(f"{num}-") and fname.endswith(".csv"):
            return os.path.join(carpeta_csv, fname)
    return None

# Comprobar losargumentos del script y recuperlos
if len(sys.argv) < 3:
    print("Uso: python crearTablas.py num1 nombre1 [num2 nombre2]")
    sys.exit(1)

num1, nombre1 = sys.argv[1], sys.argv[2]
num2, nombre2 = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else (None, None)


# Directorio con los archivos csv de los resultados y directorio en el que se guardará el archivo de salida con el código de la tabla
carpeta_csv = "../Results/mediascsv"
carpeta_salida = "../Results/tablas"
os.makedirs(carpeta_salida, exist_ok=True) # crear carpeta de salida si no existe

# Cargar los resultados de referencia (1) y de los resultados que se quieren incluir en la tabla
archivo_ref = encontrar_archivo_por_num(carpeta_csv, "1")
archivo1 = encontrar_archivo_por_num(carpeta_csv, num1)
archivo2 = encontrar_archivo_por_num(carpeta_csv, num2) if num2 else None

if not archivo_ref or not archivo1 or (num2 and not archivo2):
    print("No se encontraron todos los archivos csv necesarios.")
    print(archivo_ref)
    print(archivo1)
    sys.exit(1)

datos_ref = cargar_datos_csv(archivo_ref)
datos1 = cargar_datos_csv(archivo1)
datos2 = cargar_datos_csv(archivo2) if archivo2 else None

# Escribir en el fichero de salida
nombre_fichero_salida = os.path.join(
    carpeta_salida,
    f"comparativa-{num1}-{num2}.txt" if num2 else f"comparativa-{num1}.txt"
)

with open(nombre_fichero_salida, "w") as out:
    # Formato
    col_format = "|r|r|r|" if not num2 else "|r|r|r|r|r|"
    out.write(f"\\begin{{table}}[h!]\n")
    out.write("\\centering\n")
    out.write(f"\\begin{{tabular}}{{{col_format}}}\n")
    out.write("\\hline\n")

    # Nombre de los algoritmos
    if num2:
        out.write(f" & \\multicolumn{{2}}{{c|}}{{{nombre1}}} & \\multicolumn{{2}}{{c|}}{{{nombre2}}} \\\\ \n")
        out.write("\\hline\n")
    
    # Cabecera
    headers = ["Dataset", "{{Score}}", "{{Time (s)}}"] if not num2 else ["Dataset", "{{Score}}", "{{Time (s)}}", "{{Score}}", "{{Time (s)}}"]
    out.write(" & ".join(headers) + " \\\\ \n")
    out.write("\\hline\n")

    # Acumuladores para el promedio de score, tiempo y el número de veces que mejora, iguala o empeora al algoritmo de referencia
    total_score1 = total_time1 = 0
    total_score2 = total_time2 = 0
    count = 0
    mejor1 = igual1 = peor1 = 0
    mejor2 = igual2 = peor2 = 0

    # Resultados para cada dataset
    for dataset in datasets:
        ref = datos_ref.get(dataset)    # datos del algoritmo de referencia para el dataset
        fila1 = datos1.get(dataset)     # datos del algoritmo num1 para el dataset
        fila2 = datos2.get(dataset) if datos2 else None     # datos del algoritmo num2 para el dataset (si se especifica)

        # Función para comporbar si la fila contiene datos válidos
        def datos_validos(fila):
            try:
                _ = float(fila["score"])
                _ = float(fila["time(s)"])
                return True
            except:
                return False
            
        # Escribir una fila en blanco (solo con el nombre del dataset) si faltan datos para dicho dataset
        if not ref or not fila1 or not datos_validos(ref) or not datos_validos(fila1) or (datos2 and (not fila2 or not datos_validos(fila2))):
            out.write(f"{formatear_nombre_dataset(dataset)} & " + " & ".join(["" for _ in range(4 if num2 else 2)]) + " \\\\ \n")
            continue

        # Score y time del algoritmo de referencia
        score_ref = float(ref["score"])
        time_ref = float(ref["time(s)"])

        # Fila a escribir
        fila = [formatear_nombre_dataset(dataset)]

        # Resultados del algoritmo num1
        score1 = float(fila1["score"])
        time1 = float(fila1["time(s)"])
        fila.append(formatear_diferencia(score_ref, score1, 3))
        fila.append(formatear_diferencia(time_ref, time1, 1))
        total_score1 += (score1-score_ref)
        total_time1 += (time1-time_ref)

        if score1 < score_ref:
            mejor1 += 1
        elif score1 == score_ref:
            igual1 += 1
        else:
            peor1 += 1

        # Resultados del algoritmo num2
        if num2:
            score2 = float(fila2["score"])
            time2 = float(fila2["time(s)"])
            fila.append(formatear_diferencia(score_ref, score2, 3))
            fila.append(formatear_diferencia(time_ref, time2, 1))
            total_score2 += (score2-score_ref)
            total_time2 += (time2-time_ref)

            if score2 < score_ref:
                mejor2 += 1
            elif score2 == score_ref:
                igual2 += 1
            else:
                peor2 += 1

        count += 1  # aumentar en 1 el número de datasets con datos válidos
        out.write(" & ".join(fila) + " \\\\ \n")

    out.write("\\hline\n")

    # Fila de promedio
    fila_prom = ["Promedio"]
    fila_prom.append(formatear_promedio(total_score1 / count))
    fila_prom.append(formatear_promedio(total_time1 / count, 1))
    if num2:
        fila_prom.append(formatear_promedio(total_score2 / count))
        fila_prom.append(formatear_promedio(total_time2 / count, 1))
    out.write(" & ".join(fila_prom) + " \\\\ \n")
    out.write("\\hline\n")

    # Fila de veces mejor
    fila_mejor = ["Veces mejor"]
    fila_mejor.append(f"{mejor1}")
    fila_mejor.append(f"-")
    if num2:
        fila_mejor.append(f"{mejor2}")
        fila_mejor.append(f"-")
    out.write(" & ".join(fila_mejor) + " \\\\ \n")

    # Fila de veces igual
    fila_igual = ["Veces igual"]
    fila_igual.append(f"{igual1}")
    fila_igual.append(f"-")
    if num2:
        fila_igual.append(f"{igual2}")
        fila_igual.append(f"-")
    out.write(" & ".join(fila_igual) + " \\\\ \n")

    # Fila de veces peor
    fila_peor = ["Veces peor"]
    fila_peor.append(f"{peor1}")
    fila_peor.append(f"-")
    if num2:
        fila_peor.append(f"{peor2}")
        fila_peor.append(f"-")
    out.write(" & ".join(fila_peor) + " \\\\ \n")

    # Final de la tabla, caption y label
    out.write("\\hline\n")
    out.write(f"\\end{{tabular}}\n")
    if num2:
        cap= f"Comparativa {nombre1} y {nombre2}"
    else:
        cap = f"Resultados {nombre1}"
    if num2:
        lab=f"tabla-{nombre1}-{nombre2}"
    else:
        lab=f"tabla-{nombre1}"
    out.write(f"\\caption{{{cap}}}\n")
    out.write(f"\\label{{{lab}}}\n")
    out.write(f"\\end{{table}}\n")


print(f"Tabla guardada con éxito en {nombre_fichero_salida}")