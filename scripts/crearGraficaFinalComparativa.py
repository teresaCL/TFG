"""
Script para crear la gráfica comparativa de todas las propuestas
"""
import matplotlib.pyplot as plt
import numpy as np

# Datos
ids = np.arange(1, 27)
nombres = [
    "GRASP", "AGR-RAND", "AGR-RAND-P", "SHADE", "PBEST1-F", "PBEST2-F", "PBEST1-F/2",
    "PBEST2-F/2", "SW-V1-WO-PEN", "SW-WO-PEN", "SW-W-PEN", "SW-WO-PEN-AGR-RAND",
    "PBEST1-F-SW-WO-PEN", "PBEST1-F-AGR-RAND", "PBEST1-F-AGR-RAND-SW", "SEL-BL",
    "R-2IT-4", "R-3IT-2", "R-3IT-2-SW", "R-3IT-2-DIS", "R-3IT-2-DIS-SW",
    "R-3IT-2-PBEST1-F", "R-3IT-2-PBEST1-F-SW", "R-3IT-2-DIS-PBEST1-F",
    "R-3IT-2-DIS-PBEST1-F-SW", "R-3IT-2-PBEST1-F-AGR-RAND"
]
veces_mejor = [1, 4, 5, 5, 4, 6, 5, 5, 4, 4, 2, 4, 5, 4, 7, 1, 7, 6, 8, 7, 8, 8, 8, 7, 8, 8]
veces_igual = [0, 8, 6, 7, 8, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6]
veces_peor = [13, 2, 3, 2, 2, 1, 2, 2, 3, 4, 6, 3, 2, 3, 0, 6, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0]

# Convertir a arrays de numpy
mejor = np.array(veces_mejor)
igual = np.array(veces_igual)
peor = np.array(veces_peor)

# Crear gráfica
plt.figure(figsize=(16, 8))
plt.bar(ids, mejor, color='blue', label='Veces Mejor')
plt.bar(ids, igual, bottom=mejor, color='gray', label='Veces Igual')
plt.bar(ids, peor, bottom=mejor+igual, color='red', label='Veces Peor')

# Ejes y etiquetas
plt.xlabel("Algoritmo")
plt.ylabel("Número de comparaciones: mejor, igual y peor")
plt.title("Comparación de Algoritmos por Veces Mejor, Igual y Peor")
plt.xticks(ids, nombres, rotation=90)  # Mostrar todos los ticks con nombres rotados
max_y = max((mejor + igual + peor))  # Altura máxima total
plt.yticks(np.arange(0, max_y + 1, 1))  # Ticks
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("graficaComparativaFinal.png", dpi=300, bbox_inches='tight')

plt.show()
