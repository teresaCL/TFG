[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![license](https://img.shields.io/badge/license-apache_2.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

# Trabajo Fin de Grado - Mejorando Metaheurísticas para el problema del clustering semi-supervisado

**Alumna**: Teresa Córdoba Lillo

**Titulación**: Grado en Ingeniería Informática, Universidad de Granada

**Tutores**:
- Daniel Molina Cabrera
- Francisco Javier Rodríguez Díaz


### Instrucciones de ejecución
Este repositorio contiene dos programas:
- El primero (```main_smdeclust.py```) implementa diversas modificaciones sobre el algoritmo memético S-MDEClust propuesto por Mansueto, P. & Schoen, F.

  Más información en:
  - [Artículo original](https://arxiv.org/abs/2403.04322)
  - [Repositorio oficial](https://github.com/pierlumanzu/s_mdeclust?tab=readme-ov-file)

  **Ejemplo de uso**:

  ```sh
  python main_smdeclust.py --dataset Data/iris.txt --constraints Data/constraint_sets/iris/ml_25_cl_25_0.json --P 20 --K 3 --apply_LS_all assignment greedy_rand --crossover pbest_v1 --restart 3
  ```


- El segundo programa implementa un enfoque GRASP para abordar el problema de clustering semisupervisado (```main_grasp.py```)

  **Ejemplo de uso**:

  ```sh
  python main_grasp.py --dataset Data/iris.txt --constraints Data/constraint_sets/iris/ml_25_cl_25_0.json --K 3 --assignment greedy --Nmax 3 --max_iter 30
  ```

En los archivos ```args_utils_smdeclust.py``` y ```args_utils_grasp.py``` se pueden encontrar todos los posibles argumentos para cada programa.

Para ejecutar el código es necesario un entorno [Anaconda](https://www.anaconda.com/). Además, será necesario instalar [Gurobi](https://www.gurobi.com/) Optimizer y poseer de una licencia válida.

#### Paquetes principales 

* ```python v3.11.9```
* ```pip v24.0```
* ```numpy v2.0.0```
* ```scipy v1.14.0```
* ```pandas v2.2.2```
* ```gurobipy v11.0.2```
* ```networkx v3.3```
