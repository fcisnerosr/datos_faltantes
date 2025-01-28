import os
import janitor
import matplotlib.pyplot as plt
import missingno
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
import session_info
import upsetplot
import subprocess
os.system('clear')


pima_indians_diabetes_url = "https://nrvis.com/data/mldata/pima-indians-diabetes.csv"

output_path = "./data/prima_indians-diabetes.csv"

subprocess.run(["wget", "-O", output_path, pima_indians_diabetes_url, "-q"])

diabetes_df = pd.read_csv('~/Documents/datos_faltantes/curso-datos-faltantes-main/data/pima-indians-diabetes.csv',
    sep=',',
    names = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "outcome",
    ]
)

# Automatizacion de obtencion de datos de un sitio web
base_url = "https://github.com/njtierney/naniar/raw/master/data/"
datasets_names = ("oceanbuoys", "pedestrian", "riskfactors")

extension = ".rda"

dataset_dfs = {}
# Descargar y cargar los datasets
for dataset in datasets_names:
    file_name = f"{dataset}{extension}" # string que guarda el nombrel dataset y de la extension en cuestion 
    output_path = f"./data/{file_name}" # string que guarda la ubicacion de cada archivo .rda descargado
    
    # Descargar el archivo
    subprocess.run(["wget", "-O", output_path, f"{base_url}{file_name}", "-q"]) # -q Sirve para que no muestre el estado de la descarga, simplemente ejecuta el comando sin mostrar que paso por detras. OJO que si hay algun error en la descarga tampoco lo muestra
    
    # Leer el archivo .rda y almacenarlo en un diccionario (.rda es un formato de R, estos dataframes fueron creados originalmente en R)
    result = pyreadr.read_r(output_path)  # Devuelve un diccionario con dataframes
    dataset_dfs[dataset] = result[next(iter(result))]  # Extraer el dataframe
    #  Primera iteracion mostrada:
    #  next(iter(result))  # Retorna "oceanbuoys"
    #  result[next(iter(result))]  # Retorna el dataframe asociado a "oceanbuoys"
    #  dataset_dfs["oceanbuoys"] = result["oceanbuoys"]  # Almacena el dataframe


oceanbuoys_df = dataset_dfs["oceanbuoys"] # esta sintaxis me permite extraer los valores de la clave del diccionario dataset_dfs
pedestrian_df = dataset_dfs["pedestrian"]
riskfactors_df = dataset_dfs["riskfactors"]

#  print('riskfactors_df')
#  print(riskfactors_df.shape)
#  print(riskfactors_df.info())

# Extención la API de Pandas
df = pd.DataFrame.from_dict(
    data = {
        "a": list("asdfasdfas"),
        "b": range(0, 10)
    }
)
# asginacion de valores faltantes
df.iloc[2:5, 0] = None
df.iloc[6:7, 1] = None
#  print(df)

#  print(df.a.str)

# Creacion de una nueva clase para extender pandas
@pd.api.extensions.register_dataframe_accessor('missing')
class MissingMethods:
    def __init__(self, pandas_obj):
        self._df = pandas_obj
    def number_missing(self):
        return self._df.isna().sum().sum()
    def number_complete(self):
        return sel._df.size - self._df.missing.number_missing()
    def prportion_missing(self):
        pass


#  df = pd.DataFrame(df)
#  print(df.missing.number_missing())
