import pandas as pd
import numpy as np
from bases_02 import riskfactors_df
import pandas_missing_extension
import seaborn as sns
import matplotlib.pyplot as plt
import os
import missingno as msno

#  os.system('clear')

# Tabulación de valores faltantes
print(riskfactors_df.isna())

# Resumenes básicos de valores faltantes
print(riskfactors_df.shape)
valores_completos = riskfactors_df.missing.number_complete()
print(f'Valores completos = {valores_completos}')
valores_faltantes = riskfactors_df.missing.number_missing()
print(f'Valores faltantes = {valores_faltantes}')

# Resúmenes tabulares de valores faltantes
# Variables son las C# olumnas
# Resumen por variable
print(riskfactors_df.missing.missing_variable_summary()) 
# Realiza un análisis de valores faltantes a nivel de columna en el DataFrame.
# La tabla resultante proporciona, para cada variable:
#   - El número de valores considerados como faltantes.
#   - La cuenta total de observaciones en la columna.
#   - El porcentaje que representan los valores faltantes en la columna.

print(riskfactors_df.missing.missing_variable_table())  # Imprime una tabla que agrupa variables por la cantidad de valores faltantes que contienen y su porcentaje.
print(riskfactors_df.missing.missing_case_summary())    # Imprime una tabla con tres columnas: la primera indica el índice de cada observación, la segunda muestra la cantidad de valores faltantes en esa observación y la tercera presenta el porcentaje de valores faltantes con respecto al total de variables.
print(riskfactors_df.missing.missing_case_table())      # Imprime una tabla que agrupa las observaciones según la cantidad de valores faltantes que tienen, mostrando cuántas filas pertenecen a cada grupo y el porcentaje que representan en el total del dataset.
print(
    riskfactors_df
    .missing
    .missing_variable_span(
        variable='weight_lbs',
        span_every=50
        )
)
# Genera un análisis de valores faltantes en un DataFrame, dividiendo los datos en intervalos definidos.
# Para cada intervalo, la función calcula:
#   - La cantidad de valores faltantes.
#   - La cantidad de valores completos.
#   - El porcentaje de completitud (valores completos / total de valores)
#  # La función devuelve una tabla resumen que facilita la identificación de patrones de datos faltantes en el DataFrame.

print((
    riskfactors_df
    .missing
    .missing_variable_run(
        variable='weight_lbs',
    )
))
# Crea una tabla con información sobre las rachas de datos faltantes y completos en una columna. 
# La tabla muestra, para cada racha:
#   - El tipo de racha (faltante o completo).
#   - La longitud de la racha actual.

#  riskfactors_df.missing.missing_variable_plot()
#  riskfactors_df.missing.missing_case_plot() # No está bien graficado
#  riskfactors_df.missing.missing_case_summary().plot(
#      kind='hist',
#      x='case',
#      y='pct_missing',
#      bins=15
#  )
#  plt.xlabel("Number of missings in case")  # Cambia el nombre del eje x
#  plt.ylabel("Number of cases")  # Cambia el nombre del eje y
#  plt.show()

#  riskfactors_df.missing.missing_variable_span_plot(
#          variable = 'weight_lbs',
#          span_every = 10,
#          rot = 0
#          )

# Falta integrar todas las graficas al sheet.md y además falta explicar desde .matrix
#  msno.bar(df = riskfactors_df)
#  msno.matrix(df = riskfactors_df)
#  riskfactors_df.missing.missing_upsetplot(variables = None, element_size = 60)
#  riskfactors_df.missing.missing_upsetplot(variables = ['pregnant', 'weight_lbs', 'smoke_stop'], element_size = 60)
#  msno.heatmap(df = riskfactors_df)
#  plt.show()

common_na_strings = (
    "missing",
    "NA",
    "N A",
    "N/A",
    "#N/A",
    "NA ",
    "N/A ",
    " N/A",
    "N / A ",
    "na",
    "na  ",
    "n/a",
    "na  ",
    "na   ",
    "n  / a",
    "n / a",
    "NULL",
    "null",
    "",
    "?",
    "*",
    ".",
)

common_na_numbers = (-9, -99, -999, -9999, 999, 66, 77, 88, -1)

missing_data_example_df = pd.DataFrame.from_dict(
    dict(
        x = [1, 3, "NA", -99, -98, -99],
        y = ["A", "N/A", "NA", "E", "F", "G"],
        z = [-100, -99, -98, -101, -1, -1]
    )
)

#  print(missing_data_example_df)
#  print(missing_data_example_df.missing.number_missing())
#  print(missing_data_example_df.dtypes)
#  print(missing_data_example_df.x.unique())

# Revisar valores únicos de los datos
#  print((
#      missing_data_example_df
#      .select_dtypes(object)
#      .apply(pd.unique)
#  ))

# Sustituyendo valores comúnmente asociados a valores faltantes
# Sustitución desde la lectura de los datos
dataframe_viejo = pd.read_csv('./data/missing_data_enconding_example.csv')
print('dataframe viejo')
print(dataframe_viejo)

dataframe_filtrado = pd.read_csv(
    "./data/missing_data_enconding_example.csv",
    na_values = [-99, -1]
)
print('dataframe nuevo')
print(dataframe_filtrado)

# Sustitución global de valores a NaN
data_sustitucion = (
    missing_data_example_df
    .replace(
        to_replace=[-99, 'NA', 'N/A'],
        value=np.nan
        )
    )
print(data_sustitucion)

data_sustitucion = pd.read_csv('./data/missing_data_enconding_example.csv', na_values=['NA', 'N/A', -99, -1, -98, -101, -100])
print(data_sustitucion)

# Sustitución dirigida (aplicado a una columna en específico)
data_dirigido = (
        missing_data_example_df
        .replace(
            to_replace={
                'x': {
                    -99: np.nan
                    }
                }
            )
        )
print(data_dirigido)

