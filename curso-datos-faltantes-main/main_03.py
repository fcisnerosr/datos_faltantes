import pandas as pd
from bases_02 import riskfactors_df
import pandas_missing_extension
import os

#  os.system('clear')

# Tabulación de valores faltantes
#  print(riskfactors_df.isna())

# Resuenes basicos de valores faltantes
print(riskfactors_df.shape)
valores_completos = riskfactors_df.missing.number_complete()
print(f'Valores completos = {valores_completos}')
valores_faltantes = riskfactors_df.missing.number_missing()
print(f'Valores faltantes = {valores_faltantes}')

# Resúmenes tabulares de valores faltantes
# Variables/Columnas
# Resumen por variable
print(riskfactors_df.missing.missing_variable_summary()) 
# Realiza un análisis de valores faltantes a nivel de columna en el DataFrame.
# La tabla resultante proporciona, para cada variable:
#   - El número de valores considerados como faltantes.
#   - La cuenta total de observaciones en la columna.
#   - El porcentaje que representan los valores faltantes en la columna.

print(riskfactors_df.missing.missing_variable_table())  # Imprime una tabla que agrupa variables por la cantidad de valores faltantes que contienen y su porcentaje.
#  print((
#      riskfactors_df
#      .missing
#      .missing_variable_span(
#          variable='weight_lbs',
#          span_every=50
#      )
#  ))
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
