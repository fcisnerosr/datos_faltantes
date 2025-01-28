# Curso de datos faltantes
print(None or True,
        None or False,
        None == None,
        None is None,
        #  None + True
        type(None),
        sep="\n"
        )

numpy
import numpy as np
print(
        np.nan or True,
        np.nan == np.nan,
        np.nan is np.nan,
        np.nan / 2,
        type(np.nan),
        np.isnan(np.nan),
        sep='\n'
        )

# pandas
import pandas as pd

test_missing_df = pd.DataFrame.from_dict(
    data=dict(
        x=[0, 1, np.nan, np.nan, None],
        y=[0, 1, pd.NA, np.nan, None]
    )
)

print(test_missing_df)
print(test_missing_df.isna())
print(test_missing_df.isnull())
print(test_missing_df.x.isna())

print(pd.Series([1, np.nan]))
print(pd.Series([pd.to_datetime('2022-01-01'), np.nan]))
rint(pd.Series([-1]).isnull())
