import pandas as pd

df = pd.DataFrame.from_dict(
    data = {
        "a": list("asdfasdfas"),
        "b": range(0, 10)
    }
)

df.iloc[2:5, 0] = None
df.iloc[6:7, 1] = None

df

df.a.str

# df.missing

try:
    del pd.DataFrame.missing
except AttributeError:
    pass

@pd.api.extensions.register_dataframe_accessor("missing")
class DontMissMe:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def number_missing(self):
        return self._obj.isna().sum().sum()

    def number_complete(self):
        return self._obj.size - self._obj.missing.number_missing()

    def proportion_missing(self):
        pass

df = pd.DataFrame(df)

df.missing.number_missing()

df.missing.number_complete()

df.missing.proportion_missing()
