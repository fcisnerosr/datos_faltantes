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
















