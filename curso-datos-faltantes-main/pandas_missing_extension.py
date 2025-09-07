import itertools
import pandas as pd
import upsetplot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    del pd.DataFrame.missing
except AttributeError:
    pass


@pd.api.extensions.register_dataframe_accessor("missing")
class MissingMethods:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def number_missing(self) -> int:
        return self._obj.isna().sum().sum()

    def number_complete(self) -> int:
        return self._obj.size - self._obj.missing.number_missing()

    def missing_variable_summary(self) -> pd.DataFrame:
        return self._obj.isnull().pipe(
            lambda df_1: (
                df_1.sum()
                .reset_index(name="n_missing")
                .rename(columns={"index": "variable"})
                .assign(
                    n_cases=len(df_1),
                    pct_missing=lambda df_2: df_2.n_missing / df_2.n_cases * 100,
                )
            )
        )

    def missing_case_summary(self) -> pd.DataFrame:
        return self._obj.assign(
            case=lambda df: df.index,
            n_missing=lambda df: df.apply(
                axis="columns", func=lambda row: row.isna().sum()
            ),
            pct_missing=lambda df: df["n_missing"] / df.shape[1] * 100,
        )[["case", "n_missing", "pct_missing"]]

    #  def missing_variable_table(self) -> pd.DataFrame:
    #      return (
    #          self._obj.missing.missing_variable_summary()
    #          .value_counts("n_missing")
    #          .reset_index()
    #          .rename(columns={"n_missing": "n_missing_in_variable", 0: "n_variables"})
    #          .assign(
    #              pct_variables=lambda df: df.n_variables / df.n_variables.sum() * 100
    #          )
    #          .sort_values("pct_variables", ascending=False)
    #      )
    def missing_variable_table(self) -> pd.DataFrame:
        return (
            self._obj.missing.missing_variable_summary()
            .value_counts("n_missing")
            .reset_index(name="n_variables")
            .rename(columns={"n_missing": "n_missing_in_variable"})
            .assign(
                pct_variables=lambda df: df.n_variables / df.n_variables.sum() * 100
            )
            .sort_values("pct_variables", ascending=False)
        )

    #  def missing_case_table(self) -> pd.DataFrame():
    #      return (
    #          self._obj.missing.missing_case_summary()
    #          .value_counts("n_missing")
    #          .reset_index()
    #          .rename(columns={"n_missing": "n_missing_in_case", 0: "n_cases"})
    #          .assign(pct_case=lambda df: df.n_cases / df.n_cases.sum() * 100)
    #          .sort_values("pct_case", ascending=False)
    #      )

    def missing_case_table(self) -> pd.DataFrame:
        return (
            self._obj.missing.missing_case_summary()
            .value_counts(
                subset=["n_missing"]
            )  # Se agrupa por el número de valores faltantes
            .to_frame(
                name="n_cases"
            )  # Convierte a DataFrame y asigna nombre a la columna
            .reset_index()
            .rename(
                columns={"n_missing": "n_missing_in_case"}
            )  # Renombra columna correctamente
            .assign(
                pct_case=lambda df: (df["n_cases"] / df["n_cases"].sum() * 100).astype(
                    float
                )
            )  # Calcula porcentaje
            .sort_values(
                "pct_case", ascending=False
            )  # Ordena por porcentaje descendente
        )

    def missing_variable_span(self, variable: str, span_every: int) -> pd.DataFrame:
        return (
            self._obj.assign(
                span_counter=lambda df: (
                    np.repeat(a=range(df.shape[0]), repeats=span_every)[: df.shape[0]]
                )
            )
            .groupby("span_counter")
            .aggregate(
                n_in_span=(variable, "size"),
                n_missing=(variable, lambda s: s.isnull().sum()),
            )
            .assign(
                n_complete=lambda df: df.n_in_span - df.n_missing,
                pct_missing=lambda df: df.n_missing / df.n_in_span * 100,
                pct_complete=lambda df: 100 - df.pct_missing,
            )
            .drop(columns=["n_in_span"])
            .reset_index()
        )

    def missing_variable_run(self, variable) -> pd.DataFrame:
        rle_list = self._obj[variable].pipe(
            lambda s: [[len(list(g)), k] for k, g in itertools.groupby(s.isnull())]
        )

        return pd.DataFrame(data=rle_list, columns=["run_length", "is_na"]).replace(
            {False: "complete", True: "missing"}
        )

    def sort_variables_by_missingness(self, ascending=False):

        return self._obj.pipe(
            lambda df: (df[df.isna().sum().sort_values(ascending=ascending).index])
        )

    def create_shadow_matrix(
        self,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return (
            self._obj.isna()
            .pipe(lambda df: df[df.columns[df.any()]] if only_missing else df)
            .replace({False: false_string, True: true_string})
            .add_suffix("_NA")
        )

    def bind_shadow_matrix(
        self,
        true_string: str = "Missing",
        false_string: str = "Not Missing",
        only_missing: bool = False,
    ) -> pd.DataFrame:
        return pd.concat(
            objs=[
                self._obj,
                self._obj.missing.create_shadow_matrix(
                    true_string=true_string,
                    false_string=false_string,
                    only_missing=only_missing,
                ),
            ],
            axis="columns",
        )

    def missing_scan_count(self, search) -> pd.DataFrame:
        hits = self._obj.apply(lambda col: col.isin(search))  # axis=0 por defecto (por columnas)
        out = (hits.sum()
                 .rename("n")
                 .reset_index()
                 .rename(columns={"index": "variable"}))
        types = self._obj.dtypes.astype(str)
        return out.assign(original_type=lambda d: d["variable"].map(types))

    # Plotting functions ---

    #  def missing_variable_plot(self):
    #      df = self._obj.missing.missing_variable_summary().sort_values("n_missing")
    #
    #      plot_range = range(1, len(df.index) + 1)
    #
    #      plt.hlines(y=plot_range, xmin=0, xmax=df.n_missing, color="black")
    #
    #      plt.plot(df.n_missing, plot_range, "o", color="black")
    #
    #      plt.yticks(plot_range, df.variable)
    #
    #      plt.grid(axis="y")
    #
    #      plt.xlabel("Number missing")
    #      plt.ylabel("Variable")

    def missing_variable_plot(self):
        df = self._obj.missing.missing_variable_summary().sort_values("n_missing")

        plot_range = np.arange(1, len(df.index) + 1)  # Convertir a NumPy array
        n_missing_values = df.n_missing.to_numpy()  # Convertir a NumPy array

        plt.hlines(y=plot_range, xmin=0, xmax=n_missing_values, color="black")
        plt.plot(n_missing_values, plot_range, "o", color="black")

        plt.yticks(plot_range, df.variable)
        plt.grid(axis="y")

        plt.xlabel("Number missing")
        plt.ylabel("Variable")
        plt.show()  # Asegurar que la gráfica se muestre correctamente

        df = self._obj.missing.missing_case_summary()

        sns.displot(data=df, x="n_missing", binwidth=1, color="black")

        plt.grid(axis="x")
        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")

    def missing_case_plot(self):
        df = self._obj.missing.missing_case_summary().plot(
            kind="hist", x="case", y="pct_missing", bins=15
        )

        plt.xlabel("Number of missings in case")
        plt.ylabel("Number of cases")
        plt.show()

    def missing_variable_span_plot(
        self, variable: str, span_every: int, rot: int = 0, figsize=None
    ):

        (
            self._obj.missing.missing_variable_span(
                variable=variable, span_every=span_every
            ).plot.bar(
                x="span_counter",
                y=["pct_missing", "pct_complete"],
                stacked=True,
                width=1,
                color=["black", "lightgray"],
                rot=rot,
                figsize=figsize,
            )
        )

        plt.xlabel("Span number")
        plt.ylabel("Percentage missing")
        plt.legend(["Missing", "Present"])
        plt.title(
            f"Percentage of missing values\nOver a repeating span of { span_every } ",
            loc="left",
        )
        plt.grid(False)
        plt.margins(0)
        plt.tight_layout(pad=0)
        plt.show()

    def missing_upsetplot(self, variables: list[str] = None, **kwargs):

        if variables is None:
            variables = self._obj.columns.tolist()

        return (
            self._obj.isna()
            .value_counts(variables)
            .pipe(lambda df: upsetplot.plot(df, **kwargs))
        )
    

    def column_fill_with_dummies(
        self,
        variable: str,              # nombre de la columna a procesar
        proportion_below: float = 0.10,
        jitter: float = 0.075,
        seed: int = 42
    ) -> pd.Series:
        # 1) Crear copia profunda para no alterar la serie original
        column = self._obj[variable].copy(deep=True)
        # El cambio de columna que procese en mi función fill_with_dummies()
        # no cambia la serie del DataFrame original, solo la copia local.

        # 2) Identificar posiciones con NaN
        missing_mask = column.isna()           # Serie booleana: True donde había NaN
        number_missing_values = missing_mask.sum()  # Cantidad total de NaN

        # 3) Calcular rango auténtico de la serie (max – min)
        real_min = column.min()                # Mínimo real (sin NaN)
        real_max = column.max()                # Máximo real
        column_range = real_max - real_min     # Diferencia para escalar el jitter

        # 4) Definir 'piso' de dummies: un poco por debajo del mínimo real
        #    Ejemplo: con proportion_below=0.10, shift = real_min – 10%·real_min
        column_shift = real_min - (real_min * proportion_below)

        # 5) Generar ruido (jitter) para cada dummy
        #    - np.random.seed fija la semilla para reproducibilidad
        #    - np.random.rand(n) crea n valores en [0,1)
        #    - restar 2 pasa esos valores a [-2, -1) ⇒ desplazamiento negativo garantizado
        #    - multiplicar por column_range * jitter ajusta la magnitud del ruido
        np.random.seed(seed)
        column_jitter = (np.random.rand(number_missing_values) - 2) * column_range * jitter

        # 6) Asignar a cada posición NaN un valor dummy = piso + su jitter correspondiente
        column[missing_mask] = column_shift + column_jitter

        # 7) Devolver la serie modificada (mismos índices, dummies en reemplazo de NaN)
        return column

    def missing_scatterplot_with_dummies(
        self,
        x: str,
        y: str,
        proportion_below: float = 0.05,
        jitter: float = 0.075,
        seed: int = 42,
    ):
        """
        Scatter bivariado con relleno dummy para NaN y coloreo por nullity (x_NA | y_NA).
        Devuelve el Axes de seaborn.
        """
        # 1) Numéricas y garantizar que x,y estén incluso si no tienen NaN
        num = self._obj.select_dtypes(exclude="category")
        # 1) Detecta las columnas que tienen AL MENOS un NaN.
        #    Recorremos todos los nombres en num.columns y nos quedamos con aquellos
        #    para los que num[c].isna().any() es True.
        cols_with_na = [c for c in num.columns if num[c].isna().any()] # Recorre los nombres de columna num y regresa los que tienen al menos un NaN
        # 2) Construye el conjunto de columnas a conservar:
        #    - {x, y} garantiza que las variables que quieres graficar se incluyan
        #      aunque NO tengan NaN (de lo contrario podrían “perderse” si filtras solo por NaN).
        #    - .union(cols_with_na) agrega todas las columnas con NaN.
        #    - Convertimos a list(...) porque union devuelve un set (sin duplicados).
        #    Nota: los sets no preservan orden; si quieres mantener el orden original,
        #    usa la variante del bloque alternativo más abajo.
        keep = list({x, y}.union(cols_with_na))
        
        # 3) Subselecciona únicamente esas columnas: reducimos el DataFrame a lo esencial
        #    (x, y y las que tienen NaN) para acelerar los pasos siguientes (sombra, dummies, plot).
        df = num[keep]

        # 2) Matriz de sombras
        df = df.missing.bind_shadow_matrix(true_string=True, false_string=False)

        # 3) Relleno dummy solo en columnas NO _NA
        def _maybe_fill(col):
            if col.name.endswith("_NA"):
                return col
            # usa la función univariada que trabaja con Series
            return column_fill_with_dummies(col, proportion_below=proportion_below,
                                            jitter=jitter, seed=seed)

        df = df.apply(_maybe_fill)

        # 4) Máscara bivariada
        df = df.assign(nullity=lambda d: d[f"{x}_NA"] | d[f"{y}_NA"])

        # 5) Gráfica
        ax = sns.scatterplot(data=df, x=x, y=y, hue="nullity")
        return ax


