import janitor
import matplotlib.pyplot as plt
import missingno
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
import session_info
import upsetplot

%run pandas-missing-extension.ipynb

%matplotlib inline

#  sns.set(
#      rc={
#          "figure.figsize": (10, 10)
#      }
#  )
#
#  sns.set_style("whitegrid")
#
#  print(
#      None or True,
#      None or False,
#      None == None,
#      None is None,
#      # None + True,
#      # None / False,
#      type(None),
#      sep="\n"
#  )
#
#  print(
#      np.nan or True,
#      np.nan or False,
#      np.nan == np.nan,
#      np.nan is np.nan,
#      np.nan / 2,
#      np.nan * 7,
#      type(np.nan),
#      np.isnan(np.nan),
#      sep="\n"
#  )
#
#  test_missing_df = pd.DataFrame.from_dict(
#      data=dict(
#          x=[0, 1, np.nan, np.nan, None],
#          y=[0, 1, pd.NA, np.nan, None]
#      )
#  )
#
#  test_missing_df
#
#  test_missing_df.isna()
#  test_missing_df.isnull()
#
#  test_missing_df.x.isnull()
#
#  pd.Series([1, np.nan])
#
#  pd.Series([pd.to_datetime("2022-01-01"), np.nan])
#
#  pd.Series([-1]).isnull()
#
#  pima_indians_diabetes_url = "https://nrvis.com/data/mldata/pima-indians-diabetes.csv"
#
#  !wget -O ./data/pima-indians-diabetes.csv { pima_indians_diabetes_url } -q
#
#  diabetes_df = pd.read_csv(
#      filepath_or_buffer="./data/pima-indians-diabetes.csv", # or pima_indians_diabetes_url
#      sep=",",
#      names=[
#          "pregnancies",
#          "glucose",
#          "blood_pressure",
#          "skin_thickness",
#          "insulin",
#          "bmi",
#          "diabetes_pedigree_function",
#          "age",
#          "outcome",
#      ]
#  )
#
#  base_url = "https://github.com/njtierney/naniar/raw/master/data/"
#  datasets_names = ("oceanbuoys", "pedestrian", "riskfactors")
#  extension = ".rda"
#
#  datasets_dfs = {}
#
#  for dataset_name in datasets_names:
#
#      dataset_file = f"{ dataset_name }{ extension }"
#      dataset_output_file = f"./data/{ dataset_file }"
#      dataset_url = f"{ base_url }{ dataset_file }"
#
#      !wget -q -O { dataset_output_file } { dataset_url }
#
#      datasets_dfs[f"{ dataset_name }_df"] = pyreadr.read_r(dataset_output_file).get(dataset_name)
#
#  datasets_dfs.keys()
#
#  locals().update(**datasets_dfs)
#  del datasets_dfs
#
#  oceanbuoys_df.shape, pedestrian_df.shape, riskfactors_df.shape, diabetes_df.shape
#
#  riskfactors_df.info()
#
#  riskfactors_df.isna()
#
#  riskfactors_df.shape
#
#  riskfactors_df.missing.number_complete()
#
#
#  riskfactors_df.missing.number_missing()
#
#  riskfactors_df.missing.missing_variable_summary()
#
#  riskfactors_df.missing.missing_variable_table()
#
#  riskfactors_df.missing.missing_case_summary()
#
#  riskfactors_df.missing.missing_case_table()
#
#  (
#      riskfactors_df
#      .missing
#      .missing_variable_span(
#          variable="weight_lbs",
#          span_every=50
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .missing_variable_run(
#          variable="weight_lbs"
#      )
#  )
#
#
#
#  riskfactors_df.missing.number_complete()
#
#  riskfactors_df.missing.number_missing()
#
#  riskfactors_df.missing.missing_variable_summary()
#
#
#  riskfactors_df.missing.missing_variable_table()
#
#  riskfactors_df.missing.missing_case_summary()
#
#  riskfactors_df.missing.missing_case_table()
#
#  (
#      riskfactors_df
#      .missing
#      .missing_variable_span(
#          variable="weight_lbs",
#          span_every=50
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .missing_variable_run(
#          variable="weight_lbs"
#      )
#  )
#
#  riskfactors_df.missing.missing_variable_plot()
#
#  riskfactors_df.missing.missing_case_plot()
#
#  (
#      riskfactors_df
#      .missing
#      .missing_variable_span_plot(
#          variable="weight_lbs",
#          span_every=10,
#          rot=0
#      )
#  )
#
#  missingno.bar(df = riskfactors_df)
#
#  missingno.matrix(df=riskfactors_df)
#
#  (
#      riskfactors_df
#      .missing
#      .missing_upsetplot(
#          variables = None,
#          element_size = 60
#      )
#  )
#
#  common_na_strings = (
#      "missing",
#      "NA",
#      "N A",
#      "N/A",
#      "#N/A",
#      "NA ",
#      " NA",
#      "N /A",
#      "N / A",
#      " N / A",
#      "N / A ",
#      "na",
#      "n a",
#      "n/a",
#      "na ",
#      " na",
#      "n /a",
#      "n / a",
#      " a / a",
#      "n / a ",
#      "NULL",
#      "null",
#      "",
#      "?",
#      "*",
#      ".",
#  )
#
#  common_na_numbers = (-9, -99, -999, -9999, 9999, 66, 77, 88, -1)
#
#  missing_data_example_df = pd.DataFrame.from_dict(
#      dict(
#          x = [1, 3, "NA", -99, -98, -99],
#          y = ["A", "N/A", "NA", "E", "F", "G"],
#          z = [-100, -99, -98, -101, -1, -1]
#      )
#  )
#
#  missing_data_example_df
#
#  missing_data_example_df.missing.number_missing()
#
#  missing_data_example_df.dtypes
#
#  missing_data_example_df.x.unique()
#
#  (
#      missing_data_example_df
#      .select_dtypes(object)
#      .apply(pd.unique)
#  )
#
#  pd.read_csv(
#      "./data/missing_data_enconding_example.csv",
#      na_filter=True,
#      na_values=[-99, -1]
#  )
#
#  (
#      missing_data_example_df
#      .replace(
#
#          to_replace=[-99, "NA"],
#          value=np.nan
#      )
#  )
#
#  (
#      missing_data_example_df
#      .replace(
#          to_replace={
#              "x": {
#                  -99: np.nan
#              }
#          }
#      )
#  )
#
#  implicit_to_explicit_df = pd.DataFrame.from_dict(
#      data={
#          "name": ["lynn", "lynn", "lynn", "zelda"],
#          "time": ["morning", "afternoon", "night", "morning"],
#          "value": [350, 310, np.nan, 320]
#      }
#  )
#
#  implicit_to_explicit_df
#
#  (
#      implicit_to_explicit_df
#      .pivot_wider(
#          index="name",
#          names_from="time",
#          values_from="value"
#      )
#  )
#
#  (
#      implicit_to_explicit_df
#      .value_counts(
#          subset=["name"]
#      )
#      .reset_index(name="n")
#      .query("n < 2")
#  )
#
#  (
#      implicit_to_explicit_df
#      .complete(
#          "name",
#          "time",
#      )
#  )
#
#  (
#      implicit_to_explicit_df
#      # pyjanitor
#      .complete(
#          {"name": ["lynn", "zelda"]},
#          {"time": ["morning", "afternoon"]},
#          sort=True
#      )
#  )
#
#  (
#      implicit_to_explicit_df
#      # pyjanitor
#      .complete(
#          "name",
#          "time",
#          fill_value=np.nan
#      )
#  )
#
#  (
#      implicit_to_explicit_df
#      # pyjanitor
#      .complete(
#          "name",
#          "time",
#          fill_value=0,
#          explicit=False
#      )
#  )
#
#  diabetes_df.missing.missing_variable_plot()
#
#  diabetes_df[diabetes_df.columns[1:6]] = diabetes_df[diabetes_df.columns[1:6]].replace(0, np.nan)
#  diabetes_df.missing.missing_variable_plot()
#
#  (
#      diabetes_df
#      .missing.sort_variables_by_missingness()
#      .pipe(missingno.matrix)
#  )
#
#  (
#      diabetes_df
#      .missing.sort_variables_by_missingness()
#      .sort_values(by = "blood_pressure")
#      .pipe(missingno.matrix)
#  )
#
#  (
#      diabetes_df
#      .missing.sort_variables_by_missingness()
#      .sort_values("insulin")
#      .pipe(missingno.matrix)
#  )
#
#  (
#      riskfactors_df
#      .isna()
#      .replace({
#          False: "Not missing",
#          True: "Missing"
#      })
#      .add_suffix("_NA")
#      .pipe(
#          lambda shadow_matrix: pd.concat(
#              [riskfactors_df, shadow_matrix],
#              axis="columns"
#          )
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing = True)
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing=True)
#      .groupby(["weight_lbs_NA"])
#      ["age"]
#      .describe()
#      .reset_index()
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing=True)
#      .pipe(
#          lambda df: (
#              sns.displot(
#                  data=df,
#                  x="age",
#                  hue="weight_lbs_NA",
#                  kind="kde"
#              )
#          )
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing=True)
#      .pipe(
#          lambda df: (
#              sns.boxenplot(
#                  data=df,
#                  x="weight_lbs_NA",
#                  y="age",
#              )
#          )
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing=True)
#      .pipe(
#          lambda df: (
#              sns.displot(
#                  data=df,
#                  x="age",
#                  col="weight_lbs_NA",
#                  facet_kws={
#                      "sharey": False
#                  }
#              )
#          )
#      )
#  )
#
#  (
#      riskfactors_df
#      .missing
#      .bind_shadow_matrix(only_missing=True)
#      .pipe(
#          lambda df: (
#              sns.displot(
#                  data=df,
#                  x="age",
#                  col="marital_NA",
#                  row="weight_lbs_NA"
#              )
#          )
#      )
#  )
#
#  def column_fill_with_dummies(
#      column: pd.Series,
#      proportion_below: float=0.10,
#      jitter: float=0.075,
#      seed: int=42
#  ) -> pd.Series:
#
#      column = column.copy(deep=True)
#
#      # Extract values metadata.
#      missing_mask = column.isna()
#      number_missing_values = missing_mask.sum()
#      column_range = column.max() - column.min()
#
#      # Shift data
#      column_shift = column.min() - column.min() * proportion_below
#
#      # Create the "jitter" (noise) to be added around the points.
#      np.random.seed(seed)
#      column_jitter = (np.random.rand(number_missing_values) - 2) * column_range * jitter
#
#      # Save new dummy data.
#      column[missing_mask] = column_shift + column_jitter
#
#      return column
#
#  plt.figure(figsize=(10, 10))
#
#  (
#      riskfactors_df
#      .select_dtypes(
#          exclude="category"
#      )
#      .pipe(
#          lambda df: df[df.columns[df.isna().any()]]
#      )
#      .missing.bind_shadow_matrix(true_string=True, false_string=False)
#      .apply(
#          lambda column: column if "_NA" in column.name else column_fill_with_dummies(column, proportion_below=0.05, jitter=0.075)
#      )
#      .assign(
#          nullity=lambda df: df.weight_lbs_NA | df.height_inch_NA
#      )
#      .pipe(
#          lambda df: (
#              sns.scatterplot(
#                  data=df,
#                  x="weight_lbs",
#                  y="height_inch",
#                  hue="nullity"
#              )
#          )
#      )
#  )
#
#  missingno.heatmap(
#      df=riskfactors_df
#  )
#
#  missingno.dendrogram(
#      df=riskfactors_df
#  )
#
#  riskfactors_df.shape
#
#  (
#      riskfactors_df
#      .weight_lbs
#      .mean()
#  )
#
#  riskfactors_df.weight_lbs.size, riskfactors_df.weight_lbs.count()
#
#  riskfactors_df.weight_lbs.mean(skipna=False)
#
#  (
#      riskfactors_df
#      .dropna(
#          subset=["weight_lbs"],
#          how="any"
#      )
#      .shape
#  )
#
#  (
#      riskfactors_df
#      .dropna(
#          subset=["weight_lbs", "height_inch"],
#          how="any"
#      )
#      .shape
#  )
#
#  (
#      riskfactors_df
#      .dropna(
#          subset=["weight_lbs", "height_inch"],
#          how="all"
#      )
#      .shape
#  )
#
#  (
#      riskfactors_df
#      .dropna(
#          subset=["weight_lbs", "height_inch"],
#          how="any"
#      )
#      .select_columns(["weight_lbs", "height_inch"])
#      .pipe(
#          lambda df: missingno.matrix(df)
#      )
#  )
#
#  (
#      riskfactors_df
#      .dropna(
#          subset=["weight_lbs", "height_inch"],
#          how="all"
#      )
#      .select_columns(["weight_lbs", "height_inch"])
#      .pipe(
#          lambda df: missingno.matrix(df)
#      )
#  )
#
#  implicit_to_explicit_df = pd.DataFrame(
#      data={
#          "name": ["lynn", np.nan, "zelda", np.nan, "shadowsong", np.nan],
#          "time": ["morning", "afternoon", "morning", "afternoon", "morning", "afternoon",],
#          "value": [350, 310, 320, 350, 310, 320]
#      }
#  )
#
#  implicit_to_explicit_df
#
#  implicit_to_explicit_df.ffill()
#
#  plt.figure(figsize=(10, 10))
#
#  (
#      riskfactors_df
#      .select_columns("weight_lbs", "height_inch", "bmi")
#      .missing.bind_shadow_matrix(true_string=True, false_string=False)
#      .apply(
#          axis="rows",
#          func=lambda column: column.fillna(column.mean()) if "_NA" not in column.name else column
#      )
#      .pipe(
#          lambda df: (
#              sns.displot(
#                  data=df,
#                  x="weight_lbs",
#                  hue="weight_lbs_NA"
#              )
#          )
#      )
#  )
#
#  plt.figure(figsize=(10, 10))
#
#  (
#      riskfactors_df
#      .select_columns("weight_lbs", "height_inch", "bmi")
#      .missing.bind_shadow_matrix(true_string=True, false_string=False)
#      .apply(
#          axis="rows",
#          func=lambda column: column.fillna(column.mean()) if "_NA" not in column.name else column
#      )
#      .assign(
#          imputed=lambda df: df.weight_lbs_NA | df.height_inch_NA
#      )
#      .pipe(
#          lambda df: (
#              sns.scatterplot(
#                  data=df,
#                  x="weight_lbs",
#                  y="height_inch",
#                  hue="imputed"
#              )
#          )
#      )
#  )
#
#  plt.figure(figsize=(10, 10))
#
#  (
#      riskfactors_df
#      .select_columns("weight_lbs", "height_inch", "bmi")
#      .missing.bind_shadow_matrix(true_string=True, false_string=False)
#      .apply(
#          axis="rows",
#          func=lambda column: column.fillna(column.mean())
#          if "_NA" not in column.name
#          else column,
#      )
#      .pivot_longer(
#          index="*_NA"
#      )
#      .pivot_longer(
#          index=["variable", 'value'],
#          names_to="variable_NA",
#          values_to="value_NA"
#      )
#      .assign(
#          valid=lambda df: df.apply(axis="columns", func=lambda column: column.variable in column.variable_NA)
#      )
#      .query("valid")
#      .pipe(
#          lambda df: (
#              sns.displot(
#                  data=df,
#                  x="value",
#                  hue="value_NA",
#                  col="variable",
#                  common_bins=False,
#                  facet_kws={
#                      "sharex": False,
#                      "sharey": False
#                  }
#              )
#          )
#      )
)


#  session_info.show()
