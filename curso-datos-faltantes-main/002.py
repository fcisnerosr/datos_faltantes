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

