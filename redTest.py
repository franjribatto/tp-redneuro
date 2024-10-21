import pandas as pd
import numpy as np


dataBase = pd.read_csv("heart.csv")

df_dataBase = dataBase[["age","cp","trtbps","chol","fbs","thalachh"]]

df_desc = df_dataBase.describe().T

df_descNorm = (df_dataBase - df_desc['mean']) / df_desc['std']


print(df_descNorm.corr())