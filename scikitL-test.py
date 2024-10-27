import pandas as pd
# cargar datos
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('heart.csv', delimiter=",")
df_dataBase = df[["age","cp","trtbps","chol","fbs","thalachh","exng"]]
df_desc = df_dataBase.describe().T
df_descNorm = (df_dataBase - df_desc['mean']) / df_desc['std']

def df_cleanner(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Definir límites inferior y superior
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Calcular la media de la columna
        mean_value = df[column].mean()

        # Reemplazar outliers por la media
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = mean_value

    return df

# Reemplazar outliers por la media
df_clean = df_cleanner(df_descNorm)

# Extraer variables de entrada (todas las filas, todas las columnas menos la última)
# Nota que deberíamos hacer algún escalado lineal aquí
X = df_clean.iloc[:, 0:6].values

# Extraer columna de salida (todas las filas, última columna)
Y = df_clean.iloc[:, -1].values

# Separar los datos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(6,),
                   activation='relu',
                   max_iter=100_000,
                   learning_rate_init=.05)

nn.fit(X_train, Y_train)

y_pred = nn.predict(X_test)

# Imprimir pesos y sesgos
print(nn.coefs_)
print(nn.intercepts_)

print("Puntaje del conjunto de entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje del conjunto de prueba: %f" % nn.score(X_test, Y_test))