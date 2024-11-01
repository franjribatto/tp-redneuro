##Importamos las librerias a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fijar la semilla para obtener resultados consistentes
np.random.seed(42)

dataBase = pd.read_csv("heart.csv")
df_dataBase = dataBase[["age", "cp", "trtbps", "chol", "fbs", "thalachh", "exng"]]

plt.figure(figsize=(10, 3))
plt.boxplot(df_dataBase.iloc[:, :6].values, vert=True, patch_artist=True)

# Agregar etiquetas y título
plt.title("Boxplot antes de la limpieza")
plt.xlabel("Columnas")
plt.ylabel("Valores")
plt.xticks(range(1, 7), df_dataBase.columns[:6])  # Etiquetas de columnas (1-6)
plt.show()

#Realizamos este boxplot con el fin de poder demostrar que es necesario limpiar el dataframe de
#Datos atipicos porque como se puede ver son bastantes y pueden reducir la precision de la ia

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
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] =float( mean_value)

    return df

# Reemplazar outliers por la media
df_clean = df_cleanner(df_dataBase)

#Realizamos un nuevo boxplot para compararlo con el anterior

plt.figure(figsize=(10, 3))
plt.boxplot(df_clean.iloc[:, :6].values, vert=True, patch_artist=True)

plt.title("Boxplot despues de la limpieza")
plt.xlabel("Columnas")
plt.ylabel("Valores")
plt.xticks(range(1, 7), df_clean.columns[:6])
plt.show()

#Como podemos observar, la base de datos ahora tiene muchos menos datos fuera de los "bigotes" del
#boxplot indicando que por lo menos ahora no hay tantos datos atipicos afectando la sanidad de la misma

#Ahora realizamos un grafico y un analisis de correlacion entre los datos
#utilizando los datos crudos de la base de datos

labels = df_dataBase.columns.to_list()
fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.imshow((df_dataBase.corr().to_numpy()), cmap= 'coolwarm', vmin= -1, vmax= 1)
ax.set_xticks(np.arange(0, len(labels)))
ax.set_xticklabels(df_dataBase.columns.to_list(), rotation=45, ha='right')
ax.set_yticks(np.arange(0, len(labels)))
ax.set_yticklabels(df_dataBase.columns.to_list(), ha='right')

for i in range(len(labels)):
    for j in range(len(labels)):
        value = df_dataBase.corr().iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

ax.set_title('Correlación de las variables')
plt.show()

#Procedemos a realizar la "normalizacion"

df_desc = df_dataBase.describe().T
df_cleanNorm = (df_clean.iloc[:, :-1] - df_desc['mean'][:-1]) / df_desc['std'][:-1]
df_cleanNorm['exng'] = df_dataBase['exng']

df_cleanNorm.describe()

#Ahora ya realizado el analisis y la ejecucion de la limpieza de la base de datos, incuyendo
#la normalizacion de los datos existentes en esta procedemos a darle una estructura a la red neuronal

# Separación de datos
all_inputs = df_cleanNorm.iloc[:, :-1].values
all_outputs = df_cleanNorm.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/5, random_state=42)

# Inicialización de pesos
w_hidden = np.random.rand(6, 6) * 0.01
w_output = np.random.rand(1, 6) * 0.01
b_hidden = np.random.rand(6, 1) * 0.01
b_output = np.random.rand(1, 1) * 0.01

# Funciones de activación
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))
d_relu = lambda x: (x > 0).astype(float)
d_logistic = lambda x: logistic(x) * (1 - logistic(x))

# Propagación hacia adelante
def forward_propagation(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

# Propagación hacia atrás
def back_propagation(X, Y, Z1, A1, Z2, A2):
    dC_dA2 = 2 * (A2 - Y)
    dA2_dZ2 = d_logistic(Z2)
    dZ2_dA1 = w_output
    dC_dW2 = dC_dA2 * dA2_dZ2 @ A1.T
    dC_dB2 = dC_dA2 * dA2_dZ2
    dC_dA1 = w_output.T @ (dC_dA2 * dA2_dZ2)
    dC_dW1 = dC_dA1 * d_relu(Z1) @ X.T
    dC_dB1 = dC_dA1 * d_relu(Z1)

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

#Una vez planteada la estructura y los calculos que se utilizan procedemos a entrenar la misma
#Y calcular su precision

train_accuracies = []
test_accuracies = []

# Entrenamiento
learning_rate = 0.01
epochs = 40000
n = X_train.shape[0]

for i in range(epochs):
    idx = np.random.randint(0, n)
    X_sample = X_train[idx].reshape(-1, 1)
    Y_sample = np.array([[Y_train[idx]]])

    Z1, A1, Z2, A2 = forward_propagation(X_sample)
    dW1, dB1, dW2, dB2 = back_propagation(X_sample, Y_sample, Z1, A1, Z2, A2)

    w_hidden -= learning_rate * dW1
    b_hidden -= learning_rate * dB1
    w_output -= learning_rate * dW2
    b_output -= learning_rate * dB2

    # Calcular la precisión cada 1000 épocas
    if i % 1000 == 0:
        # Evaluar precisión en el conjunto de entrenamiento
        train_predictions = forward_propagation(X_train.T)[3]
        train_predictions = (train_predictions >= 0.5).astype(int).flatten()
        train_accuracy = np.mean(train_predictions == Y_train)
        train_accuracies.append(train_accuracy)

        # Evaluar precisión en el conjunto de prueba
        test_predictions = forward_propagation(X_test.T)[3]
        test_predictions = (test_predictions >= 0.5).astype(int).flatten()
        test_accuracy = np.mean(test_predictions == Y_test)
        test_accuracies.append(test_accuracy)

# Graficamos la precisión en los datos de entrenamiento y prueba
plt.plot(range(0, epochs, 1000), train_accuracies, label="Precisión Entrenamiento")
plt.plot(range(0, epochs, 1000), test_accuracies, label="Precisión Prueba")
plt.xlabel("Iteraciones")
plt.ylabel("Precisión")
plt.title("Precisión del Modelo en Datos de Entrenamiento y Prueba")
plt.legend()
plt.show()

# Evaluación final en el conjunto de prueba
test_predictions = forward_propagation(X_test.T)[3]
test_predictions = (test_predictions >= 0.5).astype(int).flatten()
accuracy = np.mean(test_predictions == Y_test)
print("ACCURACY:", accuracy)