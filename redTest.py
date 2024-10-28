import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importar y normalizar datos
dataBase = pd.read_csv("heart.csv")
df_dataBase = dataBase[["age", "cp", "trtbps", "chol", "fbs", "thalachh", "exng"]]
df_desc = df_dataBase.describe().T
df_descNorm = (df_dataBase.iloc[:, :-1] - df_desc['mean'][:-1]) / df_desc['std'][:-1]
df_descNorm['exng'] = df_dataBase['exng']

# Función para reemplazar outliers con la media
def df_cleanner(df):
    for column in df.columns[:-1]:  # Limitar solo a entradas
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mean_value = df[column].mean()
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = mean_value
    return df

df_clean = df_cleanner(df_descNorm)

# Separación de datos
all_inputs = df_clean.iloc[:, :-1].values
all_outputs = df_clean.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=0.2, random_state=42)

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

# Entrenamiento
learning_rate = 0.01
epochs = 50000
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

# Evaluación en el conjunto de prueba
test_predictions = forward_propagation(X_test.T)[3]
test_predictions = (test_predictions >= 0.5).astype(int).flatten()
accuracy = np.mean(test_predictions == Y_test)
print("ACCURACY:", accuracy)
