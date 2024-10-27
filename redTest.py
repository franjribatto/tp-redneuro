import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataBase = pd.read_csv("heart.csv")

df_dataBase = dataBase[["age","cp","trtbps","chol","fbs","thalachh","exng"]]

df_desc = df_dataBase.describe().T

df_descNorm = (df_dataBase - df_desc['mean']) / df_desc['std']



labels = df_descNorm.columns.to_list()
fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.imshow((df_descNorm.corr().to_numpy()), cmap= 'coolwarm', vmin= -1, vmax= 1)
ax.set_xticks(np.arange(0, len(labels)))
ax.set_xticklabels(df_descNorm.columns.to_list(), rotation=45, ha='right')
ax.set_yticks(np.arange(0, len(labels)))
ax.set_yticklabels(df_descNorm.columns.to_list(), ha='right');

for i in range(len(labels)):
    for j in range(len(labels)):
        value = df_descNorm.corr().iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')

ax.set_title('Correlación de las variables')
plt.show()


# Función para reemplazar outliers con la media
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

df_clean.describe()

all_inputs = df_clean.iloc[:, 0:6].values
all_outputs = df_clean.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(all_inputs, all_outputs, test_size=1/3)

w_hidden = np.random.rand(6, 6)
w_output = np.random.rand(1, 6)

b_hidden = np.random.rand(6, 1)
b_output = np.random.rand(1, 1)

relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

def forward_propagation(X):
  Z1 = w_hidden @ X + b_hidden
  A1 = relu(Z1)
  Z2 = w_output @ A1 + b_output
  A2 = logistic(Z2)
  return Z1, A1, Z2, A2

d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x))**2

def back_propagation(X, Y, Z1, A1, Z2, A2):
  dC_dA2 = 2 * A2 - 2 * Y
  dA2_dZ2 = d_logistic(Z2)
  dZ2_dA1 = w_output
  dZ2_dW2 = A1
  dZ2_dB2 = 1
  dA1_dZ1 = d_relu(Z1)
  dZ1_dW1 = X
  dZ1_dB1 = 1

  dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T

  dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2

  dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1

  dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T

  dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

  return dC_dW1, dC_dB1, dC_dW2, dC_dB2

n = X_train.shape[0]

for i in range(100_000):
  idx = np.random.choice(n, 1, replace=False)
  X_sample = X_train[idx].transpose()
  Y_sample = Y_train[idx]

  Z1, A1, Z2, A2 = forward_propagation(X_sample)

  dW1, dB1, dW2, dB2 = back_propagation(X_sample, Y_sample, Z1, A1, Z2, A2)

  w_hidden -= 0.01 * dW1
  b_hidden -= 0.01 * dB1
  w_output -= 0.01 * dW2
  b_output -= 0.01 * dB2

test_predictions = forward_propagation(X_test.transpose())[3] # me interesa solo la capa de salida, A2
test_comparisons = np.equal((test_predictions >= .5).flatten().astype(int), Y_test)
accuracy = sum(test_comparisons.astype(int) / X_test.shape[0])
print("ACCURACY: ", accuracy)