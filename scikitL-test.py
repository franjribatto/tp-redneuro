import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

dataBase = pd.read_csv("heart.csv")
df_dataBase = dataBase[["age", "cp", "trtbps", "chol", "fbs", "thalachh", "exng"]]

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

df_desc = df_dataBase.describe().T
df_cleanNorm = (df_clean.iloc[:, :-1] - df_desc['mean'][:-1]) / df_desc['std'][:-1]
df_cleanNorm['exng'] = df_dataBase['exng']

df_cleanNorm.describe()

data = df_cleanNorm

X = data.iloc[:, :6].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y configurar la red neuronal
model = MLPClassifier(
    hidden_layer_sizes=(6,6,),      # Una capa oculta con 6 neuronas
    activation='relu',            # Activación ReLU en la capa oculta
    solver='sgd',                 # Descenso de gradiente estocástico como optimizador
    max_iter=40000,                 # Número máximo de iteraciones
    random_state=42,
    learning_rate_init=0.01,      # Tasa de aprendizaje inicial
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Precisión del modelo: {accuracy * 100:.2f}%')
print('Reporte de clasificación:')
print(classification_rep)