# Importamos las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargamos el dataset
heart_data = pd.read_csv('heart.csv')

# Separación de características (X) y etiquetas (y)
X = heart_data.drop(columns='output')  # Características
y = heart_data['output']  # Etiquetas

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Imprimimos los resultados
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
print('Reporte de clasificación:')
print(classification_rep)