# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Datos
datos = [
    {"Edad": 24, "Uso_datos": 2.5, "Linea_fija": "No", "Acepto": "No"},
    {"Edad": 38, "Uso_datos": 6.0, "Linea_fija": "Sí", "Acepto": "Sí"},
    {"Edad": 29, "Uso_datos": 3.0, "Linea_fija": "No", "Acepto": "No"},
    {"Edad": 45, "Uso_datos": 8.0, "Linea_fija": "Sí", "Acepto": "Sí"},
    {"Edad": 52, "Uso_datos": 7.5, "Linea_fija": "Sí", "Acepto": "Sí"},
    {"Edad": 33, "Uso_datos": 4.0, "Linea_fija": "No", "Acepto": "No"},
    {"Edad": 41, "Uso_datos": 5.5, "Linea_fija": "Sí", "Acepto": "Sí"},
    {"Edad": 27, "Uso_datos": 2.0, "Linea_fija": "No", "Acepto": "No"},
    {"Edad": 36, "Uso_datos": 6.5, "Linea_fija": "Sí", "Acepto": "Sí"},
    {"Edad": 31, "Uso_datos": 3.5, "Linea_fija": "No", "Acepto": "No"}
]

df = pd.DataFrame(datos)

# Preparar datos
le = LabelEncoder()
df['Linea_fija_cod'] = le.fit_transform(df['Linea_fija'])
df['Acepto_cod'] = le.fit_transform(df['Acepto'])

# Entrenar modelo
X = df[['Edad', 'Uso_datos', 'Linea_fija_cod']]
y = df['Acepto_cod']

modelo = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
modelo.fit(X, y)

# Graficar árbol
plt.figure(figsize=(20,10))
plot_tree(
    modelo,
    feature_names=['Edad', 'Uso_datos', 'Línea_fija'],
    class_names=['Rechaza', 'Acepta'],
    filled=True,
    rounded=True,
    fontsize=12,
)
plt.title("Árbol de Decisión - Predicción de Aceptación de Oferta", fontsize=16)
plt.show()