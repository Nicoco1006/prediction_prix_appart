import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

surface = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110]).reshape(-1, 1)  # Superficie en m²
prix = np.array([15000, 22000, 30000, 35000, 45000, 52000, 60000, 68000, 75000, 83000])  # Prix en €

X_train, X_test, y_train, y_test = train_test_split(surface, prix, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")

plt.scatter(surface, prix, color="blue", label="Données réelles")
plt.plot(X_test, y_pred, color="red", label="Prédiction")
plt.xlabel("Surface (m²)")
plt.ylabel("Prix (€)")
plt.legend()
plt.show()
