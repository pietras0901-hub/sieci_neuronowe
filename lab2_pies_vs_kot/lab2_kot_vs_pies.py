import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# dane – koty i psy
X = np.array([
    # waga, wysokość, szczeka
    [4,  25, 0],  # kot
    [5,  28, 0],  # kot
    [3,  22, 0],  # kot
    [4,  24, 0],  # kot
    [6,  27, 0],  # kot
    [15, 55, 1],  # pies
    [20, 60, 1],  # pies
    [10, 45, 1],  # pies
    [12, 50, 1],  # pies
    [8,  40, 1],  # pies
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# 0 = kot, 1 = pies

# wyświetlamy tabelkę
df = pd.DataFrame(X, columns=['Waga (kg)', 'Wysokość (cm)', 'Szczeka (0/1)'])
df['Klasa'] = ['Kot' if c == 0 else 'Pies' for c in y]
print(df)
print(f"\nKształt danych: {X.shape}")

# wizualizacja – waga vs wysokość
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=120, edgecolors='k')
plt.xlabel('Waga (kg)')
plt.ylabel('Wysokość (cm)')
plt.title('Wizualizacja: Kot vs Pies')
plt.colorbar(label='0=Kot, 1=Pies')
plt.show()

#podział danych 70% trening 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"Trening: {X_train.shape[0]} próbek, Test: {X_test.shape[0]} próbek")

# trenowanie modeli
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)

mlp.fit(X_train, y_train)
knn.fit(X_train, y_train)

print("Modele wytrenowane!")

# ocena modeli
mlp_preds = mlp.predict(X_test)
knn_preds = knn.predict(X_test)

print(f"Dokładność MLP: {accuracy_score(y_test, mlp_preds):.2f}")
print(f"Dokładność KNN: {accuracy_score(y_test, knn_preds):.2f}")

# macierz pomyłek najlepszego modelu
if accuracy_score(y_test, mlp_preds) >= accuracy_score(y_test, knn_preds):
    best_preds = mlp_preds
    best_name = "MLP"
else:
    best_preds = knn_preds
    best_name = "KNN"

print(f"Najlepszy model: {best_name}")

cm = confusion_matrix(y_test, best_preds)
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=['Kot', 'Pies']).plot(cmap='Blues')
plt.title(f'Macierz Pomyłek – {best_name}')
plt.show()

# optymalizacja hiperparametrów
param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (10,)],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid, cv=2
)
grid_search.fit(X_train, y_train)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Dokładność po GridSearch: {grid_search.score(X_test, y_test):.2f}")

# predykcja dla nowego zwierzaka
# 6 kg, 30 cm, nie szczeka
new_animal = np.array([[6, 30, 0]])
predicted_class = grid_search.predict(new_animal)

if predicted_class[0] == 0:
    print("Nowe zwierzę to: Kot")
else:
    print("Nowe zwierzę to: Pies ")