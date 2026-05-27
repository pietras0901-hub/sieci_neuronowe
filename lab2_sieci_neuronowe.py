import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# część 1 Podstawowy model
# dane wejściowe kolor i waga
X = np.array([
    [1, 200], # jabłko
    [0, 120], # banan
    [1, 180],
    [0, 110],
    [1, 220],
    [0, 130],
])

y = np.array([0, 1, 0, 1, 0, 1]) # 0=jabłko, 1=banan

# tworzenie i trenowanie modelu
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500, random_state=42)
model.fit(X, y)

# ocena dokładności
accuracy = model.score(X, y)
print(f"Dokładność: {accuracy:.2f}")

# predykcja dla nowego owocu
new_sample = np.array([[1, 190]])
prediction = model.predict(new_sample)

if prediction[0] == 0:
    print("jabłko")
else:
    print("banan")

# wykres granicy decyzji
def plot_decision_boundary(X, y, model):
    X_min, X_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 20, X[:, 1].max() + 20
    XX, yy = np.meshgrid(
        np.linspace(X_min, X_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model.predict(np.c_[XX.ravel(), yy.ravel()])
    Z = Z.reshape(XX.shape)
    plt.contourf(XX, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=120, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Kolor (1=czerwony, 0=żółty)')
    plt.ylabel('Waga (g)')
    plt.title('Granica decyzyjna sieci neuronowej')
    plt.show()

plot_decision_boundary(X, y, model)


# część 2 rozbudowa modelu

# Dane
X2 = np.array([
    [1, 200], [1, 180], [1, 220], [1, 190],
    [0, 120], [0, 130], [0, 110], [0, 115],
])
y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# wyświetlamy dane w tabelce
df = pd.DataFrame(X2, columns=['Kolor (1=czerwony)', 'Waga (g)'])
df['Klasa'] = ['Jabłko' if c == 0 else 'Banan' for c in y2]
print(df)


# wizualizacja danych
plt.figure(figsize=(7, 5))
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='coolwarm', s=120, edgecolors='k')
plt.xlabel('Kolor (1=czerwony, 0=żółty)')
plt.ylabel('Waga (g)')
plt.title('Wizualizacja: Jabłko vs Banan')
plt.colorbar(label='0=Jabłko, 1=Banan')
plt.show()

# podział danych na trening (70%) i test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42
)
print(f"Trening: {X_train.shape[0]} próbek, Test: {X_test.shape[0]} próbek")

# trenowanie dwóch modeli
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
    best_model = mlp
    best_preds = mlp_preds
    best_name = "MLP"
else:
    best_model = knn
    best_preds = knn_preds
    best_name = "KNN"

print(f"Najlepszy model: {best_name}")

cm = confusion_matrix(y_test, best_preds)
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=['Jabłko', 'Banan']).plot(cmap='Blues')
plt.title(f'Macierz Pomyłek – {best_name}')
plt.show()

# optymalizacja 
param_grid = {
    'hidden_layer_sizes': [(3,), (5,), (10,)],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(
    MLPClassifier(max_iter=500, random_state=42),
    param_grid, cv=2
)
grid_search.fit(X_train, y_train)

print(f"Najlepsze parametry: {grid_search.best_params_}")
print(f"Dokładność po GridSearch: {grid_search.score(X_test, y_test):.2f}")


# predykcja dla nowego owocu
new_fruit = np.array([[1, 195]])
predicted_class = grid_search.predict(new_fruit)

if predicted_class[0] == 0:
    print("Nowy owoc to: Jabłko ")
else:
    print("Nowy owoc to: Banan ")
