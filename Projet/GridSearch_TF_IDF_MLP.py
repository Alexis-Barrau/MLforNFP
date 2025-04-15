import logging
import pandas as pd
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Configuration du log
logging.basicConfig(
    filename='Sorties/GridSearch_TF_IDF_MLP_output.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Chargement des données
Isot_true_df = pd.read_csv("data/True.csv")
Isot_fake_df = pd.read_csv("data/Fake.csv")
Isot_true_df["label"] = 0
Isot_fake_df["label"] = 1
Isot_data = pd.concat([Isot_true_df, Isot_fake_df], ignore_index=True)
Isot = Isot_data[['text', 'label']]

# Split train/test
X = Isot['text']
y = Isot['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Affichage des infos sur les données
print(f"Nombre total de documents : {len(X)}")
print(f"Train set : {len(X_train)} documents")
print(f"Test set : {len(X_test)} documents")
print(f"Répartition des classes (train set) :\n{y_train.value_counts()}")
logging.info(f"Nombre total de documents : {len(X)}")
logging.info(f"Train size : {len(X_train)} | Test size : {len(X_test)}")
logging.info(f"Répartition des classes (train) :\n{y_train.value_counts().to_string()}")

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('svd', TruncatedSVD()),
    ('mlp', MLPClassifier(max_iter=500, early_stopping=True)
)
])


param_grid = {
    'svd__n_components': [100, 200],         # nombre de dimensions après réduction
    'mlp__hidden_layer_sizes': [(100,), (50, 50)],  # une ou deux couches
    'mlp__activation': ['relu', 'tanh'],
    'mlp__alpha': [0.0001, 0.001],           # régularisation (L2)
    'mlp__learning_rate_init': [0.001, 0.01]
}
# Nombre total de combinaisons
total_configs = 1
for param, values in param_grid.items():
    total_configs *= len(values)
print(f" Nombre total de combinaisons testées : {total_configs}")
logging.info(f"Nombre total de combinaisons testées : {total_configs}")

# GridSearch
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    verbose=2,
    n_jobs=70
)

# Entraînement
start_time = time.time()
print("Début de l'entraînement GridSearchCV...")
logging.info("Début de l'entraînement avec GridSearchCV...")
grid_search.fit(X_train, y_train)
train_duration = time.time() - start_time
print(f"Entraînement terminé en {train_duration:.2f} secondes.")
logging.info(f"Durée entraînement GridSearchCV : {train_duration:.2f} secondes.")

# Meilleurs hyperparamètres
print("Meilleurs hyperparamètres :")
print(grid_search.best_params_)
logging.info("Best parameters found:")
logging.info(grid_search.best_params_)

# Prédiction
start_time = time.time()
print("Prédiction sur le test set...")
y_pred = grid_search.predict(X_test)
predict_duration = time.time() - start_time
print(f"Prédictions terminées en {predict_duration:.2f} secondes.")
logging.info(f"Durée prédictions : {predict_duration:.2f} secondes.")

# Évaluation
print("Évaluation du modèle...")
report = classification_report(y_test, y_pred)
print(report)
logging.info("Rapport de classification sur le test set :")
logging.info(report)

print("Fin du script.")
logging.info("Fin du script.")