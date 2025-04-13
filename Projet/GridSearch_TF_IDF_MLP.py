import logging # Pour enregistrer les sorties
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

logging.basicConfig(
    filename='Sorties/GridSearch_TF_IDF_MLP_output.log',  # Nom du fichier où les logs seront enregistrés
    level=logging.INFO,           # Niveau de log (tout au-dessus de INFO sera enregistré)
    format='%(asctime)s - %(message)s'  # Format du log (inclut la date et l'heure)
)


# Mise en forme dataset ISOT https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset/
Isot_true_df = pd.read_csv("data/True.csv")
Isot_fake_df = pd.read_csv("data/Fake.csv")

#Création d'un dataset unique
Isot_true_df["label"] = 0  # Vraie news
Isot_fake_df["label"] = 1  # Fake news
Isot_data = pd.concat([Isot_true_df, Isot_fake_df], ignore_index=True)
Isot = Isot_data[['text', 'label']]

# Train-Test  split
X = Isot['text']  # Les articles/textes
y = Isot['label']  # Les labels (1 = Fake)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('svd', TruncatedSVD()),
    ('mlp', MLPClassifier(max_iter=300))  # Tu peux ajuster max_iter si l'entraînement est long
])

param_grid = {
    'svd__n_components': [100, 200],         # nombre de dimensions après réduction
    'mlp__hidden_layer_sizes': [(100,), (50, 50)]#,  # une ou deux couches
    #'mlp__activation': ['relu', 'tanh'],
    #'mlp__alpha': [0.0001, 0.001],           # régularisation (L2)
    #'mlp__learning_rate_init': [0.001, 0.01]
}

# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,              # 5-fold cross-validation
    verbose=2,         # affiche les étapes
    n_jobs=-1          # utilise tous les cœurs disponibles
)


# Entraînement sur ton jeu de données
logging.info("Début de l'entraînement avec GridSearchCV...")
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
logging.info("Best parameters found:")
logging.info(grid_search.best_params_)

# Prédictions sur test set
y_pred = grid_search.predict(X_test)

# Évaluer le modèle
logging.info("Rapport de classification sur le test set :")
logging.info(classification_report(y_test, y_pred))

logging.info("Fin du script.")