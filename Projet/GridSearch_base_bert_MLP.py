import pandas as pd
import numpy as np
import time
import logging
import os
import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

# Mise en place du logging
log_filename = "logs/GridSearch_base_Bert_MLP.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def log_and_print(msg):
    print(msg)
    logging.info(msg)


# Utilisation du GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_and_print(f"Using device: {device}")

# On charge les données d'entrainement déjà Embedded (ISOT)
log_and_print("Chargement des embbeddings du dataset de train")
embeddings_isot_base_bert = pd.read_csv("data/Embedded/embeddings_isot_base_bert.csv")  # Exemple

X = embeddings_isot_base_bert.drop(columns=['label']).values
y = embeddings_isot_base_bert['label'].values

log_and_print("Train/Test Split")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On définie notre MLP PyTorch
class MLPModule(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=1):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)

net = NeuralNetClassifier(
    MLPModule,
    module__input_dim=X.shape[1],
    max_epochs=20,
    lr=0.001,
    batch_size=64,
    device=device,
    iterator_train__shuffle=True,
)

# On effectue le GridSearch
param_grid = {
    'lr': [0.001, 0.0005],
    'module__hidden_dim': [64, 128],
    'module__num_layers': [1, 2],
    'max_epochs': [20, 30],
}

log_and_print("Début du GridSearch")
start_grid = time.time()
grid_search = GridSearchCV(net, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
end_grid = time.time()

log_and_print(f"GridSearch took {end_grid - start_grid:.2f} seconds.")
log_and_print(f"Best parameters: {grid_search.best_params_}")
log_and_print(f"Best CV score: {grid_search.best_score_:.4f}")

# Application sur le test Set
best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

acc_test = accuracy_score(y_test, y_pred_test)
conf_test = confusion_matrix(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test)

log_and_print(f"Test Accuracy: {acc_test:.4f}")
log_and_print("Confusion Matrix:\n" + str(conf_test))

# On enregistre le model
os.makedirs("Modele", exist_ok=True)
dump(best_model, "Modele/pytorch_base_bert_mlp.joblib")
log_and_print("Modele sauvegardé sous Modele/pytorch_base_bert_mlp.joblib")
log_and_print("Classification Report:\n" + report_test)

# Evaluation sur Fake_news
log_and_print("Chargement de Fake News")
embeddings_fake_news_base_bert = pd.read_csv("data/Embedded/embeddings_fake_news_base_bert.csv")
X_fake_news = embeddings_fake_news_base_bert.drop(columns=['label']).values.astype(np.float32)
y_fake_news = embeddings_fake_news_base_bert['label'].values.astype(np.int64)

y_pred_fake_news = best_model.predict(X_fake_news)

acc_fake_news = accuracy_score(y_fake_news, y_pred_fake_news)
conf_fake_news = confusion_matrix(y_fake_news, y_pred_fake_news)
report_fake_news = classification_report(y_fake_news, y_pred_fake_news)

log_and_print(f"Fake News Accuracy: {acc_fake_news:.4f}")
log_and_print("Confusion Matrix (Fake News):\n" + str(conf_fake_news))
log_and_print("Classification Report (Fake News):\n" + report_fake_news)

# Evaluation sur Fake_nreal
log_and_print("Chargement de Fake Real")
embeddings_fake_real_base_bert = pd.read_csv("data/Embedded/embeddings_fake_real_base_bert.csv")
X_fake_real = embeddings_fake_real_base_bert.drop(columns=['label']).values.astype(np.float32)
y_fake_real = embeddings_fake_real_base_bert['label'].values.astype(np.int64)

y_pred_fake_real = best_model.predict(X_fake_real)

acc_fake_real = accuracy_score(y_fake_real, y_pred_fake_real)
conf_fake_real = confusion_matrix(y_fake_real, y_pred_fake_real)
report_fake_real = classification_report(y_fake_real, y_pred_fake_real)

log_and_print(f"Fake Real Accuracy: {acc_fake_real:.4f}")
log_and_print("Confusion Matrix (FFake Real):\n" + str(conf_fake_real))
log_and_print("Classification Report (Fake Real):\n" + report_fake_real)

log_and_print("Script finished.")