import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import logging
import json
from datetime import datetime
import time

# Configuration du système de logging
def setup_logger(log_dir="logs"):
    # Créer le répertoire de logs s'il n'existe pas
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Nom du fichier de log basé sur la date et l'heure actuelles
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"bert_fine_tuned_{timestamp}.log")
    
    # Configuration du logger
    logger = logging.getLogger("BERT_FakeNews")
    logger.setLevel(logging.INFO)
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Handler pour le fichier
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Ajouter les handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_filename

# Initialiser le logger
logger, log_file = setup_logger()

# Configuration des hyperparamètres
CONFIG = {
    "MAX_LEN": 512,  
    "BATCH_SIZE": 16,
    "EPOCHS": 3,
    "LEARNING_RATE": 3e-5, 
    "RANDOM_SEED": 13101990,
    "MODEL_NAME": "bert-base-uncased",
    "OUTPUT_DIR": "./Modele/bert_fine_tuned/",
    "USE_MIXED_PRECISION": True,  
}

output_dir = CONFIG["OUTPUT_DIR"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Enregistrer la configuration
logger.info("Configuration des hyperparamètres:")
logger.info(json.dumps(CONFIG, indent=2))

# Définir le device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Utilisation de l'appareil: {DEVICE}")

# Chargement des données ISOT
def load_isot_dataset(fake_news_path, true_news_path):
    logger.info(f"Chargement des données depuis {fake_news_path} et {true_news_path}")
    
    # Charger les fake news
    fake_df = pd.read_csv(fake_news_path)
    fake_df['label'] = 1  
    logger.info(f"Données fake news chargées: {len(fake_df)} exemples")
    
    # Charger les vraies news
    true_df = pd.read_csv(true_news_path)
    true_df['label'] = 0  
    logger.info(f"Données true news chargées: {len(true_df)} exemples")
    
    # Combiner les deux datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    return df[['text', 'label']]

# Chemin des fichiers ISOT
fake_news_path = "data/Fake.csv"
true_news_path = "data/True.csv"

# Charger les données
df = load_isot_dataset(fake_news_path, true_news_path)

# Afficher un aperçu des données
logger.info(f"Total d'exemples: {len(df)}")
logger.info(f"Distribution des labels: {df['label'].value_counts().to_dict()}")

# Diviser en train et validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG["RANDOM_SEED"], stratify=df['label'])

logger.info(f"Exemples d'entraînement: {len(train_df)}")
logger.info(f"Exemples de validation: {len(val_df)}")

# Charger le tokenizer
logger.info(f"Chargement du tokenizer {CONFIG['MODEL_NAME']}")
tokenizer = BertTokenizer.from_pretrained(CONFIG["MODEL_NAME"])

# Classe pour le dataset BERT
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Créer les datasets
logger.info("Création des datasets")
train_dataset = NewsDataset(
    texts=train_df['text'].values,
    labels=train_df['label'].values,
    tokenizer=tokenizer,
    max_len=CONFIG["MAX_LEN"]
)

val_dataset = NewsDataset(
    texts=val_df['text'].values,
    labels=val_df['label'].values,
    tokenizer=tokenizer,
    max_len=CONFIG["MAX_LEN"]
)

# Créer les dataloaders
logger.info("Création des dataloaders")
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=CONFIG["BATCH_SIZE"]
)

val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=CONFIG["BATCH_SIZE"]
)

# Charger le modèle BERT pré-entraîné
logger.info(f"Chargement du modèle {CONFIG['MODEL_NAME']}")
model = BertForSequenceClassification.from_pretrained(
    CONFIG["MODEL_NAME"],
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

model.to(DEVICE)

# Préparer l'optimiseur
logger.info("Initialisation de l'optimiseur")
optimizer = AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], eps=1e-8)

# Configuration pour mixed precision si activé
if CONFIG["USE_MIXED_PRECISION"] and torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    logger.info("Mixed precision activée")
else:
    scaler = None
    logger.info("Mixed precision désactivée")

# Fonction d'entraînement
def train_epoch(model, dataloader, optimizer, device, epoch_num, scaler=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    batch_times = []
    
    logger.info(f"Début de l'epoch {epoch_num}")
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Transférer batch au device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass avec mixed precision si activé
        if scaler:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass avec scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass standard
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass standard
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Calculer le temps par batch
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        total_loss += loss.item()
        
        # Log tous les 100 batchs
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f} - Temps: {batch_time:.2f}s")
    
    epoch_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times)
    
    logger.info(f"Fin de l'epoch {epoch_num} - Temps total: {epoch_time:.2f}s - Temps moyen par batch: {avg_batch_time:.2f}s")
    
    return total_loss / len(dataloader)

# Fonction d'évaluation
def evaluate(model, dataloader, device, phase="validation"):
    model.eval()
    predictions = []
    true_labels = []
    total_eval_time = 0
    
    logger.info(f"Début de l'évaluation ({phase})")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()
            
            # Transférer batch au device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            
            batch_end = time.time()
            total_eval_time += (batch_end - batch_start)
    
    # Calculer les métriques
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)

    cm = confusion_matrix(true_labels, predictions)
    logger.info(f"Matrice de confusion ({phase}):\n{cm}")
    
    eval_time = time.time() - start_time
    logger.info(f"Évaluation terminée - Temps total: {eval_time:.2f}s")
    logger.info(f"Accuracy ({phase}): {accuracy:.4f}")
    logger.info(f"Rapport de classification ({phase}):\n{report}")
    
    return accuracy, report, predictions, true_labels

# Fonction pour sauvegarder les métriques et hyperparamètres
def save_metrics(metrics, output_dir, filename="metrics.json"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Métriques sauvegardées dans {filepath}")

# Entraînement du modèle
logger.info("Début de l'entraînement...")
training_metrics = {
    "config": CONFIG,
    "epochs": []
}

for epoch in range(CONFIG["EPOCHS"]):
    epoch_start_time = time.time()
    epoch_num = epoch + 1
    
    # Entraînement
    train_loss = train_epoch(model, train_dataloader, optimizer, DEVICE, epoch_num, scaler)
    logger.info(f"Epoch {epoch_num}/{CONFIG['EPOCHS']} - Train loss: {train_loss:.4f}")
    
    # Évaluation
    val_accuracy, val_report, val_preds, val_true = evaluate(model, val_dataloader, DEVICE, phase="validation")

    cm_val = confusion_matrix(val_true, val_preds)
    logger.info(f"Matrice de confusion (validation):\n{cm_val}")

    val_cm_path = os.path.join(output_dir, "confusion_matrix_validation.txt")
    np.savetxt(val_cm_path, cm_val, fmt='%d')
    logger.info(f"Matrice de confusion validation sauvegardée à {val_cm_path}")
    
    # Enregistrer les métriques de l'époque
    epoch_metrics = {
        "epoch": epoch_num,
        "train_loss": train_loss,
        "val_accuracy": val_accuracy,
        "val_report": val_report,
        "val_confusion_matrix": cm_val.tolist(),
        "epoch_time": time.time() - epoch_start_time
    }
    training_metrics["epochs"].append(epoch_metrics)

# Enregistrer le modèle
logger.info(f"Sauvegarde du modèle dans {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Sauvegarder les métriques d'entraînement
save_metrics(training_metrics, output_dir)

# Fonction pour tester sur un nouveau dataset
def test_on_new_dataset(model, tokenizer, test_csv_path, text_col, label_col, device, dataset_name):
    logger.info(f"Test sur {dataset_name} ({test_csv_path})")
    
    try:
        # Charger le dataset de test
        test_df = pd.read_csv(test_csv_path)
        logger.info(f"Dataset chargé: {len(test_df)} exemples")
        
        # Afficher la distribution des labels si disponible
        if label_col in test_df.columns:
            logger.info(f"Distribution des labels: {test_df[label_col].value_counts().to_dict()}")
        
        # Créer le dataset
        test_dataset = NewsDataset(
            texts=test_df[text_col].values,
            labels=test_df[label_col].values,
            tokenizer=tokenizer,
            max_len=CONFIG["MAX_LEN"]
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=CONFIG["BATCH_SIZE"]
        )
        
        # Évaluer le modèle
        test_accuracy, test_report, preds, labels = evaluate(model, test_dataloader, device, phase=f"test_{dataset_name}")

        # Confusion matrix simple
        cm = confusion_matrix(labels, preds)
        logger.info("Matrice de confusion :\n" + str(cm))

        cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.txt")
        np.savetxt(cm_path, cm, fmt='%d')
        logger.info(f"Matrice de confusion sauvegardée à {cm_path}")
        
        # Sauvegarder les résultats du test
        test_metrics = {
            "dataset": dataset_name,
            "path": test_csv_path,
            "accuracy": test_accuracy,
            "report": test_report
        }
        save_metrics(test_metrics, output_dir, f"test_metrics_{dataset_name}.json")
        
        return test_accuracy, test_report, cm
    
    except Exception as e:
        logger.error(f"Erreur lors du test sur {dataset_name}: {str(e)}")
        return None, None

# Tests sur différents datasets
logger.info("Début des tests sur les datasets externes")

logger.info("Test sur Fake_Real dataset")
fake_real_path = "data/fake_real.csv"
test_on_new_dataset(model, tokenizer, fake_real_path, "text", "label", DEVICE, "fake_real")

logger.info("Test sur Fake_News dataset")
fake_news_path = "data/train.csv"
test_on_new_dataset(model, tokenizer, fake_news_path, "text", "label", DEVICE, "fake_news")

logger.info("Script terminé avec succès")
logger.info(f"Journal complet disponible dans: {log_file}")