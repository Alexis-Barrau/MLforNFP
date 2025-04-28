**Projet de détection automatique de "Fake News"**

Les éléments de ce projet sont organisés de la façon suivante :

1) Un dossier "data" comprenant les données utilisées :
   - Le dataset ISOT, avec les vraies articles dans True.csv et les fausses dans Fake.csv (https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset/)
   - Le dataset Kaggle Fake News Competition, appelé "fake_news" dans les scripts, dans train.csv (https://www.kaggle.com/competitions/fake-news/data?select=train.csv)
   - Le dataset Kaggle Fake or Real, appelé "fake_real" dans les scripts, dans fake_or_real_news.csv (https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news)
2) Un dossier "Modele" dans lequel certains modèles de prédicteurs sont sauvegardés
3) Un dossier "Source" contenant l'article scientifique ayant servi d'inspiration pour ce travail.
4) Un dossier "logs", qui contient les logs d'output de certains scripts
5) Divers scripts et Notebook. D'une manière générale, les Notebooks ont été privilégiés sauf quand le temps de calcul rendait leur utilisation impossible (travail réalisé sur le Datalab Onyxia - SSPCloud). Ils s'organisent de la façon suivante :
   - 0_Installation packages.ipynb permet l'installation de packages utilisé non installés sur le Datalab par défaut
   - 1_TF_IDF_logistic.ipynb contient l'essentiel des résultats présentés dans la partie 3 du rapport, notamment le modèle baseline, la variation avec modificatin des stop_words, le panachage de dataset
   - 2_Titres TF-IDF logistic.ipynb, qui contient les résultats de la partie 3.2.2 du rapport (modèle baseline sur les titres des articles)
   - 3_GridSearch_TF_IDF_MLP.py, script qui permet de choisir les paramètres du MLP appliqué dans 4. Les résultats du script sont dans le log GridSearch_TF_IDF_MLP_output.log
   - 4_TF_IDF_MLP.ipynb qui contient les résultats présentés dans la partie 3.2.3 du rapport
   - 5_Bert.ipynb contient le modèle reposant sur une tokenisation avec (base)-BERT suivi d'une régression logistique, présenté dans la partie 4.1 du rapport
   - 6_GridSearch_base_bert_MLP.py (log d'output : GridSearch_base_Bert_MLP.log) permettant de réaliser le GridSearch pour l'implémentation d'un MLP sur les embeddings calculés avec BERT (il suppose que ceux-ci, obtenus avec 5, ont été enregistrés dans "data"), évoqué dans la partie 4.1 du rapport
   - 7_Bert_mixed_datasets.ipynb qui contient un modèle logistique sur des panachages des datasets, avec embedding BERT, évoqué dans la partie 4.1 du rapport
   - 8_Fine_tuned_bert.py, script qui permet de faire un fine-tuning de BERT pour notre tâche. Output : bert_fine_tuned.log. Les résultats sont présentés dans la partie 4.2 du rapport
   - 9_Fine_tuned_bert_titre.py, script qui effectue la même tâche que le précédent sur les titres. Output : bert_fine_tuned_titres.log. Les résultats sont mobilisés dans la partie 4.2 du rapport
6) Le rapport final : rapport_fake_news_BARRAU.pdf