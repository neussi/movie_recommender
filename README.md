# 🎬 MovieGCNRec - Système de Recommandation Graph-Based (LightGCN)

**MovieGCNRec** est une application Django de recommandation de films basée sur les réseaux de neurones convolutionnels de graphes (**GCN**). En utilisant l'architecture **LightGCN** sur le jeu de données MovieLens, l'application capture les interactions d'ordre supérieur entre utilisateurs et films dans un graphe bipartite, offrant des recommandations personnalisées hautement pertinentes et visualisables sous forme de graphe interactif de propagation de similarités.

---

## 🚀 Fonctionnalités Clés

1. **Sélection Utilisateur Dynamique** : Analyse du profil historique d'un utilisateur sélectionné parmi le catalogue (notes attribuées $\geq 4.0$).
2. **Recommandations LightGCN (Top 10)** : Prédiction des scores de préférence par produit scalaire des embeddings utilisateurs-films affinés par propagation de graphe.
3. **Visualisation Interactive de Graphes (Vis.js)** : Rendu 3D/2D en temps réel du sous-graphe reliant l'utilisateur à ses films préférés historiques (nœuds verts) et aux recommandations proposées (nœuds roses en étoile) avec les poids des liens de recommandation.
4. **Architecture Hybride Résiliente** : En cas d'absence de PyTorch, le système bascule automatiquement sur une décomposition matricielle de type **TruncatedSVD** (Singular Value Decomposition), garantissant 100% de disponibilité avec des performances de recommandation équivalentes en temps réel.
5. **Esthétique Dark Mode** : Interface ergonomique avec verres givrés (glassmorphism) et animations de transition construites en Tailwind CSS.

---

## 🛠️ Stack Technique

- **Framework Web** : Django 6.0+
- **Algorithmes Graph** : PyTorch (LightGCN avec fonction de perte BPR), Scipy Sparse
- **Algorithmes Fallback** : Scikit-Learn (TruncatedSVD Matrix Factorization)
- **Visualisation de Graphe** : Vis.js Network Engine
- **Manipulation Données** : Pandas, NumPy
- **Style frontend** : Tailwind CSS, Outfit Typography

---

## 📦 Installation et Lancement Local

### Prerequisites
- Python 3.10+
- Pip & Virtualenv

### Étapes d'installation

1. **Activer l'environnement virtuel et installer les dépendances** :
   ```bash
   source ../venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Télécharger le jeu de données MovieLens** :
   Récupérez la base de données MovieLens-Small à l'aide du script de téléchargement automatique :
   ```bash
   python download_data.py
   ```

3. **Générer les embeddings (Entraînement)** :
   Exécutez le script d'entraînement pour générer l'artéfact d'embeddings `recommender_model_assets.pkl` :
   ```bash
   python train_recommender.py
   ```

4. **Appliquer les migrations de base de données** :
   ```bash
   python manage.py migrate
   ```

5. **Lancer le serveur de développement** :
   ```bash
   python manage.py runserver 0.0.0.0:8002
   ```

Accédez à l'application sur [http://localhost:8002/](http://localhost:8002/).

---

## 📂 Structure du Projet

```
├── recommender_project/     # Configuration et Vues Django (API REST de calcul de scores)
│   ├── settings.py
│   ├── urls.py
│   └── views.py             # Inférence à la volée (produit scalaire d'embeddings)
├── dataset/                 # Dataset MovieLens (ml-latest-small)
├── models/                  # Embeddings sérialisés (recommender_model_assets.pkl)
├── static/                  # Assets static local
├── templates/
│   └── index.html           # Interface web interactive et Canvas Vis.js
├── download_data.py         # Script utilitaire de téléchargement du dataset
├── train_recommender.py     # Script d'entraînement (LightGCN / TruncatedSVD fallback)
├── movie_recommender.ipynb  # Notebook Jupyter d'analyse descriptive de graphe
└── manage.py
```
