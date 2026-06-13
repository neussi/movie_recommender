# Analyse et Conception — Movie Recommender GCN

**Projet:** Systeme de Recommandation de Films par Filtrage Collaboratif GNN  
**Niveau:** Master 2 — Intelligence Artificielle, ENSPY  
**Annee:** 2025-2026

---

## 1. Problematique et Contexte

Les systemes de recommandation classiques (filtrage collaboratif matriciel, ALS) ne capturent pas explicitement les relations d'ordre superieur dans les graphes d'interaction utilisateur-article. Un utilisateur peut etre similaire a un autre non pas par ses notes directes, mais par les items qu'ils ont tous deux apprecies, qui eux-memes sont lies a d'autres utilisateurs. LightGCN exploite cette structure de graphe de maniere native.

**Objectif:** Apprendre des representations latentes d'utilisateurs et de films a partir du graphe d'interaction pour generer des recommandations Top-K personnalisees, avec support du cold-start pour les nouveaux utilisateurs.

---

## 2. Analyse Exploratoire du Dataset

### 2.1 Statistiques du Graphe (MovieLens ml-latest-small)

```
Entite                      Valeur
---------                   ------
Utilisateurs                610
Films                       9 742
Interactions (ratings)      100 836
Densite du graphe           0.017 (1.7%)
Note moyenne globale        3.50 / 5.00
Distribution des notes:
  5.0 etoiles               13.2%
  4.0 etoiles               26.5%
  3.5 etoiles               13.8%
  3.0 etoiles               20.1%
  Inferieure a 3.0          26.4%
```

### 2.2 Distribution des Interactions par Utilisateur

```
Percentile    Nombre de ratings
----------    -----------------
P10           19
P25           39
P50 (median)  71
P75           141
P90           280
P99           737
```

### 2.3 Genres Dominants

```
Genre           Films      Proportion
-------         -----      ----------
Drama           4361        44.8%
Comedy          3756        38.6%
Thriller        1894        19.5%
Action          1828        18.8%
Romance         1596        16.4%
Adventure       1263        13.0%
Crime           1199        12.3%
```

---

## 3. Architecture de la Solution

### 3.1 Vue Globale du Systeme

```
+----------------------------------------------------------+
|                      CLIENT (Navigateur)                 |
|         /recommender/  (Tab 1: Existant | Tab 2: Nouveau)|
+----------------------------|----------------------------|+
                             | HTTP / AJAX / JSON
+----------------------------v-----------------------------+
|                   Django Application                     |
|                                                          |
|  +------------------+   +---------------------------+    |
|  |  Views Layer     |   |  URL Router               |    |
|  |  views.py        |   |  8 endpoints              |    |
|  +--------+---------+   +---------------------------+    |
|           |                                              |
|  +--------v-----------------------------------------+    |
|  |            Recommender Engine                    |    |
|  |                                                  |    |
|  |  load_assets() -> pickle                         |    |
|  |  {                                               |    |
|  |    user_embeddings:   (610, 64)                  |    |
|  |    item_embeddings:   (9742, 64)                 |    |
|  |    user_to_idx:       dict                       |    |
|  |    item_to_idx:       dict                       |    |
|  |    idx_to_item:       dict                       |    |
|  |    train_interactions: dict[user_idx -> [items]] |    |
|  |    ratings_df:        DataFrame                  |    |
|  |    movies_df:         DataFrame                  |    |
|  |  }                                               |    |
|  |                                                  |    |
|  |  Inference:                                      |    |
|  |    scores = item_embeddings @ u_emb              |    |
|  |    mask seen items -> argsort -> Top-10          |    |
|  +--------------------------------------------------+    |
|                                                          |
|  Cold-Start (Nouveau Profil):                            |
|    e_new = sum(r_k * e_ik) / sum(r_k)                    |
|    scores = item_embeddings @ e_new                      |
|                                                          |
+----------------------------------------------------------+
                         |
                Vercel Serverless Runtime (Python 3.12)
```

### 3.2 Flux de Donnees — Utilisateur Existant

```
GET /recommender/?user_id=42
         |
         v
+--------------------+
|  load_assets()     |   Charge pickle depuis models/
+--------------------+
         |
         v
+--------------------+
|  user_to_idx[42]   |   user_idx = 37 (par exemple)
+--------------------+
         |
         v
+--------------------+
|  u_emb = user_     |   Vecteur de taille 64
|  embeddings[37]    |
+--------------------+
         |
         v
+--------------------+
|  scores =          |   Produit matriciel (9742, 64) x (64,)
|  item_emb @ u_emb  |   = vecteur de scores (9742,)
+--------------------+
         |
         v
+--------------------+
|  Masque items vus  |   scores[train_interactions[37]] = -inf
+--------------------+
         |
         v
+--------------------+
|  argsort -> Top-10 |   np.argpartition + argsort
+--------------------+
         |
         v
  JSON: { recommendations, graph.nodes, graph.edges }
```

---

## 4. Architecture du Modele LightGCN

### 4.1 Principe Mathematique

LightGCN elimine les transformations non lineaires traditionnelles des GCN pour ne conserver que l'agregation de voisinage symmetriquement normalisee.

**Graphe Bipartite:**
```
G = (U union I,  E)
(u, i) in E  ssi  rating(u, i) >= seuil
```

**Matrice d'Adjacence Normalisee:**
```
A_tilde = D^(-1/2) * A * D^(-1/2)

avec A = [ 0    R  ]    R in R^(|U| x |I|)
         [ R^T  0  ]

D = matrice diagonale des degres
```

**Regle de Propagation par Couche:**
```
E^(k+1) = A_tilde * E^(k)

ce qui donne explicitement :
e_u^(k+1) = sum_{i in N(u)} e_i^(k) / sqrt(|N(u)| * |N(i)|)
e_i^(k+1) = sum_{u in N(i)} e_u^(k) / sqrt(|N(i)| * |N(u)|)
```

**Representation Finale (mean pooling):**
```
e_u = (1/(K+1)) * sum_{k=0}^{K} e_u^(k)
e_i = (1/(K+1)) * sum_{k=0}^{K} e_i^(k)

K = 3 couches de propagation
```

**Score de Prediction:**
```
y_hat(u, i) = e_u^T * e_i
```

**Fonction de Perte — BPR (Bayesian Personalized Ranking):**
```
L = - sum_{(u,i,j) in D} log sigma( y_hat(u,i) - y_hat(u,j) )
    + lambda * (||e_u^(0)||^2 + ||e_i^(0)||^2 + ||e_j^(0)||^2)

D : ensemble de triplets (utilisateur, item positif, item negatif)
lambda : coefficient de regularisation L2
```

### 4.2 Parametres du Modele

```
Parametre                    Valeur
---------                    ------
Dimension d'embedding (d)    64
Nombre de couches (K)        3
Taux d'apprentissage         0.001
Optimiseur                   Adam
Regularisation L2 (lambda)   1e-4
Batch size                   1024 triplets
Epochs                       100
Negative sampling             Uniforme (1 negatif par positif)
Split train/test              80% / 20% par timestamp
```

### 4.3 Complexite

```
Parametre       Formule                  Valeur
---------       -------                  -----
Parametres      (|U| + |I|) * d         (610 + 9742) * 64 = 662 976
Memoire         O((|U| + |I|) * d)      ~5.1 MB (float32)
Inference       O(|I| * d)              O(623 488) par requete
```

---

## 5. Strategie Cold-Start

Pour les nouveaux utilisateurs sans historique, une representation synthetique est construite par agregation ponderee des embeddings d'items notes :

```
Donnees entrees: { (i_1, r_1), ..., (i_n, r_n) }

Construction de l'embedding:
  e_new = sum_k( r_k * e_{i_k}^(0) ) / sum_k( r_k )

Ou:
  e_{i_k}^(0) : embedding appris de l'item i_k
  r_k          : note donnee par le nouvel utilisateur (1-5)

Inference:
  scores = E_items @ e_new         (vecteur de dim |I|)
  masque: items deja notes
  Top-10: argsort(scores)[::-1]
```

Cette approche ne necessite aucun re-entrainement du modele.

---

## 6. Evaluation

```
Metrique            Valeur (test set)
--------            -----------------
Precision@10        0.127
Recall@10           0.082
NDCG@10             0.142
AUC (ROC)           0.763
Coverage            72.3% du catalogue
```

---

## 7. Decisions de Conception

| Decision | Justification |
|----------|--------------|
| LightGCN vs MF | Capture les relations d'ordre superieur dans le graphe |
| Mean pooling des couches | Equivalent a des connexions residuelles — stabilite |
| BPR Loss vs BCE | Optimise le rang relatif, plus adapte aux recommandations |
| d=64 | Compromis expressivite / memoire serverless |
| Cold-start embedding | Permet l'utilisation immediate sans re-entrainement |
| Vis.js pour le graphe | Visualisation interactive du graphe bipartite en JS natif |

---

## 8. Diagramme de Deploiement

```
+--------------------+         +--------------------+
|  Developpeur       |  push   |  GitHub            |
|  (local)           | ------> |  neussi/           |
|                    |         |  movie_recommender |
|  train_recommender.|         +--------+-----------+
|  py -> .pkl assets |                  |
+--------------------+                  | Vercel CI/CD
                                        v
+--------------------------------------------+
|              Vercel Platform               |
|  Runtime: Python 3.12 Serverless           |
|  Handler: recommender_project/wsgi.py      |
|  Static:  /staticfiles/ via WhiteNoise     |
+--------------------------------------------+
                    |
       https://movie-recommender-indol-nine.vercel.app
```
