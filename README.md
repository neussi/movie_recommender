# Movie Recommender — GCN-Based Collaborative Filtering

A production-ready recommendation system powered by Light Graph Convolutional Networks (LightGCN), capable of serving personalized movie recommendations for existing users and cold-start recommendations for brand-new profiles.

**Production URL:** https://movie-recommender-indol-nine.vercel.app  
**Repository:** https://github.com/neussi/movie_recommender

---

## Platform Overview

The platform implements collaborative filtering over a bipartite user-item interaction graph using the LightGCN propagation scheme. Users receive Top-10 personalized recommendations computed via learned user and item embedding dot products. New users without historical data are supported via a cold-start inference strategy based on rated item embedding aggregation.

| Route | Section | Description |
|-------|---------|-------------|
| `/` | Home | Platform overview, graph statistics, model summary |
| `/recommender/` | Simulateur GCN | Existing user recommendations + new user cold-start |
| `/analytics/` | Analytique | Embedding distributions, loss curves, precision metrics |
| `/contact/` | Contact | SMTP contact form (Gmail backend) |
| `/recommend/` | API | JSON endpoint for user recommendations |
| `/user-history/` | API | JSON endpoint for user viewing history |
| `/movies-catalog/` | API | Searchable movie catalog for new user form |
| `/new-user-recommend/` | API | Cold-start recommendation endpoint |

---

## Project Structure

```
movie_recommender/
|
+-- recommender_project/        Django project configuration
|   +-- settings.py             Application settings (WhiteNoise, SMTP, CORS)
|   +-- urls.py                 URL routing (8 endpoints)
|   +-- views.py                View functions, GCN inference, cold-start logic
|   +-- wsgi.py                 WSGI entry point (Vercel serverless)
|   +-- asgi.py                 ASGI entry point
|
+-- templates/
|   +-- base.html               Master layout (Tailwind CSS, Outfit font, MathJax)
|   +-- home.html               Landing page with graph stats
|   +-- recommender.html        Dual-tab recommender (existing + new user)
|   +-- analytics.html          Model evaluation and embedding analysis
|   +-- contact.html            Contact form with AJAX submission
|
+-- static/
|   +-- images/                 Pre-generated training and evaluation plots
|
+-- staticfiles/                Collected static assets (WhiteNoise, Vercel)
|
+-- dataset/
|   +-- movies.csv              MovieLens movie catalog (title, genres)
|   +-- ratings.csv             User-movie interaction ratings (userId, movieId, rating)
|
+-- models/
|   +-- recommender_model_assets.pkl    Serialized LightGCN embeddings and metadata
|
+-- docs/
|   +-- images/                 High-resolution plots for documentation
|
+-- train_recommender.py        Offline LightGCN training pipeline
+-- movie_recommender.ipynb     Full Jupyter analysis and training notebook
+-- requirements.txt
+-- vercel.json
+-- manage.py
+-- .gitignore
```

---

## AI Model Architecture

### LightGCN — Light Graph Convolutional Network

LightGCN simplifies graph collaborative filtering by removing feature transformation and nonlinear activation from the propagation step, retaining only neighborhood aggregation.

```
Interaction Graph G = (U union I, E)
  U : set of users
  I : set of items (movies)
  E : edges where (u, i) in E iff user u rated item i

Layer-0 Embeddings (trainable):
  e_u^(0)  in  R^d        for each user  u
  e_i^(0)  in  R^d        for each item  i
  d = 64 (embedding dimension)

Propagation Rule (layer k -> k+1):
  e_u^(k+1) = sum over i in N(u) of  e_i^(k) / sqrt(|N(u)| * |N(i)|)
  e_i^(k+1) = sum over u in N(i) of  e_u^(k) / sqrt(|N(i)| * |N(u)|)

Final Embedding (mean pooling across K=3 layers):
  e_u = (1 / K+1) * sum_{k=0}^{K} e_u^(k)
  e_i = (1 / K+1) * sum_{k=0}^{K} e_i^(k)

Prediction Score:
  y_hat(u, i) = e_u^T * e_i        (dot product)

Objective — BPR (Bayesian Personalized Ranking) Loss:
  L_BPR = - sum_{(u,i,j) in D} log( sigma( y_hat(u,i) - y_hat(u,j) ) ) + lambda * ||Theta||^2
  where i is a positive item and j is a negative (unobserved) item
```

### Architecture Summary

```
Input: User-Item Interaction Matrix R (n_users x n_items)
          |
          v
+----------------------------+
|  Laplacian Normalization   |   A_hat = D^(-1/2) * A * D^(-1/2)
+----------------------------+
          |
          v
+----------------------------+
|  Layer 0: Initial Embeds   |   E^(0) in R^(n x d),  d=64
+----------------------------+
          |
    +-----------+
    | Layer 1   |   E^(1) = A_hat * E^(0)
    +-----------+
    | Layer 2   |   E^(2) = A_hat * E^(1)
    +-----------+
    | Layer 3   |   E^(3) = A_hat * E^(2)
    +-----------+
          |
          v
+----------------------------+
|  Mean Pooling (K=3 layers) |   E_final = mean(E^0, E^1, E^2, E^3)
+----------------------------+
          |
          v
+----------------------------+
|  Dot-Product Scoring       |   score(u, i) = e_u . e_i
+----------------------------+
          |
          v
    Top-K Recommendations
```

### Cold-Start Strategy (New Users)

For users with no interaction history, a synthetic user embedding is constructed from rated items:

```
Given: { (i_1, r_1), (i_2, r_2), ..., (i_n, r_n) }  (movie, rating) pairs

e_new_user = sum_k( r_k * e_{i_k} ) / sum_k( r_k )

Inference: score(new_user, j) = e_new_user^T * e_j   for all j not in rated
```

---

## Dataset

**Source:** MovieLens (ml-latest-small)  
**Users:** 610 unique users  
**Movies:** 9,742 unique movies  
**Ratings:** 100,836 interactions  
**Rating scale:** 0.5 to 5.0 stars  
**Train / Test split:** 80% / 20% by timestamp

---

## Local Development

```bash
git clone https://github.com/neussi/movie_recommender.git
cd movie_recommender

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train_recommender.py
python manage.py runserver
```

---

## Deployment

Deployed on **Vercel** via `@vercel/python`. WhiteNoise serves all static files through the WSGI handler.

| Variable | Description |
|----------|-------------|
| `EMAIL_HOST_PASSWORD` | Gmail App Password for SMTP |

---

## Dependencies

| Package | Role |
|---------|------|
| `django>=5.0` | Web framework |
| `numpy`, `pandas` | Matrix operations and data handling |
| `scipy` | Sparse matrix computation for graph Laplacian |
| `scikit-learn` | Cosine similarity utilities |
| `whitenoise` | Static file serving |
| `django-cors-headers` | Cross-origin request handling |

---

## Contact

**Institution:** Ecole Nationale Superieure Polytechnique de Yaounde (ENSPY)  
**Level:** AIA4 - Intelligence Artificielle  
**Contact:** npe.techs@gmail.com | +237 650 970 526
