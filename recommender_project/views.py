import os
import pickle
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load GCN assets
ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'recommender_model_assets.pkl')
recommender_assets = None

def load_assets():
    global recommender_assets
    if recommender_assets is None and os.path.exists(ASSETS_PATH):
        with open(ASSETS_PATH, 'rb') as f:
            recommender_assets = pickle.load(f)
    return recommender_assets

def home(request):
    return render(request, 'home.html')

def recommender_view(request):
    assets = load_assets()
    users_list = []
    if assets:
        users_list = list(assets['user_to_idx'].keys())[:200]
    return render(request, 'recommender.html', {'users_list': users_list})

def analytics_view(request):
    return render(request, 'analytics.html')

def contact_view(request):
    return render(request, 'contact.html')

def user_history(request):
    user_id = request.GET.get('user_id')
    assets = load_assets()
    if not assets or not user_id:
        return JsonResponse({'error': 'Requête invalide ou modèle non chargé'}, status=400)
        
    try:
        user_id = int(user_id)
        user_idx = assets['user_to_idx'][user_id]
        
        # Get user historical highly rated movies (e.g. rated >= 4.0)
        ratings_df = assets['ratings_df']
        movies_df = assets['movies_df']
        
        user_ratings = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4.0)]
        # Join with movie titles
        history_df = user_ratings.merge(movies_df, on='movieId')
        
        history = []
        for _, row in history_df.head(10).iterrows():
            history.append({
                'title': row['title'],
                'genres': row['genres'],
                'rating': float(row['rating'])
            })
            
        return JsonResponse({'history': history})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def recommend(request):
    user_id = request.GET.get('user_id')
    assets = load_assets()
    if not assets or not user_id:
        return JsonResponse({'error': 'Requête invalide ou modèle non chargé'}, status=400)
        
    try:
        user_id = int(user_id)
        user_idx = assets['user_to_idx'][user_id]
        
        user_embeddings = assets['user_embeddings']
        item_embeddings = assets['item_embeddings']
        train_interactions = assets['train_interactions']
        movies_df = assets['movies_df']
        idx_to_item = assets['idx_to_item']
        
        # Calculate LightGCN predicted preference score (dot product)
        u_emb = user_embeddings[user_idx]
        scores = np.dot(item_embeddings, u_emb)
        
        # Mask already interacted items
        seen_items = train_interactions.get(user_idx, [])
        scores[seen_items] = -np.inf
        
        # Top-10 recommendations
        top_k = 10
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
        
        recommendations = []
        # Build node relationships for interactive graph visualization
        nodes = [{'id': f'user_{user_id}', 'label': f'Utilisateur {user_id}', 'color': '#3B82F6', 'shape': 'dot', 'size': 25}]
        edges = []
        
        # Get history to add to graph
        ratings_df = assets['ratings_df']
        history_df = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4.0)].merge(movies_df, on='movieId').head(5)
        for _, row in history_df.iterrows():
            m_id = int(row['movieId'])
            title = row['title']
            nodes.append({'id': f'movie_{m_id}', 'label': title[:20]+'...', 'color': '#10B981', 'shape': 'triangle', 'size': 15})
            edges.append({'from': f'user_{user_id}', 'to': f'movie_{m_id}', 'label': 'A aimé'})
            
        for idx in top_indices:
            movie_id = idx_to_item[idx]
            movie_details = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            score = float(scores[idx])
            
            recommendations.append({
                'movie_id': int(movie_id),
                'title': movie_details['title'],
                'genres': movie_details['genres'],
                'score': score
            })
            
            nodes.append({'id': f'rec_{movie_id}', 'label': movie_details['title'][:20]+'...', 'color': '#EC4899', 'shape': 'star', 'size': 18})
            edges.append({'from': f'user_{user_id}', 'to': f'rec_{movie_id}', 'label': f'Recommandé ({score:.2f})'})
            
        return JsonResponse({
            'recommendations': recommendations,
            'graph': {
                'nodes': nodes,
                'edges': edges
            }
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

from django.core.mail import send_mail
from django.conf import settings

@csrf_exempt
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
        
        if not name or not email or not message:
            return JsonResponse({'error': 'Veuillez remplir tous les champs obligatoires.'}, status=400)
            
        full_message = f"Message de {name} ({email}) :\n\n{message}"
        try:
            send_mail(
                subject=f"[Contact Plateforme] {subject or 'Nouveau Message'}",
                message=full_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=['npe.techs@gmail.com'],
                fail_silently=False,
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)


def movies_catalog(request):
    """Return a searchable list of movies for the new user rating form."""
    assets = load_assets()
    if not assets:
        return JsonResponse({'movies': []})
    
    query = request.GET.get('q', '').strip().lower()
    movies_df = assets['movies_df']
    
    if query:
        mask = movies_df['title'].str.lower().str.contains(query, na=False)
        results = movies_df[mask].head(20)
    else:
        # Return a diverse random sample for discovery
        results = movies_df.sample(min(30, len(movies_df)), random_state=42)
    
    movies = [
        {
            'movieId': int(row['movieId']),
            'title': row['title'],
            'genres': row['genres']
        }
        for _, row in results.iterrows()
    ]
    return JsonResponse({'movies': movies})


@csrf_exempt
def new_user_recommend(request):
    """
    Cold-start recommendation for a brand new user.
    Accepts a JSON body: { "name": str, "ratings": [{"movieId": int, "rating": float}, ...] }
    Strategy: build a new user embedding as a weighted average of rated item embeddings,
    then compute dot-product scores against all items.
    """
    import json
    if request.method != 'POST':
        return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

    assets = load_assets()
    if not assets:
        return JsonResponse({'error': 'Modèle non disponible. Exécutez le script d\'entraînement.'}, status=500)

    try:
        body = json.loads(request.body)
        name = body.get('name', 'Nouvel Utilisateur').strip() or 'Nouvel Utilisateur'
        ratings_input = body.get('ratings', [])  # [{"movieId": int, "rating": float}]

        if not ratings_input:
            return JsonResponse({'error': 'Veuillez noter au moins un film.'}, status=400)

        item_embeddings = assets['item_embeddings']   # shape (n_items, dim)
        item_to_idx = assets['item_to_idx']
        idx_to_item = assets['idx_to_item']
        movies_df = assets['movies_df']

        # Build new user embedding: weighted mean of item embeddings by rating
        vectors = []
        weights = []
        rated_item_indices = []

        for entry in ratings_input:
            movie_id = int(entry['movieId'])
            rating = float(entry.get('rating', 3.0))
            if movie_id in item_to_idx:
                idx = item_to_idx[movie_id]
                vectors.append(item_embeddings[idx] * rating)
                weights.append(rating)
                rated_item_indices.append(idx)

        if not vectors:
            return JsonResponse({'error': 'Aucun film reconnu dans le catalogue. Essayez d\'autres titres.'}, status=400)

        # Weighted average → synthetic user embedding
        user_embedding = np.sum(vectors, axis=0) / (np.sum(weights) + 1e-8)

        # Score all items
        scores = np.dot(item_embeddings, user_embedding)

        # Mask already rated items
        for idx in rated_item_indices:
            scores[idx] = -np.inf

        top_k = 10
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]

        recommendations = []
        nodes = [{'id': 'new_user', 'label': name[:15], 'color': '#7C3AED', 'shape': 'dot', 'size': 28}]
        edges = []

        # Add rated films to graph
        for entry in ratings_input[:5]:
            movie_id = int(entry['movieId'])
            rating = float(entry.get('rating', 3.0))
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                title = movie_row.iloc[0]['title']
                nodes.append({'id': f'rated_{movie_id}', 'label': title[:20] + '...', 'color': '#10B981', 'shape': 'triangle', 'size': 14})
                edges.append({'from': 'new_user', 'to': f'rated_{movie_id}', 'label': f'Note: {rating}'})

        for idx in top_indices:
            movie_id = idx_to_item[idx]
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if movie_row.empty:
                continue
            movie_details = movie_row.iloc[0]
            score = float(scores[idx])

            recommendations.append({
                'movie_id': int(movie_id),
                'title': movie_details['title'],
                'genres': movie_details['genres'],
                'score': score
            })

            nodes.append({'id': f'rec_{movie_id}', 'label': movie_details['title'][:20] + '...', 'color': '#EC4899', 'shape': 'star', 'size': 18})
            edges.append({'from': 'new_user', 'to': f'rec_{movie_id}', 'label': f'Recommandé ({score:.2f})'})

        return JsonResponse({
            'user_name': name,
            'recommendations': recommendations,
            'graph': {'nodes': nodes, 'edges': edges}
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

