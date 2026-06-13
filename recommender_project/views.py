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

def index(request):
    assets = load_assets()
    users_list = []
    if assets:
        users_list = list(assets['user_to_idx'].keys())[:200] # get first 200 users
    return render(request, 'index.html', {'users_list': users_list})

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

