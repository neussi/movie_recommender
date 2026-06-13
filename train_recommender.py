import os
import random
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_norm_adj_matrix(num_users, num_items, train_df):
    if not HAS_TORCH:
        return None
    users = train_df['user_idx'].values
    items = train_df['item_idx'].values
    
    row = np.concatenate([users, items + num_users])
    col = np.concatenate([items + num_users, users])
    data = np.ones_like(row, dtype=np.float32)
    
    N = num_users + num_items
    adj = coo_matrix((data, (row, col)), shape=(N, N))
    
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    deg_inv_sqrt = np.power(deg, -0.5)
    
    deg_inv_sqrt_mat = coo_matrix((deg_inv_sqrt, (np.arange(N), np.arange(N))), shape=(N, N))
    norm_adj = deg_inv_sqrt_mat.dot(adj).dot(deg_inv_sqrt_mat)
    
    coo = norm_adj.tocoo()
    indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size([N, N]))

if HAS_TORCH:
    class LightGCN(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
            super(LightGCN, self).__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            
        def forward(self, norm_adj_matrix):
            users_emb = self.user_embedding.weight
            items_emb = self.item_embedding.weight
            all_emb = torch.cat([users_emb, items_emb], dim=0)
            
            embs = [all_emb]
            for layer in range(self.num_layers):
                all_emb = torch.sparse.mm(norm_adj_matrix, all_emb)
                embs.append(all_emb)
                
            embs = torch.stack(embs, dim=1)
            final_embs = torch.mean(embs, dim=1)
            
            final_users_emb, final_items_emb = torch.split(final_embs, [self.num_users, self.num_items])
            return final_users_emb, final_items_emb

def sample_bpr_triplets(num_users, num_items, user_pos_dict, batch_size=1024):
    users = list(user_pos_dict.keys())
    sampled_users = random.choices(users, k=batch_size)
    
    pos_items = []
    neg_items = []
    
    for u in sampled_users:
        pos_item = random.choice(user_pos_dict[u])
        while True:
            neg_item = random.randint(0, num_items - 1)
            if neg_item not in user_pos_dict[u]:
                break
        pos_items.append(pos_item)
        neg_items.append(neg_item)
        
    return torch.LongTensor(sampled_users), torch.LongTensor(pos_items), torch.LongTensor(neg_items)

def main():
    print("Loading MovieLens dataset...")
    ratings_path = "dataset/ml-latest-small/ratings.csv"
    movies_path = "dataset/ml-latest-small/movies.csv"
    
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print("Error: MovieLens dataset files not found. Run download_data.py first.")
        return
        
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    
    num_users = len(user_ids)
    num_items = len(movie_ids)
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    item_to_idx = {iid: idx for idx, iid in enumerate(movie_ids)}
    idx_to_item = {idx: iid for iid, idx in item_to_idx.items()}
    
    ratings['user_idx'] = ratings['userId'].map(user_to_idx)
    ratings['item_idx'] = ratings['movieId'].map(item_to_idx)
    
    # Train / test split
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    user_train_pos = train_ratings.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    if HAS_TORCH:
        # Adjacency matrix
        norm_adj = get_norm_adj_matrix(num_users, num_items, train_ratings)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3).to(device)
        norm_adj = norm_adj.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        reg_lambda = 1e-4
        epochs = 40
        batch_size = 1024
        
        print(f"Training LightGCN for {epochs} epochs on {device}...")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            users, pos_items, neg_items = sample_bpr_triplets(num_users, num_items, user_train_pos, batch_size)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            user_embs, item_embs = model(norm_adj)
            
            u_emb = user_embs[users]
            pos_emb = item_embs[pos_items]
            neg_emb = item_embs[neg_items]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
            
            l2_reg = (model.user_embedding(users).norm(2).pow(2) + 
                      model.item_embedding(pos_items).norm(2).pow(2) + 
                      model.item_embedding(neg_items).norm(2).pow(2)) / batch_size
                      
            loss = bpr_loss + reg_lambda * l2_reg
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
                
        model.eval()
        with torch.no_grad():
            final_user_embs, final_item_embs = model(norm_adj)
            final_user_embs = final_user_embs.cpu().numpy()
            final_item_embs = final_item_embs.cpu().numpy()
    else:
        print("PyTorch not installed. Using Matrix Factorization (TruncatedSVD) as fallback...")
        from scipy.sparse import csr_matrix
        from sklearn.decomposition import TruncatedSVD
        
        row_indices = train_ratings['user_idx'].values
        col_indices = train_ratings['item_idx'].values
        data = train_ratings['rating'].values
        
        interaction_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_users, num_items))
        
        svd = TruncatedSVD(n_components=64, random_state=42)
        final_item_embs = svd.fit_transform(interaction_matrix.T) # shape [num_items, 64]
        final_user_embs = svd.components_.T # shape [num_users, 64]
        print("Matrix Factorization SVD embeddings generated successfully!")
        
    # Save assets
    os.makedirs("models", exist_ok=True)
    assets = {
        "user_embeddings": final_user_embs,
        "item_embeddings": final_item_embs,
        "user_to_idx": user_to_idx,
        "idx_to_user": idx_to_user,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "movies_df": movies,
        "ratings_df": ratings,
        "train_interactions": user_train_pos
    }
    
    with open("models/recommender_model_assets.pkl", "wb") as f:
        pickle.dump(assets, f)
    print("Recommender GCN assets saved successfully to models/recommender_model_assets.pkl!")

if __name__ == "__main__":
    main()
