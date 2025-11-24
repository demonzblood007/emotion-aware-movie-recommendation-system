import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Default Hugging Face checkpoint used for emotion detection.
DEFAULT_EMOTION_MODEL_ID = "SamLowe/roberta-base-go_emotions"
# Define emotion to genre mapping
emotion_to_genres = {
    "joy": {
        "resonant": ["Comedy", "Family", "Animation", "Music"],
        "contrast": ["Horror", "War", "Thriller"]
    },
    "amusement": {
        "resonant": ["Comedy", "Animation", "Family"],
        "contrast": ["Film-Noir", "Crime", "War"]
    },
    "excitement": {
        "resonant": ["Action", "Adventure", "Sci-Fi", "Fantasy"],
        "contrast": ["Documentary", "Drama"]
    },
    "optimism": {
        "resonant": ["Adventure", "Romance", "Family"],
        "contrast": ["Thriller", "War", "Horror"]
    },
    "pride": {
        "resonant": ["Biography", "History", "Drama"],
        "contrast": ["Horror", "Film-Noir"]
    },
    "gratitude": {
        "resonant": ["Romance", "Family", "Comedy"],
        "contrast": ["Horror", "War"]
    },
    "love": {
        "resonant": ["Romance", "Drama", "Music"],
        "contrast": ["Action", "Horror"]
    },
    "caring": {
        "resonant": ["Family", "Romance", "Drama"],
        "contrast": ["Thriller", "War"]
    },
    "admiration": {
        "resonant": ["Biography", "History", "Documentary"],
        "contrast": ["Horror", "Film-Noir"]
    },
    "relief": {
        "resonant": ["Comedy", "Romance", "Family"],
        "contrast": ["Thriller", "Horror"]
    },
    "satisfaction": {
        "resonant": ["Drama", "Romance", "Comedy"],
        "contrast": ["Horror", "Crime"]
    },
    "approval": {
        "resonant": ["Comedy", "Family", "Animation"],
        "contrast": ["War", "Thriller"]
    },
    "surprise": {
        "resonant": ["Mystery", "Sci-Fi", "Fantasy"],
        "contrast": ["Documentary", "Drama"]
    },
    "curiosity": {
        "resonant": ["Sci-Fi", "Mystery", "Adventure"],
        "contrast": ["War", "Action"]
    },
    "realization": {
        "resonant": ["Drama", "Documentary", "Mystery"],
        "contrast": ["Action", "Comedy"]
    },
    "confusion": {
        "resonant": ["Mystery", "Drama", "Sci-Fi"],
        "contrast": ["Comedy", "Family"]
    },
    "neutral": {
        "resonant": ["Documentary", "Drama"],
        "contrast": ["Horror", "Action"]
    },
    "sadness": {
        "resonant": ["Drama", "Romance", "Music"],
        "contrast": ["Comedy", "Action"]
    },
    "grief": {
        "resonant": ["History", "Drama", "Biography"],
        "contrast": ["Comedy", "Animation"]
    },
    "loneliness": {
        "resonant": ["Drama", "Romance", "Music"],
        "contrast": ["Family", "Adventure"]
    },
    "remorse": {
        "resonant": ["Drama", "Crime", "History"],
        "contrast": ["Action", "Comedy"]
    },
    "anger": {
        "resonant": ["Action", "Crime", "Thriller"],
        "contrast": ["Family", "Romance"]
    },
    "annoyance": {
        "resonant": ["Comedy", "Crime", "Drama"],
        "contrast": ["Music", "Animation"]
    },
    "disapproval": {
        "resonant": ["Crime", "Drama", "War"],
        "contrast": ["Romance", "Comedy"]
    },
    "fear": {
        "resonant": ["Horror", "Thriller", "Mystery"],
        "contrast": ["Romance", "Comedy"]
    },
    "nervousness": {
        "resonant": ["Thriller", "Drama", "Sci-Fi"],
        "contrast": ["Comedy", "Family"]
    },
    "embarrassment": {
        "resonant": ["Comedy", "Romance", "Drama"],
        "contrast": ["War", "Horror"]
    },
    "disgust": {
        "resonant": ["Film-Noir", "Drama", "Crime"],
        "contrast": ["Comedy", "Romance"]
    },
    "disappointment": {
        "resonant": ["Biography", "Romance", "Drama"],
        "contrast": ["Animation", "Comedy"]
    }
}


# Define paths to saved models and data
EMOTION_MODEL_PATH = DEFAULT_EMOTION_MODEL_ID
MOVIE_MODEL_PATH = "./best_model.pt"
MOVIE_DATA_PATH = "./database_movielens/filtered_data.csv"
TRAINED_NUM_USERS = None
TRAINED_NUM_MOVIES = None

movies_df = pd.read_csv('./database_movielens/movies.csv')  # make sure path is correct
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))


# Define the movie recommender model
class RecommenderModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, emb_dim=32, dropout_rate=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.movie_embedding = nn.Embedding(num_movies, emb_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, mean=0, std=0.01)
        self.fc1 = nn.Linear(2 * emb_dim + num_genres, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output = nn.Linear(32, 1)

    def forward(self, user_ids, movie_ids, genres):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb, genres], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.output(x).squeeze()

# Load and preprocess the movies dataset
def load_movies_data():
    print(f"Loading movie data from: {MOVIE_DATA_PATH}")
    
    dtypes = {
        'userId': 'int32',
        'movieId': 'int32',
        'rating': 'float32'
    }
    
    data = pd.read_csv(MOVIE_DATA_PATH, usecols=['userId', 'movieId', 'rating', 'genres'], dtype=dtypes)
    data = data.dropna(subset=['userId', 'movieId', 'rating'])
    
    # Create genre mapping
    genre_set = set()
    for genres_str in data['genres'].dropna().unique():
        if isinstance(genres_str, str):
            genre_set.update(genres_str.split('|'))
    genre_list = sorted(list(genre_set))
    genre_map = {genre: idx for idx, genre in enumerate(genre_list)}
    
    # Create user and movie mappings
    unique_users = data['userId'].unique()
    unique_movies = data['movieId'].unique()
    user_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    reverse_movie_map = {idx: movie_id for movie_id, idx in movie_map.items()}
    
    # Map to internal IDs
    data['user_idx'] = data['userId'].map(user_map)
    data['movie_idx'] = data['movieId'].map(movie_map)
    
    print(f"Loaded {len(data)} ratings, {len(unique_users)} users, {len(unique_movies)} movies")
    return data, genre_map, user_map, movie_map, reverse_movie_map

# Function to process genres for a movie
def process_genres(genres_str, genre_map):
    num_genres = len(genre_map)
    genre_vector = torch.zeros(num_genres, dtype=torch.float32)
    if isinstance(genres_str, str):
        for genre in genres_str.split('|'):
            idx = genre_map.get(genre)
            if idx is not None:
                genre_vector[idx] = 1.0
    return genre_vector

# Load the emotion detection model
def load_emotion_model(model_path: str | None):
    """
    Load a multi-label emotion classifier.

    Args:
        model_path: Hugging Face model id or local directory.

    Returns:
        Tuple[model, tokenizer, label_names, device]
    """
    model_source = model_path or DEFAULT_EMOTION_MODEL_ID
    print(f"Loading emotion detection model from: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, problem_type="multi_label_classification"
    )

    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and id2label:
        try:
            sorted_keys = sorted(
                id2label.keys(),
                key=lambda k: int(k) if isinstance(k, str) and k.isdigit() else k,
            )
        except TypeError:
            sorted_keys = list(id2label.keys())
        emotion_labels = [id2label[key] for key in sorted_keys]
    else:
        emotion_labels = [f"label_{idx}" for idx in range(model.config.num_labels)]

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model, tokenizer, emotion_labels, device

# Load the movie recommendation model
def load_movie_model(model_path, num_users, num_movies, num_genres):
    print("Loading movie recommendation model...")
    
    device = torch.device("cpu")
    saved_state = torch.load(model_path, map_location=device)

    state_dict = saved_state
    if isinstance(state_dict, dict) and not any(
        key.endswith("user_embedding.weight") for key in state_dict.keys()
    ):
        for candidate in ("state_dict", "model_state_dict"):
            if candidate in state_dict and isinstance(state_dict[candidate], dict):
                state_dict = state_dict[candidate]
                break

    user_weight_key = next(
        (k for k in state_dict.keys() if k.endswith("user_embedding.weight")), None
    )
    movie_weight_key = next(
        (k for k in state_dict.keys() if k.endswith("movie_embedding.weight")), None
    )
    if user_weight_key is None or movie_weight_key is None:
        raise KeyError("Embedding weights not found in checkpoint")

    user_emb_weight = state_dict[user_weight_key]
    movie_emb_weight = state_dict[movie_weight_key]
    saved_num_users = user_emb_weight.shape[0]
    saved_num_movies = movie_emb_weight.shape[0]

    model = RecommenderModel(saved_num_users, saved_num_movies, num_genres)
    global TRAINED_NUM_USERS, TRAINED_NUM_MOVIES
    TRAINED_NUM_USERS = saved_num_users
    TRAINED_NUM_MOVIES = saved_num_movies
    
    # Handle DataParallel wrapping if present in saved model
    filtered_state_dict = state_dict
    if all(k.startswith('module.') for k in filtered_state_dict.keys()):
        # Remove 'module.' prefix from keys
        filtered_state_dict = {k[7:]: v for k, v in filtered_state_dict.items()}
    
    model.load_state_dict(filtered_state_dict)
    model.to(device)
    model.eval()
    
    return model, saved_num_users, saved_num_movies

# Predict emotions from text
def predict_emotions(text, model, tokenizer, emotion_labels, device, top_k=3):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Get top emotions
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_emotions = [(emotion_labels[idx], probs[idx]) for idx in top_indices if probs[idx] > 0.2]
    
    return top_emotions

# Map emotions to genres
def map_emotions_to_genres(emotions, emotion_to_genres, user_choice="neutral"):
    genre_scores = {}
    
    print(f"DEBUG: Emotions received: {emotions}")
    print(f"DEBUG: User choice: {user_choice}")
    
    for emotion, score in emotions:
        print(f"DEBUG: Processing emotion: {emotion} with score: {score}")
        
        if emotion in emotion_to_genres:
            print(f"DEBUG: Found matching emotion in mapping")
            resonant_genres = emotion_to_genres[emotion]['resonant']
            contrast_genres = emotion_to_genres[emotion]['contrast']
            
            print(f"DEBUG: Resonant genres: {resonant_genres}")
            print(f"DEBUG: Contrast genres: {contrast_genres}")

            if user_choice == "match": 
                resonant_weight = 0.7
                contrast_weight = 0.3
            elif user_choice == "shift":
                resonant_weight = 0.3
                contrast_weight = 0.7
            elif user_choice == "neutral":
                resonant_weight = 0.5
                contrast_weight = 0.5
            else:
                resonant_weight = 0.5
                contrast_weight = 0.5
                
            print(f"DEBUG: Weights - resonant: {resonant_weight}, contrast: {contrast_weight}")
            
            # Process resonant genres
            for genre in resonant_genres:
                if genre in genre_scores:
                    genre_scores[genre] += score * resonant_weight
                else:
                    genre_scores[genre] = score * resonant_weight
            
            # Process contrast genres
            for genre in contrast_genres:
                if genre in genre_scores:         
                    genre_scores[genre] += score * contrast_weight
                else:
                    genre_scores[genre] = score * contrast_weight
        else:
            print(f"DEBUG: Emotion '{emotion}' not found in emotion_to_genres mapping")
    
    # Normalize genre scores
    total_score = sum(genre_scores.values())
    print(f"DEBUG: Total score before normalization: {total_score}")
    print(f"DEBUG: Genre scores before normalization: {genre_scores}")
    
    if total_score > 0:
        genre_scores = {genre: score/total_score for genre, score in genre_scores.items()}
        print(f"DEBUG: Genre scores after normalization: {genre_scores}")
    else:
        print("DEBUG: Total score is 0, no normalization applied")
    
    return genre_scores

# Create a genre vector from genre scores
def create_genre_vector(genre_scores, genre_map, top_n=4):
    num_genres = len(genre_map)
    genre_vector = torch.zeros(num_genres, dtype=torch.float32)
    top_genres=sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"DEBUG: Selected top genres for one-hot encoding: {top_genres}")
    
    for genre, _ in top_genres:
        idx = genre_map.get(genre)
        if idx is not None:
            genre_vector[idx] = 1.0
    
    return genre_vector

# Recommend movies for a specific user based on emotion-derived genres
def recommend_movies(user_id, genre_vector, movie_model, data, genre_map, top_n=10, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get all movies to evaluate
    all_movies = data[['movie_idx', 'movieId', 'genres']].drop_duplicates('movie_idx')
    if TRAINED_NUM_MOVIES is not None:
        all_movies = all_movies[all_movies['movie_idx'] < TRAINED_NUM_MOVIES]
    
    # Convert user_id to tensor and expand to match the number of movies
    if TRAINED_NUM_USERS is not None and user_id >= TRAINED_NUM_USERS:
        user_id = 0
    user_tensor = torch.tensor([user_id] * len(all_movies), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(all_movies['movie_idx'].values, dtype=torch.long).to(device)
    
    # Process genre vectors for all movies
    genre_tensors = []
    for idx, row in all_movies.iterrows():
        genre_tensors.append(process_genres(row['genres'], genre_map))
    genre_tensors = torch.stack(genre_tensors).to(device)
    
    # Evaluate in batches to avoid memory issues
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(all_movies), batch_size):
            batch_users = user_tensor[i:i+batch_size]
            batch_movies = movie_tensor[i:i+batch_size]
            batch_genres = genre_tensors[i:i+batch_size]
            
            # Make predictions
            batch_preds = movie_model(batch_users, batch_movies, batch_genres)
            predictions.extend(batch_preds.cpu().numpy())
    
    # Add predictions to the movie dataframe
    all_movies['predicted_rating'] = predictions
    
    # Sort by predicted rating
    recommended = all_movies.sort_values('predicted_rating', ascending=False).head(top_n)
    
    return recommended[['movieId', 'genres', 'predicted_rating']]

# Main emotion-based recommendation function
def recommend_movies_by_mood(user_id, mood_text,user_choice="neutral", top_n=10):
    print("\n==== Mood-Based Movie Recommender ====\n")
    
    # Step 1: Load movie data and create mappings
    data, genre_map, user_map, movie_map, reverse_movie_map = load_movies_data()
    num_users = len(user_map)
    num_movies = len(movie_map)
    num_genres = len(genre_map)
    
    # Step 2: Load both models
    emotion_model, tokenizer, emotion_labels, device = load_emotion_model(EMOTION_MODEL_PATH)
    movie_model = load_movie_model(MOVIE_MODEL_PATH, num_users, num_movies, num_genres)
    
    # Step 3: Detect emotions from text
    print(f"Analyzing mood from text: '{mood_text}'")
    emotions = predict_emotions(mood_text, emotion_model, tokenizer, emotion_labels, device)
    print("\nDetected emotions:")
    for emotion, score in emotions:
        print(f"  - {emotion}: {score:.3f}")
    
    # Step 4: Map emotions to genres
    genre_scores = map_emotions_to_genres(emotions, emotion_to_genres, user_choice)
    print("\nRelevant movie genres for your mood:")
    for genre, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {genre}: {score:.3f}")
    
    # Step 5: Create genre vector based on emotions
    genre_vector = create_genre_vector(genre_scores, genre_map)
    
    # Step 6: Convert external user_id to internal index
    if user_id in user_map:
        user_idx = user_map[user_id]
    else:
        # For new users, use a default/random user or add a new entry
        print(f"\nUser ID {user_id} not found in training data.")
        # Choose the first user as default or add new entry to user_map
        user_idx = 0  # Using first user as default
    
    # Step 7: Get movie recommendations
    print(f"\nGenerating recommendations for user {user_id} based on mood...")
    recommendations = recommend_movies(user_idx, genre_vector, movie_model, data, genre_map, top_n, device)
    
    print("\n✨ Top Movie Recommendations Based on Your Mood ✨\n")
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        #print(f"{i}. {row['title']}")
        title = movie_id_to_title.get(int(row['movieId']), "Unknown Title")
        print(f"{i}. {title}")

        print(f"   Genres: {row['genres']}")
        print(f"   Predicted Rating: {row['predicted_rating']:.2f}/5.0\n")
    
    return recommendations

# Example usage
if __name__ == "__main__":
    # Example mood text and user ID
    user_id = 4 # Replace with actual user ID
    mood_text = input("ENTER A SENTENCE-")
    user_choice = input("Would you like to match your mood, shift it, or keep it neutral? (match/shift/neutral): ").strip().lower()
    
    # Get recommendations based on mood
    recommendations = recommend_movies_by_mood(user_id, mood_text, user_choice, top_n=10)