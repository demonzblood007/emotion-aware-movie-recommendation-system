import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
from tqdm import tqdm
import os




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
EMOTION_MODEL_PATH = "./bert_emotion_model"
MOVIE_MODEL_PATH = "./best_model.pt"
MOVIE_DATA_PATH = "./filtered_data.csv"

movies_df = pd.read_csv('movies.csv')  # make sure path is correct
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))


# Define the emotion detection model class (BERT-based)
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
       
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
def load_emotion_model(model_path):
    print("Loading emotion detection model...")
    
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    

    
    # Get emotion labels from Go Emotions dataset
    from datasets import load_dataset
    dataset = load_dataset("go_emotions")
    emotion_labels = dataset["train"].features["labels"].feature.names
    num_labels = len(emotion_labels)
    
    # Initialize and load the model
    device = torch.device("cpu")
    model = BertForMultiLabelClassification(num_labels=num_labels)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, emotion_labels, device

# Load the movie recommendation model
def load_movie_model(model_path, num_users, num_movies, num_genres):
    print("Loading movie recommendation model...")
    
    device = torch.device("cpu")
    model = RecommenderModel(num_users, num_movies, num_genres)
    
    # Handle DataParallel wrapping if present in saved model
    state_dict = torch.load(model_path, map_location=device)
    if all(k.startswith('module.') for k in state_dict.keys()):
        # Remove 'module.' prefix from keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

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
    
    # Convert user_id to tensor and expand to match the number of movies
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