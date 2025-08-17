import torch
from transformers import BertTokenizer, BertForSequenceClassification
from chatbot import get_user_history
import os

# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the model directory
model_path = os.path.join(script_dir, "bert_emotion_model")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=28)

# Updated emotion label order based on provided mapping
id2emotion = [
    "joy", "amusement", "excitement", "optimism", "pride", "gratitude", "love", "caring", "admiration", "relief", "satisfaction", "approval", "surprise", "curiosity", "realization", "confusion", "neutral", "sadness", "grief", "loneliness", "remorse", "anger", "annoyance", "disapproval", "fear", "nervousness", "embarrassment", "disgust", "disappointment"
]

# Use the provided mapping for emotion to genres
emotion_to_genres = {
    "joy": {"resonant": ["Comedy", "Family", "Animation", "Music"], "contrast": ["Horror", "War", "Thriller"]},
    "amusement": {"resonant": ["Comedy", "Animation", "Family"], "contrast": ["Film-Noir", "Crime", "War"]},
    "excitement": {"resonant": ["Action", "Adventure", "Sci-Fi", "Fantasy"], "contrast": ["Documentary", "Drama"]},
    "optimism": {"resonant": ["Adventure", "Romance", "Family"], "contrast": ["Thriller", "War", "Horror"]},
    "pride": {"resonant": ["Biography", "History", "Drama"], "contrast": ["Horror", "Film-Noir"]},
    "gratitude": {"resonant": ["Romance", "Family", "Comedy"], "contrast": ["Horror", "War"]},
    "love": {"resonant": ["Romance", "Drama", "Music"], "contrast": ["Action", "Horror"]},
    "caring": {"resonant": ["Family", "Romance", "Drama"], "contrast": ["Thriller", "War"]},
    "admiration": {"resonant": ["Biography", "History", "Documentary"], "contrast": ["Horror", "Film-Noir"]},
    "relief": {"resonant": ["Comedy", "Romance", "Family"], "contrast": ["Thriller", "Horror"]},
    "satisfaction": {"resonant": ["Drama", "Romance", "Comedy"], "contrast": ["Horror", "Crime"]},
    "approval": {"resonant": ["Comedy", "Family", "Animation"], "contrast": ["War", "Thriller"]},
    "surprise": {"resonant": ["Mystery", "Sci-Fi", "Fantasy"], "contrast": ["Documentary", "Drama"]},
    "curiosity": {"resonant": ["Sci-Fi", "Mystery", "Adventure"], "contrast": ["War", "Action"]},
    "realization": {"resonant": ["Drama", "Documentary", "Mystery"], "contrast": ["Action", "Comedy"]},
    "confusion": {"resonant": ["Mystery", "Drama", "Sci-Fi"], "contrast": ["Comedy", "Family"]},
    "neutral": {"resonant": ["Documentary", "Drama"], "contrast": ["Horror", "Action"]},
    "sadness": {"resonant": ["Drama", "Romance", "Music"], "contrast": ["Comedy", "Action"]},
    "grief": {"resonant": ["History", "Drama", "Biography"], "contrast": ["Comedy", "Animation"]},
    "loneliness": {"resonant": ["Drama", "Romance", "Music"], "contrast": ["Family", "Adventure"]},
    "remorse": {"resonant": ["Drama", "Crime", "History"], "contrast": ["Action", "Comedy"]},
    "anger": {"resonant": ["Action", "Crime", "Thriller"], "contrast": ["Family", "Romance"]},
    "annoyance": {"resonant": ["Comedy", "Crime", "Drama"], "contrast": ["Music", "Animation"]},
    "disapproval": {"resonant": ["Crime", "Drama", "War"], "contrast": ["Romance", "Comedy"]},
    "fear": {"resonant": ["Horror", "Thriller", "Mystery"], "contrast": ["Romance", "Comedy"]},
    "nervousness": {"resonant": ["Thriller", "Drama", "Sci-Fi"], "contrast": ["Comedy", "Family"]},
    "embarrassment": {"resonant": ["Comedy", "Romance", "Drama"], "contrast": ["War", "Horror"]},
    "disgust": {"resonant": ["Film-Noir", "Drama", "Crime"], "contrast": ["Comedy", "Romance"]},
    "disappointment": {"resonant": ["Biography", "Romance", "Drama"], "contrast": ["Animation", "Comedy"]}
}

def get_emotion_genres(user_id: str):
    """
    Analyzes the user's chat history to determine emotions and returns a list of recommended genres.
    """
    # 1. Get user chat history (only human messages)
    history = get_user_history(user_id)
    human_messages = [msg['content'] for msg in history if msg['type'] == 'human']
    
    print(f"[DEBUG] Human messages for user {user_id}: {human_messages}")
    if not human_messages:
        print("[DEBUG] No human messages found.")
        return []

    # 2. Predict emotions for each message
    inputs = tokenizer(human_messages, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predictions = torch.argmax(logits, dim=1)
    
    # 3. Get dominant emotions
    # id2label = model.config.id2label
    # dominant_emotions = [id2label[p.item()] for p in predictions]
    dominant_emotions = [id2emotion[p.item()] for p in predictions]
    print(f"[DEBUG] Predicted emotions: {dominant_emotions}")
    
    # 4. Map emotions to genres (use only 'resonant' genres)
    recommended_genres = set()
    for emotion in dominant_emotions:
        genres = emotion_to_genres.get(emotion, {}).get("resonant")
        print(f"[DEBUG] Emotion '{emotion}' maps to resonant genres: {genres}")
        if genres:
            recommended_genres.update(genres)
    print(f"[DEBUG] Final recommended genres: {list(recommended_genres)}")
    return list(recommended_genres)

def run_reco_flow(user_id: str):
    """
    Main function to run the recommendation flow.
    """
    genres = get_emotion_genres(user_id)
    return {"recommended_genres": genres}

if __name__ == '__main__':
    # Example usage:
    # Make sure you have some user history first by running the chatbot.
    # For testing, you might need to manually create a user file like user_data/test_user.json
    test_user_id = "1" 
    recommendations = run_reco_flow(test_user_id)
    print(f"Recommended genres for user {test_user_id}: {recommendations}")
