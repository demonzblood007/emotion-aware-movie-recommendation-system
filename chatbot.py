import json
import os
import random
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()
model = GoogleGenerativeAI(model='gemini-2.0-flash')

SYSTEM_PROMPT = (
     """
You are a movie assistant who is talking to users casually to understand:
- Their current mood or emotional state
- The type of movie they’re in the mood for
- Any preferences they have: genre, themes, characters, tone

Your goal is to make the conversation engaging while subtly collecting emotional signals.
Avoid recommending movies just yet.
"""
)

msg_type_map = {
    'system': SystemMessage,
    'human': HumanMessage,
    'ai': AIMessage
}

def to_lc_message(msg):
    return msg_type_map[msg['type']](content=msg['content'])

def from_lc_message(msg):
    if isinstance(msg, SystemMessage):
        return {'type': 'system', 'content': str(msg.content)}
    elif isinstance(msg, HumanMessage):
        return {'type': 'human', 'content': str(msg.content)}
    elif isinstance(msg, AIMessage):
        return {'type': 'ai', 'content': str(msg.content)}
    else:
        raise ValueError('Unknown message type')

def get_random_initial_message():
    """Get a random initial message from the initial_messages.txt file"""
    try:
        with open('initial_messages.txt', 'r', encoding='utf-8') as f:
            messages = [line.strip() for line in f if line.strip()]
        return random.choice(messages)
    except FileNotFoundError:
        # Fallback message if file not found
        return "Hey there! 👋 How's your day going? I'm here to chat about movies and help you discover something amazing to watch!"

def generate_initial_response():
    """Generate an initial response using the LLM with inspiration from the initial_messages.txt file"""
    initial_message = get_random_initial_message()
    
    # Create a prompt that uses the selected message as inspiration
    prompt = f"""You are a friendly movie assistant starting a conversation with a user. 

Here's an example greeting you can use as inspiration:
"{initial_message}"

Based on this example, generate a natural, friendly greeting to start a conversation about movies. 
Make it feel personal and engaging, asking about their mood or day to understand what kind of movie they might be interested in.
Keep it conversational and warm, similar in tone to the example but make it your own unique greeting.

Your greeting:"""
    
    result = model.invoke(prompt)
    return result

def get_user_history(user_id):
    path = f'user_data/user_{user_id}.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    # For new users, only add system message
    if not history:
        history = [{'type': 'system', 'content': SYSTEM_PROMPT}]
    
    return history

def save_user_history(user_id, history):
    os.makedirs('user_data', exist_ok=True)
    path = f'user_data/user_{user_id}.json'
    with open(path, 'w') as f:
        json.dump(history, f)

def run_chat(user_id, user_input):
    history = get_user_history(user_id)
    lc_history = [to_lc_message(m) for m in history]
    lc_history.append(HumanMessage(content=user_input))
    result = model.invoke(lc_history)
    lc_history.append(AIMessage(content=result))
    out_history = [from_lc_message(m) for m in lc_history]
    save_user_history(user_id, out_history)
    return out_history

def reset_user_history(user_id):
    path = f'user_data/user_{user_id}.json'
    if os.path.exists(path):
        os.remove(path)
    
    # Generate new initial message for reset users
    # This will be called when get_user_history is called after reset
    return True

def count_conversations(user_id):
    """Count the number of conversations (human messages) for a user"""
    history = get_user_history(user_id)
    # Count human messages (excluding system messages)
    human_messages = [msg for msg in history if msg['type'] == 'human']
    return len(human_messages)