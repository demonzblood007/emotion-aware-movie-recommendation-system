import streamlit as st
import requests
import uvicorn
import pandas as pd

st.set_page_config(page_title='Chatbot', page_icon='🤖')
st.title('emotion aware movie recommender chatbot')

backend_url = 'http://localhost:8000/chat'
reset_url = 'http://localhost:8000/reset'
langgraph_url = 'http://localhost:8000/langgraph-flow'
conversation_count_url = 'http://localhost:8000/conversation-count'
initial_message_url = 'http://localhost:8000/initial-message'
USER_DATA_FILE = 'database_movielens/filtered_data.csv'

# --- LOGIN/SIGNUP LOGIC ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'uid' not in st.session_state:
    st.session_state['uid'] = ''
if 'recommended_genres' not in st.session_state:
    st.session_state['recommended_genres'] = []

if not st.session_state['logged_in']:
    st.title('Login or Signup')
    uid_input = st.text_input('Enter your UID (integer):', key='login_uid')
    login_btn = st.button('Login/Signup')
    if login_btn and uid_input.isdigit():
        uid = int(uid_input)
        # Check if UID exists in filtered_data.csv
        try:
            df = pd.read_csv(USER_DATA_FILE, usecols=['userId'])
            if uid in df['userId'].values:
                st.success(f'Logged in as UID {uid}')
            else:
                # Signup: add UID to filtered_data.csv (with dummy row)
                # We'll add a row with NaNs for other columns
                new_row = {'userId': uid, 'movieId': -1, 'rating': -1, 'genres': ''}
                df_full = pd.read_csv(USER_DATA_FILE)
                df_full = pd.concat([df_full, pd.DataFrame([new_row])], ignore_index=True)
                df_full.to_csv(USER_DATA_FILE, index=False)
                st.success(f'Signed up and logged in as UID {uid}')
            st.session_state['logged_in'] = True
            st.session_state['uid'] = uid
            st.session_state['user_id'] = str(uid)  # for chat logic
            st.session_state['user_id_entered'] = True
            st.rerun()
        except Exception as e:
            st.error(f'Error accessing user data: {e}')
    st.stop()

# User ID input with Enter button
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ''
if 'user_id_entered' not in st.session_state:
    st.session_state['user_id_entered'] = False

col1, col2 = st.columns([3, 1])
with col1:
    user_id = st.text_input('User ID:', st.session_state['user_id'], key='user_id_input')
with col2:
    enter_button = st.button('Enter', type='primary')

# Handle Enter button click
if enter_button and user_id:
    st.session_state['user_id'] = user_id
    st.session_state['user_id_entered'] = True
    # Get initial message immediately
    try:
        response = requests.get(f"{initial_message_url}/{user_id}")
        if response.status_code == 200:
            st.session_state['chat_history'] = response.json()['chat_history']
        else:
            st.error('Error getting initial message.')
    except Exception as e:
        st.error(f'Error: {str(e)}')
    st.rerun()

# Initialize chat history if not present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Reset chat button
if st.button('Reset Chat'):
    if st.session_state['user_id']:
        response = requests.post(reset_url, json={'user_id': st.session_state['user_id']})
        if response.status_code == 200:
            # Update chat history with the new initial message
            st.session_state['chat_history'] = response.json()['chat_history']
        else:
            st.error('Error resetting chat.')
    else:
        st.session_state['chat_history'] = []
    st.rerun()

# Display chat history
for msg in st.session_state['chat_history']:
    if msg['type'] == 'human':
        st.markdown(f"**You:** {msg['content']}")
    elif msg['type'] == 'ai':
        st.markdown(f"**chatbot.:** {msg['content']}")
    # Remove the system message display - system messages are internal only

# Get conversation count and display LangGraph button
if st.session_state['user_id'] and st.session_state['user_id_entered']:
    try:
        count_response = requests.get(f"{conversation_count_url}/{st.session_state['user_id']}")
        if count_response.status_code == 200:
            conversation_count = count_response.json()['conversation_count']
            st.info(f"Conversations: {conversation_count}/3")
            
            # LangGraph Flow Button
            col1, col2 = st.columns([1, 3])
            with col1:
                if conversation_count >= 3:
                    if st.button('🎬 Get Movie Recommendations', type='primary'):
                        with st.spinner('Analyzing your conversation to recommend genres...'):
                            try:
                                response = requests.post(langgraph_url, json={'user_id': st.session_state['user_id']})
                                if response.status_code == 200:
                                    data = response.json()
                                    st.session_state['recommended_genres'] = data.get("genres", [])
                                    if st.session_state['recommended_genres']:
                                        st.success("We've analyzed your conversation! Here are some genres you might like:")
                                    else:
                                        st.info(data.get("message", "Could not determine genres from the conversation."))
                                else:
                                    st.error('Error triggering recommendation flow.')
                            except Exception as e:
                                st.error(f'Error: {str(e)}')
                else:
                    st.button('🎬 Get Movie Recommendations', disabled=True, help=f"Need {3 - conversation_count} more conversation(s) to unlock")
            
            # Display recommended genres if they exist
            if st.session_state.get('recommended_genres'):
                st.markdown("### Recommended Genres For You")
                # Using st.columns to create a grid-like layout
                cols = st.columns(4)
                for i, genre in enumerate(st.session_state['recommended_genres']):
                    with cols[i % 4]:
                        st.container(border=True).markdown(f"**{genre}**")
        else:
            st.error('Error getting conversation count.')
    except Exception as e:
        st.error(f'Error: {str(e)}')

# Chat input - only show if user has entered their ID
if st.session_state['user_id'] and st.session_state['user_id_entered']:
    user_input = st.text_input('You:', key='input')
    if st.button('Send') and user_input:
        payload = {
            'user_id': st.session_state['user_id'],
            'user_input': user_input
        }
        response = requests.post(backend_url, json=payload)
        if response.status_code == 200:
            st.session_state['chat_history'] = response.json()['chat_history']
            st.rerun()
        else:
            st.error('Error communicating with backend.')
else:
    st.info('Please enter a User ID and click Enter to start chatting.') 