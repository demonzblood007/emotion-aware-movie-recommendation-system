import pandas as pd
from collections import Counter
from user_model import Session, UserProfile
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM and Tools
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash',temperature=0.7)
tavily_tool = TavilySearch(max_results=1)
tools = [tavily_tool]

# Create Agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that finds information about movies."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# Movie data cache
movie_cache = {}

def get_movie_details(movie_title):
    if movie_title in movie_cache:
        return movie_cache[movie_title]
    
    try:
        query = f"Provide a brief summary, the director, and main actors for the movie: {movie_title}"
        response = agent_executor.invoke({"input": query})
        movie_cache[movie_title] = response['output']
        return response['output']
    except Exception as e:
        return f"Could not retrieve details for {movie_title}. Error: {e}"


# Load your data
# Adjust the paths if needed
links_path = 'database_movielens/links.csv'
filtered_path = 'database_movielens/filtered_data.csv'
movies_path = 'database_movielens/movies.csv'

df = pd.read_csv(filtered_path)
movies = pd.read_csv(movies_path)
links = pd.read_csv(links_path)
session = Session()

# Merge movies with links to get tmdbId/imdbId
movies = movies.merge(links, on='movieId')


for user_id in df['userId'].unique():
    if user_id > 5: # Limiting to 5 users for faster testing
        break
    
    user_data = df[df['userId'] == user_id]
    user_movies = user_data.merge(movies, left_on='movieId', right_on='movieId')

    # Get top 3 favorite and disliked movies
    fav_movies_df = user_movies[user_movies['rating'] >= 4].nlargest(3, 'rating')
    disliked_movies_df = user_movies[user_movies['rating'] <= 2].nsmallest(3, 'rating')

    fav_movie_titles = fav_movies_df['title'].tolist()
    disliked_movie_titles = disliked_movies_df['title'].tolist()

    # Enrich movie data
    enriched_fav_movies = {title: get_movie_details(title) for title in fav_movie_titles}
    enriched_disliked_movies = {title: get_movie_details(title) for title in disliked_movie_titles}

    # Generate profile summary with LLM
    summary_prompt = f"""
    Based on the following user ratings, please generate a short, narrative summary of this user's cinematic tastes.
    Analyze their likely preferences for genres, directors, actors, and themes.

    Movies the user LOVED:
    {enriched_fav_movies}

    Movies the user DISLIKED:
    {enriched_disliked_movies}

    Please provide a concise summary of their profile.
    """

    try:
        summary_response = llm.invoke(summary_prompt)
        profile_summary = summary_response.content
    except Exception as e:
        profile_summary = f"Could not generate profile summary for user {user_id}. Error: {e}"


    # --- The rest of the original script logic for database insertion ---
    avg_rating = float(user_data['rating'].mean()) if not pd.isna(user_data['rating'].mean()) else None
    num_movies_watched = int(user_data['movieId'].nunique())
    genres = user_movies['genres_y'].str.split('|').explode()
    favorite_genres = [g for g, _ in Counter(genres).most_common(3)]
    disliked_genres = [g for g, _ in Counter(genres).most_common()[-3:]] if len(genres) >= 3 else []
    fav_movies = fav_movie_titles # Use the top 3
    disliked_movies = disliked_movie_titles # Use the top 3
    rating_dist = user_data['rating'].value_counts().sort_index().to_dict()

    # Upsert logic: Check if user exists, then update or create
    profile = session.query(UserProfile).filter_by(user_id=int(user_id)).one_or_none()

    if profile:
        # Update existing profile
        profile.favorite_genres=favorite_genres
        profile.favourite_movies=fav_movies
        profile.disliked_movies=disliked_movies
        profile.disliked_genres=disliked_genres
        profile.avg_rating=avg_rating
        profile.num_movies_watched=num_movies_watched
        profile.rating_distribution=rating_dist
        profile.profile_summary=profile_summary
    else:
        # Create new profile
        profile = UserProfile(
            user_id=int(user_id),
            favorite_genres=favorite_genres,
            favourite_movies=fav_movies,
            disliked_movies=disliked_movies,
            disliked_genres=disliked_genres,
            avg_rating=avg_rating,
            num_movies_watched=num_movies_watched,
            rating_distribution=rating_dist,
            profile_summary=profile_summary,
            profile_json={}
        )
        session.add(profile)

    print(f"Processed profile for user {user_id}.")

session.commit()
print("All user profiles have been successfully generated and stored.")