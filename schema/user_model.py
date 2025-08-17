from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = 'user_profiles'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True)
    favorite_genres = Column(ARRAY(String))
    favourite_movies = Column(ARRAY(String))
    disliked_movies = Column(ARRAY(String))
    disliked_genres = Column(ARRAY(String))
    avg_rating = Column(Float)
    num_movies_watched = Column(Integer)
    rating_distribution = Column(JSON)
    profile_summary = Column(String)
    profile_json = Column(JSON)  # For flexible, deep insights

# Replace with your actual Postgres credentials
engine = create_engine('postgresql://postgres:Bhavik30@localhost:5432/movierec')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)