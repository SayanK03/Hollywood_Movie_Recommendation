import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
def load_movie_data():
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                        movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    return movies_data, combined_features

# Train model and compute similarity
def train_recommendation_model(combined_features):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

# Get recommendations
def get_movie_recommendations(movie_name, movies_data, similarity):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return None  # No close match found

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    recommended_movies = []
    for movie in sorted_similar_movies[:10]:  # Top 10 recommendations
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)
    
    return recommended_movies
