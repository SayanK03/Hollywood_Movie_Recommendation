from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from main import load_movie_data, train_recommendation_model, get_movie_recommendations

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load movie data and train the model
movies_data, combined_features = load_movie_data()
similarity = train_recommendation_model(combined_features)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_name = data.get('movie_name', '')

    # Get recommendations
    recommended_movies = get_movie_recommendations(movie_name, movies_data, similarity)

    if not recommended_movies:
        return jsonify({'error': 'Movie not found'}), 404

    return jsonify({'recommendations': recommended_movies})

if __name__ == '__main__':
    app.run(debug=True)
