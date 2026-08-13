import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD MOODMATE AI COMPONENTS
# ============================================================

print("Loading MoodMate AI...")

emotion_model = joblib.load(
    os.path.join(BASE_DIR, "emotion_model.pkl")
)

emotion_vectorizer = joblib.load(
    os.path.join(BASE_DIR, "emotion_vectorizer.pkl")
)

activity_model = joblib.load(
    os.path.join(BASE_DIR, "activity_model.pkl")
)

activity_labels = joblib.load(
    os.path.join(BASE_DIR, "activity_labels.pkl")
)

movies = joblib.load(
    os.path.join(BASE_DIR, "movies.pkl")
)

ratings = joblib.load(
    os.path.join(BASE_DIR, "ratings.pkl")
)

user_preference_matrix = joblib.load(
    os.path.join(BASE_DIR, "user_preference_matrix.pkl")
)

movie_genre_matrix = joblib.load(
    os.path.join(BASE_DIR, "movie_genre_matrix.pkl")
)

print("All MoodMate AI components loaded successfully")


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# USER PERSONALIZATION
# ============================================================

def get_personal_scores(user_id):

    if user_id not in user_preference_matrix.index:

        return pd.DataFrame({
            "movie_id": movies["movie_id"],
            "personal_score": 0.0
        })

    user_vector = user_preference_matrix.loc[user_id]

    genre_values = movie_genre_matrix.copy()

    common_genres = [
        g
        for g in user_vector.index
        if g in genre_values.columns
    ]

    if not common_genres:

        return pd.DataFrame({
            "movie_id": movies["movie_id"],
            "personal_score": 0.0
        })

    scores = genre_values[
        common_genres
    ].dot(
        user_vector[
            common_genres
        ]
    )

    result = pd.DataFrame({
        "movie_id": movie_genre_matrix.index,
        "personal_score": scores.values
    })

    result["personal_score"] = minmax_normalize(
        result["personal_score"]
    )

    return result


# ============================================================
# MOOD SCORE
# ============================================================

def calculate_mood_score(genres, mood):

    genres = str(genres).lower()
    mood = str(mood).lower()

    mood_genres = {

        "joy": [
            "comedy",
            "animation",
            "musical",
            "children",
            "romance"
        ],

        "sadness": [
            "drama",
            "romance"
        ],

        "anger": [
            "action",
            "thriller"
        ],

        "fear": [
            "horror",
            "thriller"
        ],

        "surprise": [
            "adventure",
            "sci-fi",
            "fantasy"
        ],

        "love": [
            "romance",
            "comedy",
            "drama"
        ],

        "neutral": [
            "drama",
            "documentary"
        ]
    }

    preferred = mood_genres.get(
        mood,
        mood_genres["neutral"]
    )

    matches = sum(
        1
        for genre in preferred
        if genre in genres
    )

    if matches == 0:
        return 0.5

    return min(
        1.0,
        0.4 + 0.3 * matches
    )


# ============================================================
# MOVIE ENGINE
# ============================================================

def get_mood_movies(
    user_id=1,
    mood="joy",
    hour=20,
    weather="rain",
    energy="medium",
    n=5
):

    candidates = movies.copy()


    # --------------------------------------------------------
    # PERSONAL SCORE
    # --------------------------------------------------------

    personal = get_personal_scores(user_id)

    candidates = candidates.merge(
        personal,
        on="movie_id",
        how="left"
    )

    candidates["personal_score"] = (
        candidates["personal_score"]
        .fillna(0)
    )

    candidates["personal_score_norm"] = (
        minmax_normalize(
            candidates["personal_score"]
        )
    )


    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    if (
        "avg_rating" not in candidates.columns
        and "movie_id" in ratings.columns
    ):

        if "avg_rating" in ratings.columns:

            rating_columns = [
                "movie_id",
                "avg_rating"
            ]

            if "rating_count" in ratings.columns:
                rating_columns.append(
                    "rating_count"
                )

            rating_data = ratings[
                rating_columns
            ].drop_duplicates(
                "movie_id"
            )

            candidates = candidates.merge(
                rating_data,
                on="movie_id",
                how="left"
            )


    if "avg_rating" not in candidates.columns:
        candidates["avg_rating"] = 0.0

    candidates["avg_rating"] = pd.to_numeric(
        candidates["avg_rating"],
        errors="coerce"
    ).fillna(0)

    candidates["rating_norm"] = (
        candidates["avg_rating"] / 5.0
    ).clip(0, 1)


    # --------------------------------------------------------
    # MOOD
    # --------------------------------------------------------

    candidates["mood_score"] = candidates[
        "genres"
    ].apply(
        lambda value: calculate_mood_score(
            value,
            mood
        )
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    def time_score(genres):

        genres = str(genres).lower()

        if 17 <= hour <= 22:

            if any(
                genre in genres
                for genre in [
                    "comedy",
                    "romance",
                    "animation",
                    "drama"
                ]
            ):
                return 1.0

            return 0.6

        if 6 <= hour < 12:
            return 0.8

        return 0.7


    candidates["time_score"] = (
        candidates["genres"].apply(
            time_score
        )
    )


    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    def weather_score(genres):

        genres = str(genres).lower()

        if weather.lower() == "rain":

            if any(
                genre in genres
                for genre in [
                    "comedy",
                    "romance",
                    "animation",
                    "drama"
                ]
            ):
                return 1.0

            return 0.85

        return 0.9


    candidates["weather_score"] = (
        candidates["genres"].apply(
            weather_score
        )
    )


    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    def energy_score(genres):

        genres = str(genres).lower()

        if energy.lower() == "medium":

            if any(
                genre in genres
                for genre in [
                    "comedy",
                    "adventure",
                    "animation"
                ]
            ):
                return 0.95

            return 0.85

        if energy.lower() == "high":

            if any(
                genre in genres
                for genre in [
                    "action",
                    "adventure",
                    "thriller"
                ]
            ):
                return 1.0

            return 0.8

        return 0.8


    candidates["energy_score"] = (
        candidates["genres"].apply(
            energy_score
        )
    )


    # --------------------------------------------------------
    # CONTEXT SCORE
    # --------------------------------------------------------

    candidates["context_score"] = (
        0.30 * candidates["mood_score"]
        + 0.15 * candidates["time_score"]
        + 0.15 * candidates["weather_score"]
        + 0.15 * candidates["energy_score"]
    )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    candidates["final_score"] = (
        0.35 * candidates["personal_score_norm"]
        + 0.25 * candidates["mood_score"]
        + 0.20 * candidates["rating_norm"]
        + 0.20 * candidates["context_score"]
    ).clip(0, 1)


    candidates["match_percent"] = (
        candidates["final_score"] * 100
    ).round(1)


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_columns = [
        "movie_id",
        "title",
        "genres",
        "avg_rating",
        "personal_score_norm",
        "mood_score",
        "time_score",
        "weather_score",
        "energy_score",
        "rating_norm",
        "context_score",
        "final_score",
        "match_percent"
    ]

    result = candidates[
        [
            column
            for column in result_columns
            if column in candidates.columns
        ]
    ].sort_values(
        "final_score",
        ascending=False
    ).head(n)

    return result


# ============================================================
# ACTIVITY ENGINE
# ============================================================

def get_activities(
    mood,
    weather,
    energy
):

    mood = str(mood).lower()
    weather = str(weather).lower()
    energy = str(energy).lower()

    activities = []


    if mood == "joy":

        activities.append({
            "title": "Dance to your favorite songs",
            "description":
                "Use the positive mood for movement.",
            "duration": 15
        })


    if weather == "rain":

        activities.append({
            "title": "Indoor mobility",
            "description":
                "Move around indoors and step away from the screen.",
            "duration": 15
        })


    activities.append({
        "title": "Stretch session",
        "description":
            "Loosen your body and reset your attention.",
        "duration": 10
    })


    return activities[:3]


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/results.html", methods=["GET"])
def results():

    return send_from_directory(
        BASE_DIR,
        "results.html"
    )


# ============================================================
# MOODMATE API
# ============================================================

@app.route("/api/moodmate", methods=["POST"])
def moodmate_api():

    data = request.get_json(
        silent=True
    ) or {}


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    try:

        user_id = int(
            data.get(
                "user_id",
                1
            )
        )

        hour = int(
            data.get(
                "hour",
                20
            )
        )

    except (TypeError, ValueError):

        return jsonify({
            "error":
                "user_id and hour must be integers"
        }), 400


    mood = data.get(
        "mood",
        "joy"
    )

    weather = data.get(
        "weather",
        "rain"
    )

    energy = data.get(
        "energy",
        "medium"
    )


    # --------------------------------------------------------
    # GENERATE MOVIES
    # --------------------------------------------------------

    movies_result = get_mood_movies(
        user_id=user_id,
        mood=mood,
        hour=hour,
        weather=weather,
        energy=energy,
        n=5
    )


    # --------------------------------------------------------
    # GENERATE ACTIVITIES
    # --------------------------------------------------------

    activities = get_activities(
        mood,
        weather,
        energy
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({

        "context": {

            "mood": mood,

            "hour": hour,

            "weather": weather,

            "energy": energy

        },

        "movies":
            movies_result.to_dict(
                orient="records"
            ),

        "activities":
            activities

    })


# ============================================================
# RENDER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
