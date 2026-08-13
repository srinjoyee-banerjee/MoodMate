import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ============================================================
# FLASK SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

CORS(app)


# ============================================================
# LOAD MOODMATE AI COMPONENTS
# ============================================================

print("Loading MoodMate AI...")


def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        print("WARNING:", filename, "not found")
        return None

    return joblib.load(path)


emotion_model = load_pickle("emotion_model.pkl")
emotion_vectorizer = load_pickle("emotion_vectorizer.pkl")

activity_model = load_pickle("activity_model.pkl")
activity_labels = load_pickle("activity_labels.pkl")

movies = load_pickle("movies.pkl")
ratings = load_pickle("ratings.pkl")

user_preference_matrix = load_pickle(
    "user_preference_matrix.pkl"
)

movie_genre_matrix = load_pickle(
    "movie_genre_matrix.pkl"
)

print("MoodMate components loaded successfully.")


# ============================================================
# NORMALIZATION
# ============================================================

def minmax_normalize(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).fillna(0)

    if len(series) == 0:
        return series

    minimum = series.min()
    maximum = series.max()

    if minimum == maximum:
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
# PERSONALIZATION
# ============================================================

def get_personal_scores(user_id):

    if (
        movies is None
        or user_preference_matrix is None
        or movie_genre_matrix is None
    ):
        return pd.DataFrame({
            "movie_id": [],
            "personal_score": []
        })

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

        "movie_id":
            movie_genre_matrix.index,

        "personal_score":
            scores.values

    })

    result["personal_score"] = (
        minmax_normalize(
            result["personal_score"]
        )
    )

    return result


# ============================================================
# MOOD SCORING
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
# MOVIE RECOMMENDATION ENGINE
# ============================================================

def get_mood_movies(
    user_id=1,
    mood="joy",
    hour=20,
    weather="rain",
    energy="medium",
    n=5
):

    if movies is None:
        return pd.DataFrame()

    candidates = movies.copy()

    # --------------------------------------------------------
    # PERSONAL SCORE
    # --------------------------------------------------------

    personal = get_personal_scores(
        user_id
    )

    if not personal.empty:

        candidates = candidates.merge(
            personal,
            on="movie_id",
            how="left"
        )

    else:

        candidates["personal_score"] = 0.0

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
    # RATINGS
    # --------------------------------------------------------

    if (
        "avg_rating" not in candidates.columns
        and ratings is not None
    ):

        if (
            "movie_id" in ratings.columns
            and "avg_rating" in ratings.columns
        ):

            rating_columns = [
                "movie_id",
                "avg_rating"
            ]

            if "rating_count" in ratings.columns:
                rating_columns.append(
                    "rating_count"
                )

            rating_data = (
                ratings[rating_columns]
                .drop_duplicates("movie_id")
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

    candidates["mood_score"] = (
        candidates["genres"].apply(
            lambda x:
            calculate_mood_score(
                x,
                mood
            )
        )
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    def time_score(genres):

        genres = str(genres).lower()

        if 17 <= hour <= 22:

            if any(
                g in genres
                for g in [
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

        if weather == "rain":

            if any(
                g in genres
                for g in [
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

        if energy == "high":

            if any(
                g in genres
                for g in [
                    "action",
                    "adventure",
                    "thriller"
                ]
            ):
                return 1.0

            return 0.8

        if energy == "low":

            if any(
                g in genres
                for g in [
                    "drama",
                    "romance",
                    "animation"
                ]
            ):
                return 0.95

            return 0.8

        if energy == "medium":

            if any(
                g in genres
                for g in [
                    "comedy",
                    "adventure",
                    "animation"
                ]
            ):
                return 0.95

            return 0.85

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
    # FINAL AI SCORE
    # --------------------------------------------------------

    candidates["final_score"] = (

        0.35 *
        candidates["personal_score_norm"]

        + 0.25 *
        candidates["mood_score"]

        + 0.20 *
        candidates["rating_norm"]

        + 0.20 *
        candidates["context_score"]

    ).clip(0, 1)

    candidates["match_percent"] = (
        candidates["final_score"] * 100
    ).round(1)

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
            c
            for c in result_columns
            if c in candidates.columns
        ]
    ].sort_values(
        "final_score",
        ascending=False
    ).head(n)

    return result


# ============================================================
# ACTIVITIES
# ============================================================

def get_activities(
    mood,
    weather,
    energy
):

    activities = []

    mood = str(mood).lower()
    weather = str(weather).lower()
    energy = str(energy).lower()

    if mood == "joy":

        activities.append({

            "title":
                "Dance to your favorite songs",

            "description":
                "Use the positive mood for movement instead of endless scrolling.",

            "duration": 15,

            "icon": "🎵"

        })

    if weather == "rain":

        activities.append({

            "title":
                "Indoor mobility",

            "description":
                "Move around indoors and step away from the screen.",

            "duration": 15,

            "icon": "🏠"

        })

    else:

        activities.append({

            "title":
                "Take a short outdoor walk",

            "description":
                "Use the surroundings to reset your attention.",

            "duration": 15,

            "icon": "🌿"

        })

    if energy == "high":

        activities.append({

            "title":
                "Quick movement challenge",

            "description":
                "Channel your energy into a short active break.",

            "duration": 15,

            "icon": "🔥"

        })

    elif energy == "low":

        activities.append({

            "title":
                "Slow stretch & breathe",

            "description":
                "Give your body a gentle reset.",

            "duration": 10,

            "icon": "🧘"

        })

    else:

        activities.append({

            "title":
                "Stretch session",

            "description":
                "Loosen your body and reset your attention.",

            "duration": 10,

            "icon": "🧘"

        })

    return activities[:3]


# ============================================================
# YOUR MOMENT — AI CONTEXT EXPLANATION
# ============================================================

def build_moment(
    mood,
    time_of_day,
    weather,
    energy,
    hour
):

    mood_text = {

        "joy":
            "You seem ready for something light, positive and engaging.",

        "sadness":
            "This looks like a moment for comfort rather than pressure.",

        "anger":
            "You may benefit from releasing some energy before settling in.",

        "fear":
            "A calm, familiar and low-pressure experience may fit better right now.",

        "love":
            "Your moment has a warm, connection-oriented tone.",

        "surprise":
            "You seem open to novelty and something a little unexpected.",

        "neutral":
            "You appear to be in a balanced state, so a flexible recommendation fits."

    }.get(
        mood,
        "Your current mood suggests a balanced experience."
    )

    weather_text = {

        "rain":
            "The rainy setting makes an indoor experience especially natural.",

        "cloudy":
            "The cloudy surroundings support a calm, flexible plan.",

        "sunny":
            "The brighter surroundings leave room for outdoor movement.",

        "snow":
            "The colder setting makes a cozy or indoor plan appealing."

    }.get(
        weather,
        ""
    )

    energy_text = {

        "low":
            "Your lower energy suggests keeping the next step gentle.",

        "medium":
            "Your medium energy gives you room for relaxing or light movement.",

        "high":
            "Your higher energy can be channelled into something active or exciting."

    }.get(
        energy,
        ""
    )

    return {

        "headline":
            f"A {mood} moment, shaped by your context.",

        "explanation":
            f"{mood_text} {weather_text} {energy_text}".strip(),

        "time_of_day":
            time_of_day,

        "mood":
            mood,

        "weather":
            weather,

        "energy":
            energy,

        "hour":
            hour

    }


# ============================================================
# FIVE HTML PAGES
# ============================================================

@app.route("/")
@app.route("/index.html")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/discover.html")
def discover():

    return send_from_directory(
        BASE_DIR,
        "discover.html"
    )


@app.route("/moment.html")
def moment():

    return send_from_directory(
        BASE_DIR,
        "moment.html"
    )


@app.route("/recommendations.html")
def recommendations():

    return send_from_directory(
        BASE_DIR,
        "recommendations.html"
    )


@app.route("/journal.html")
def journal():

    return send_from_directory(
        BASE_DIR,
        "journal.html"
    )


# ============================================================
# AI API
# ============================================================

@app.route(
    "/api/moodmate",
    methods=["POST"]
)
def moodmate_api():

    data = request.get_json(
        silent=True
    ) or {}

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

    except (
        TypeError,
        ValueError
    ):

        return jsonify({
            "error":
                "user_id and hour must be integers"
        }), 400

    mood = str(
        data.get(
            "mood",
            "joy"
        )
    ).lower()

    weather = str(
        data.get(
            "weather",
            "rain"
        )
    ).lower()

    energy = str(
        data.get(
            "energy",
            "medium"
        )
    ).lower()

    time_of_day = str(
        data.get(
            "time_of_day",
            "evening"
        )
    ).lower()

    # AI movie recommendations
    movie_results = get_mood_movies(

        user_id=user_id,

        mood=mood,

        hour=hour,

        weather=weather,

        energy=energy,

        n=5

    )

    # Activities
    activities = get_activities(

        mood,
        weather,
        energy

    )

    # Context explanation
    moment_data = build_moment(

        mood,
        time_of_day,
        weather,
        energy,
        hour

    )

    return jsonify({

        "context": {

            "mood": mood,

            "time_of_day":
                time_of_day,

            "hour": hour,

            "weather": weather,

            "energy": energy

        },

        "moment":
            moment_data,

        "movies":
            movie_results.to_dict(
                orient="records"
            )
            if not movie_results.empty
            else [],

        "activities":
            activities

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "app": "MoodMate",

        "status": "online",

        "version": "2.0",

        "pages": 5

    })


# ============================================================
# RENDER
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
