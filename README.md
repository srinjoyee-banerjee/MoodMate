# 🎬 MoodMate AI — Project Summary

MoodMate AI is an **AI-powered, context-aware recommendation system** designed to help users decide what to watch or what to do based on their current **mood, time, weather, and energy level**.

Instead of encouraging endless scrolling through content, MoodMate creates a personalized **"moment"** for the user.

> **Don't just scroll. Feel. 🧡**

---

## 🧠 What is MoodMate?

MoodMate combines multiple Artificial Intelligence and Machine Learning components into one recommendation pipeline.

The system considers:

- 😊 User Mood
- 🕐 Time of Day
- 🌦️ Weather
- ⚡ Energy Level
- 👤 User Preferences
- 🎬 Movie Genres
- ⭐ Movie Ratings
- 🚶 Human Activity Recognition

The system then generates personalized recommendations for:

- 🎬 Movies
- 🚶 Activities
- 🧘 Relaxation / movement suggestions

---

# 🚀 Core Idea

Traditional recommendation systems mainly depend on historical user preferences:

```text
User History
     ↓

Movie Preferences
     ↓
Movie Recommendation
                 CURRENT CONTEXT
                       │
        ┌──────────────┼──────────────┐
        │              │              │
       Mood           Time         Weather
        │              │              │
        └──────────────┼──────────────┘
                       │
                     Energy
                       │
                       ▼
              CONTEXT ANALYSIS
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   MOVIE ENGINE               ACTIVITY ENGINE
          │                         │
          ▼                         ▼
   PERSONALIZATION           ACTIVITY MODEL
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
               MOODMATE RESULTS
Movie Recommendation

The movie engine combines:

User genre preferences
Movie genres
Movie ratings
Mood compatibility
Time compatibility
Weather compatibility
Energy compatibility

Datasets
MovieLens

Used for movie recommendation and personalization.
Used for emotion classification.
UCI Human Activity Recognition

Used for activity recognition.
Technology Stack
Programming
Python
JavaScript
HTML5
CSS3
Machine Learning
Scikit-learn
Pandas
NumPy
NLP
TF-IDF
Emotion Classification
Recommendation System
MovieLens
Genre-based recommendation
User preference modeling
Context-aware ranking
Movie similarity
Activity Intelligence
UCI HAR
Activity Classification
Backend
Flask
REST API
JSON
Gunicorn
Frontend
HTML
CSS
JavaScript
Responsive design
Animations
Cinematic UI
Deployment
GitHub
Render
MoodMate/
│
├── frontend/
│   ├── index.html
│   ├── results.html
│   └── about.html
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── models/
│       ├── activity_labels.pkl
│       ├── activity_model.pkl
│       ├── emotion_model.pkl
│       ├── emotion_vectorizer.pkl
│       ├── movie_genre_matrix.pkl
│       ├── movies.pkl
│       ├── ratings.pkl
│       └── user_preference_matrix.pkl
│
└── README.md
🎬 Don't Just Scroll.
🧡 Feel the Moment.


