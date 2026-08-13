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
- 
# 🧠 MoodMate AI Recommendation Pipeline

Traditional recommendation systems mainly depend on historical user preferences:

User History  
↓  
Movie Preferences  
↓  
Movie Recommendation  

## **MoodMate adds the user's current context: **


            CURRENT CONTEXT

      ├──────────────┬──────────────┐
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
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    MOVIE ENGINE          ACTIVITY ENGINE
          │                     │
          ▼                     ▼
    PERSONALIZATION       ACTIVITY MODEL
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
              MOODMATE RESULTS

###  Datasets & Technology


# 📊 Datasets

 MovieLens

Used for movie recommendation and personalization.

- 3,883 movies
- 1,000,209 ratings
- 6,040 users
- 18 genres

 Emotion Dataset

Used for emotion classification.

- 16,000 samples
- TF-IDF features
- 6 emotion classes

UCI Human Activity Recognition

Used for activity recognition.

- 7,352 training samples
- 2,947 testing samples
- 561 features
- 6 activity classes

---

# 🛠️ Technology Stack

## Programming

- Python
- JavaScript
- HTML5
- CSS3

## Machine Learning

- Scikit-learn
- Pandas
- NumPy

## NLP

- TF-IDF
- Emotion Classification

## Recommendation System

- MovieLens
- Genre-based recommendation
- User preference modeling
- Context-aware ranking
- Movie similarity

## Activity Intelligence

- UCI HAR
- Activity Classification

## Backend

- Flask
- REST API
- JSON
- Gunicorn

## Frontend

- HTML
- CSS
- JavaScript
- Responsive Design
- Animations
- Cinematic UI

## Deployment

- GitHub
- Render
# 📁 Project Structure

```text
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
│   │
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
---

