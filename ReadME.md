# 🎬 CineSuggest AI - Movie Recommendation System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![TMDB](https://img.shields.io/badge/TMDB-API-blue?style=for-the-badge)

</p>

---

# 📌 Overview

**CineSuggest AI** is a modern **Content-Based Movie Recommendation System** that intelligently recommends similar movies using **Natural Language Processing (NLP)** techniques and **TF-IDF Vectorization**.

The system analyzes movie metadata such as title, overview, genres, and keywords to compute similarity scores and generate personalized recommendations.

It also integrates with the **TMDB API** to display movie posters, backdrops, ratings, release dates, and detailed movie information inside an interactive **Streamlit dashboard**.

---

# 🚀 Features

- 🎬 Smart Movie Search
- 🔎 Autocomplete Search Suggestions
- 🧠 TF-IDF Based Content Recommendation
- 🎭 Genre-Based Recommendation
- 🖼️ High Quality Movie Posters
- 🎨 Responsive Streamlit UI
- ⭐ TMDB Ratings
- 📅 Release Date Information
- 📖 Movie Overview
- ⚡ Fast Cached API Calls
- 🔄 Local & Production API Switching
- 📱 Responsive Grid Layout
- 🚀 Real-Time Recommendation Engine

---

# 🧠 Machine Learning Workflow

```
Movie Metadata
        │
        ▼
Data Cleaning
        │
        ▼
Text Preprocessing
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Cosine Similarity Matrix
        │
        ▼
Top Similar Movies
        │
        ▼
Genre Recommendation
        │
        ▼
Streamlit Dashboard
```

---

# 🛠 Tech Stack

### Programming

- Python

### Machine Learning

- TF-IDF Vectorizer
- Cosine Similarity
- Scikit-learn

### Frontend

- Streamlit

### Backend

- FastAPI

### Data Processing

- Pandas
- NumPy
- SciPy

### API

- TMDB API

### Others

- Pickle Serialization
- Dotenv
- Requests

---

# 📂 Project Structure

```
movie-rec/

│── app.py
│── main.py
│── movies.ipynb
│── movies_metadata.csv
│── df.pkl
│── indices.pkl
│── tfidf.pkl
│── tfidf_matrix.pkl
│── requirements.txt
│── runtime.txt
│── .env
│── README.md
```

---

# ⚙️ Installation

## Clone Repository

```
git clone https://github.com/yourusername/CineSuggest-AI.git

cd CineSuggest-AI
```

---

## Create Virtual Environment

```
python -m venv venv
```

Activate

Windows

```
venv\Scripts\activate
```

Linux

```
source venv/bin/activate
```

---

## Install Requirements

```
pip install -r requirements.txt
```

---

## Configure Environment

Create

```
.env
```

Add

```
TMDB_API_KEY=YOUR_API_KEY
```

---

# ▶️ Run FastAPI Backend

```
uvicorn main:app --reload
```

---

# ▶️ Run Streamlit Frontend

```
streamlit run app.py
```

---

# 🎯 Recommendation Pipeline

```
User Searches Movie
          │
          ▼
Autocomplete Search
          │
          ▼
Movie Selected
          │
          ▼
Fetch Movie Details
          │
          ▼
TF-IDF Recommendation
          │
          ▼
Genre Recommendation
          │
          ▼
Display Posters & Details
```

---

# 📸 Screenshots

## Home Page

```
(Add Screenshot Here)
```

---

## Search Results

```
(Add Screenshot Here)
```

---

## Movie Details

```
(Add Screenshot Here)
```

---

## Recommendation Page

```
(Add Screenshot Here)
```


---

# 📊 Dataset

The project utilizes movie metadata including:

- Movie Title
- Overview
- Genres
- Release Date
- Popularity
- Vote Average
- TMDB ID
- Poster Path
- Backdrop Path

to generate intelligent recommendations.

---

# 🔮 Future Improvements

- 👤 User Login
- ❤️ Favorite Movies
- ⭐ Personalized Recommendation
- 🤖 Hybrid Recommendation Engine
- 🎥 Trailer Integration
- 💬 User Reviews
- 🌙 Dark Mode
- 📱 Mobile Responsive Design
- ☁️ Cloud Deployment

---

# 👨‍💻 Author

## Aditya Raikar

**Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer**

GitHub:
https://github.com/adityaraikar5555-collab

LinkedIn:
https://www.linkedin.com/in/aditya-raikar05

---

# ⭐ Show Your Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

🛠️ Contribute to improve it

and share it with the community!

---

## "Discover movies you'll love with the power of Machine Learning and AI."