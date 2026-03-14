import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ----------------------------
# 1. Load the dataset
# ----------------------------
movies = pd.read_csv("ml-latest-small/movies.csv")

# clean missing genres
movies["genres"] = movies["genres"].fillna("")

print("Dataset loaded")
print(movies.head())

# ----------------------------
# 2. Create genre features
# ----------------------------
vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
genre_matrix = vectorizer.fit_transform(movies["genres"])

# ----------------------------
# 3. Compute cosine similarity
# ----------------------------
similarity_matrix = cosine_similarity(genre_matrix)

# ----------------------------
# 4. Function to get similar movies
# ----------------------------
def get_similar_movies(title, top_n=10):

    if title not in movies["title"].values:
        print("Movie not found")
        return

    idx = movies[movies["title"] == title].index[0]

    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    scores = scores[1:top_n+1]

    results = []

    for i, score in scores:
        results.append({
            "title": movies.iloc[i]["title"],
            "similarity": round(score, 3)
        })

    return pd.DataFrame(results)

# ----------------------------
# 5. Query movies
# ----------------------------
query_movies = [
    "The Matrix (1999)",
    "Toy Story (1995)",
    "Finding Nemo (2003)"
]

for movie in query_movies:

    print("\nTop similar movies for:", movie)

    try:
        results = get_similar_movies(movie)
        print(results)

    except:
        print("Movie not found in dataset")

# ----------------------------
# 6. Visualization 1
# Genre distribution
# ----------------------------

all_genres = []

for g in movies["genres"]:
    for genre in g.split("|"):
        if genre != "(no genres listed)":
            all_genres.append(genre)

genre_counts = Counter(all_genres)

genres = list(genre_counts.keys())
counts = list(genre_counts.values())

plt.figure(figsize=(10,6))
plt.bar(genres, counts)
plt.xticks(rotation=90)
plt.title("Distribution of Movie Genres in MovieLens Dataset")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.tight_layout()

plt.savefig("genre_distribution.png")
plt.show()

# Visualization 2
# Toy Story similarity chart
# ----------------------------

example_movie = "Toy Story (1995)"

# find movie index
idx = movies[movies["title"] == example_movie].index[0]

# get similarity scores
scores = list(enumerate(similarity_matrix[idx]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

titles = [movies.iloc[i]["title"] for i,_ in scores]
values = [score for _,score in scores]

plt.figure(figsize=(10,6))
plt.barh(titles[::-1], values[::-1])
plt.title("Top Similar Movies to Toy Story (1995)")
plt.xlabel("Similarity Score")
plt.tight_layout()

plt.savefig("toy_story_similarity.png")

print("Saved toy_story_similarity.png")