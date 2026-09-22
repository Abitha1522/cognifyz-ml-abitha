#1. Load and preprocess
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "dataset.csv" 
df = pd.read_csv(DATA_PATH)

# Missing values: only 'Cuisines' has any (9 rows), so we drop them.
df = df.dropna(subset=["Cuisines"])

# Unrated restaurants (rating 0) have no quality signal, so we don't recommend them.
df = df[df["Aggregate rating"] > 0].reset_index(drop=True)

# Encode Yes/No columns as 1/0
for col in ["Has Table booking", "Has Online delivery"]:
    df[col] = (df[col] == "Yes").astype(int)

#2. Encode cuisines
# Turn "North Indian, Chinese" into a multi-hot vector: one column per cuisine.
vectorizer = CountVectorizer(
    tokenizer=lambda s: [c.strip().lower() for c in s.split(",")],
    lowercase=False, token_pattern=None, binary=True,
)
cuisine_matrix = vectorizer.fit_transform(df["Cuisines"])
print("Restaurants:", cuisine_matrix.shape[0], "| Distinct cuisines:", cuisine_matrix.shape[1])

#3. Recommendation function
# Criteria: cuisine (main signal), price range (closeness), plus optional filters.
#   score = 0.7 * cuisine similarity + 0.3 * price closeness
def recommend(cuisines, city=None, price_range=None, min_rating=0.0,
              online_delivery=False, table_booking=False, top_n=5):
    # Hard filters first: city, rating, delivery, booking
    mask = df["Aggregate rating"] >= min_rating
    if city:
        mask &= df["City"].str.lower() == city.lower()
    if online_delivery:
        mask &= df["Has Online delivery"] == 1
    if table_booking:
        mask &= df["Has Table booking"] == 1
    candidates = df[mask]
    if candidates.empty:
        return candidates

    # Cuisine similarity: cosine between the user's cuisines and each restaurant's cuisines
    user_vec = vectorizer.transform([", ".join(cuisines)])
    cuisine_sim = cosine_similarity(user_vec, cuisine_matrix[candidates.index]).ravel()

    # Price closeness: 1.0 = exact match, 0.0 = furthest apart (price range is 1 to 4)
    if price_range is not None:
        price_sim = 1 - (candidates["Price range"] - price_range).abs() / 3
    else:
        price_sim = pd.Series(1.0, index=candidates.index)

    out = candidates.copy()
    out["Match score"] = 0.7 * cuisine_sim + 0.3 * price_sim.values
    out = out.sort_values(["Match score", "Aggregate rating", "Votes"], ascending=False)
    cols = ["Restaurant Name", "City", "Cuisines", "Price range", "Aggregate rating", "Match score"]
    return out[cols].head(top_n).round(2)

#4. Test with sample users
pd.set_option("display.width", 250, "display.max_colwidth", 40, "display.max_columns", None)

print("User A: wants Italian, New Delhi, mid-to-high price (3)")
print(recommend(["Italian"], city="New Delhi", price_range=3), "\n")

print("User B: wants North Indian + Mughlai, Gurgaon, budget-friendly (2), online delivery")
print(recommend(["North Indian", "Mughlai"], city="Gurgaon", price_range=2, online_delivery=True), "\n")

print("User C: wants Chinese + Fast Food, Noida, cheap (1), rating 4.0+")
print(recommend(["Chinese", "Fast Food"], city="Noida", price_range=1, min_rating=4.0), "\n")