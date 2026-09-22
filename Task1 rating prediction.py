#1. Load data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = r"C:\Users\HP\Downloads\dataset.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

#2. Preprocess

# Fill missing Cuisines
df["Cuisines"] = df["Cuisines"].fillna("Unknown")

# Number of cuisines
df["Num Cuisines"] = df["Cuisines"].str.split(",").str.len()

# Encode Yes/No columns as 1/0
for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
    df[col] = (df[col] == "Yes").astype(int)

# Encode City as numeric codes
df["City"] = df["City"].astype("category").cat.codes

# Features
features = [
    "Country Code",
    "City",
    "Longitude",
    "Latitude",
    "Average Cost for two",
    "Has Table booking",
    "Has Online delivery",
    "Is delivering now",
    "Price range",
    "Votes",
    "Num Cuisines"
]

target = "Aggregate rating"

#3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=0.2,
    random_state=42
)

#4. Train and compare models

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "MSE": mean_squared_error(y_test, pred),
        "R2": r2_score(y_test, pred)
    })

print("\nModel Results:")
print(pd.DataFrame(results).round(3))

#5. Feature importance

importance = pd.Series(
    models["Random Forest"].feature_importances_,
    index=features
)

importance.sort_values().plot(
    kind="barh",
    figsize=(7, 5),
    title="What drives restaurant ratings?"
)

plt.tight_layout()
plt.show()

#6. Extra analysis

def run_rf(data, label):

    X_tr, X_te, y_tr, y_te = train_test_split(
        data[features],
        data[target],
        test_size=0.2,
        random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_tr, y_tr)

    pred = rf.predict(X_te)

    imp = pd.Series(
        rf.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print(
        f"{label}: rows={len(data)} "
        f"MSE={mean_squared_error(y_te, pred):.3f} "
        f"R2={r2_score(y_te, pred):.3f} "
        f"top features: "
        + ", ".join(
            f"{k} ({v:.2f})"
            for k, v in imp.head(4).items()
        )
    )

run_rf(df, "All restaurants")
run_rf(df[df[target] > 0], "Rated only (>0)")