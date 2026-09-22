#1. Load and preprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "dataset.csv"
df = pd.read_csv(DATA_PATH)

# Missing values: only 'Cuisines' has any (9 rows), so we drop them.
df = df.dropna(subset=["Cuisines"]).reset_index(drop=True)

# Target: primary cuisine, grouped so only the top N remain (rest -> "Other")
TOP_N = 12
df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
top_cuisines = df["Primary Cuisine"].value_counts().head(TOP_N).index
df["Cuisine Label"] = df["Primary Cuisine"].where(df["Primary Cuisine"].isin(top_cuisines), "Other")
print(df["Cuisine Label"].value_counts())

# Encode Yes/No columns as 1/0
for col in ["Has Table booking", "Has Online delivery", "Is delivering now"]:
    df[col] = (df[col] == "Yes").astype(int)

# Encode City as numeric codes
df["City"] = df["City"].astype("category").cat.codes

#2. Features / target
# Note: we do NOT use 'Cuisines' or 'Primary Cuisine' as features — that would leak the answer.
features = ["Country Code", "City", "Longitude", "Latitude", "Average Cost for two",
            "Has Table booking", "Has Online delivery", "Is delivering now",
            "Price range", "Votes", "Aggregate rating"]
target = "Cuisine Label"

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
)

#3. Train and compare models
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"),
}
predictions = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    predictions[name] = pred
    print(f"\n{name}: accuracy = {accuracy_score(y_test, pred):.3f}")

#4. Detailed evaluation (precision/recall per cuisine) for the better model
best_name = "Random Forest"
best_pred = predictions[best_name]
print(f"\nClassification report — {best_name}")
print(classification_report(y_test, best_pred))

#5. Confusion matrix — shows exactly which cuisines get confused with each other
labels = sorted(df[target].unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_name}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/task3_confusion_matrix.png")
plt.show()

#6. Discuss class imbalance / bias
# North Indian dwarfs every other class, so a model can get decent accuracy just by
# guessing North Indian often. Compare accuracy against the majority-class baseline:
baseline_acc = (y_test == y_test.mode()[0]).mean()
print(f"\nMajority-class baseline accuracy (always predict '{y_test.mode()[0]}'): {baseline_acc:.3f}")
print(f"{best_name} accuracy: {accuracy_score(y_test, best_pred):.3f}")
print("Compare per-class recall above: rare cuisines (e.g. Biryani, Beverages) will likely")
print("have much lower recall than North Indian or Chinese — that's the imbalance to call out.")