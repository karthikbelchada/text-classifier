import re

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9\s,.!?]", " ", s)
    s = " ".join(s.split())
    return s.lower()


def normalize_label(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    s = " ".join(s.split())
    bad_labels = {
        "",
        "nan",
        "no social copy",
        "newsletter (no social copy)",
        "no-social-copy",
        "no_social_copy",
    }
    if s in bad_labels:
        return None
    if s in ["event", "events"]:
        return "events"
    if "thought" in s and "leadership" in s:
        return "thought leadership"
    return s


train_df = pd.read_csv("train_80.csv")
test_df = pd.read_csv("test_20.csv")

train_df["Content Pillar"] = train_df["Content Pillar"].apply(normalize_label)
test_df["Content Pillar"] = test_df["Content Pillar"].apply(normalize_label)

train_df.dropna(subset=["Content Pillar"], inplace=True)
test_df.dropna(subset=["Content Pillar"], inplace=True)

train_df["Social Copy"] = train_df["Social Copy"].apply(clean_text)
test_df["Social Copy"] = test_df["Social Copy"].apply(clean_text)

train_df = train_df[train_df["Social Copy"].str.strip() != ""].reset_index(drop=True)
test_df = test_df[test_df["Social Copy"].str.strip() != ""].reset_index(drop=True)

all_labels = sorted(train_df["Content Pillar"].unique())
print("\nCLEAN LABELS AFTER NORMALIZATION:")
print(all_labels)

y_train = train_df["Content Pillar"]
y_test = test_df["Content Pillar"]

tfidf_text = TfidfVectorizer(
    max_features=5000, stop_words="english", ngram_range=(1, 2)
)
X_train_text = tfidf_text.fit_transform(train_df["Social Copy"])
X_test_text = tfidf_text.transform(test_df["Social Copy"])

param_grid = {
    "C": [0.1, 1, 5, 10],
    "class_weight": [None, "balanced"],
    "solver": ["lbfgs"],
    "max_iter": [5000],
}

grid = GridSearchCV(
    LogisticRegression(), param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
grid.fit(X_train_text, y_train)
model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

y_pred = model.predict(X_test_text)
proba = model.predict_proba(X_test_text)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nImproved Accuracy: {round(acc, 4)}")
print(f"Macro F1: {round(macro_f1, 4)}")
print(f"Weighted F1: {round(weighted_f1, 4)}")

plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=all_labels, xticks_rotation="vertical"
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_text_category.png", dpi=120)
print("\nConfusion Matrix saved → confusion_matrix_text_category.png")

if "Company" in test_df.columns:
    company_col = test_df["Company"]
else:
    company_col = "Unknown"

output_test = pd.DataFrame(
    {
        "Company": company_col,
        "Category": test_df["Category"],
        "Social Copy": test_df["Social Copy"],
        "Content Pillar": test_df["Content Pillar"],
        "Predicted Content Pillar": y_pred,
        "Prediction Confidence": proba.max(axis=1).round(4),
        "Is_Correct": (y_pred == y_test.values),
    }
)

class_names = model.classes_
for i, cls in enumerate(class_names):
    output_test[f"Confidence_{cls}"] = proba[:, i].round(4)

output_test.to_csv("test_results_text_category.csv", index=False)
print("\nPredictions with Company saved → test_results_text_category.csv")
print(output_test["Predicted Content Pillar"].value_counts())
print("All done.")
