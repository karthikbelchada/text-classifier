import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

DATA_PATH = "test_20.csv"

df = pd.read_csv(DATA_PATH)

TEXT_COL = "Social Copy"
LABEL_COL = "Content Pillar"

df[TEXT_COL] = df[TEXT_COL].astype(str)
df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()

print("Columns:", df.columns.tolist())
print("Total rows:", len(df))


def to_multilabel(label):
    return {
        "is_event": int(label == "events"),
        "is_thought_leadership": int(label == "thought leadership"),
        "is_hiring": int(label == "hiring"),
        "is_award": int(label == "awards"),
    }


multi = df[LABEL_COL].apply(to_multilabel).apply(pd.Series)
df = pd.concat([df, multi], axis=1)

LABELS = ["is_event", "is_thought_leadership", "is_hiring", "is_award"]


X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABELS], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)


vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = OneVsRestClassifier(
    LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")
)

model.fit(X_train_vec, y_train)

THRESHOLDS = {
    "is_event": 0.50,
    "is_thought_leadership": 0.50,
    "is_hiring": 0.60,
    "is_award": 0.40,
}

probs = model.predict_proba(X_test_vec)
y_pred = np.zeros_like(probs)

for i, label in enumerate(LABELS):
    y_pred[:, i] = (probs[:, i] >= THRESHOLDS[label]).astype(int)

print("\nMULTI-LABEL CLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

print("\nPER-LABEL ACCURACY\n")
for i, label in enumerate(LABELS):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{label}: {acc:.3f}")

exact_match = (y_test.values == y_pred).all(axis=1)

total_correct = exact_match.sum()
total_samples = len(exact_match)
total_accuracy = total_correct / total_samples

print("\nTOTAL EXACT-MATCH ACCURACY")
print(f"Correctly labelled samples: {total_correct} / {total_samples}")
print(f"Exact-match accuracy: {total_accuracy:.3f}")

for i, label in enumerate(LABELS):
    cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])

    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix – {label}")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])

    for r in range(2):
        for c in range(2):
            plt.text(c, r, cm[r, c], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{label}.png")
    plt.close()

    print(f"Saved: confusion_matrix_{label}.png")


def resolve_label(prob_row, pred_row):
    if max(prob_row) < 0.45:
        return "others"

    flags = dict(zip(LABELS, pred_row))

    if flags["is_event"] and flags["is_thought_leadership"]:
        return "thought leadership (event-based)"
    if flags["is_event"]:
        return "events"
    if flags["is_thought_leadership"]:
        return "thought leadership"
    if flags["is_hiring"]:
        return "hiring"
    if flags["is_award"]:
        return "awards"

    return "others"


final_preds = [resolve_label(probs[i], y_pred[i]) for i in range(len(y_pred))]

true_labels = df.loc[y_test.index, LABEL_COL].values

FINAL_LABELS = [
    "events",
    "thought leadership",
    "thought leadership (event-based)",
    "hiring",
    "awards",
    "others",
]

cm_final = confusion_matrix(true_labels, final_preds, labels=FINAL_LABELS)

plt.figure(figsize=(10, 8))
plt.imshow(cm_final)
plt.colorbar()
plt.xticks(range(len(FINAL_LABELS)), FINAL_LABELS, rotation=45, ha="right")
plt.yticks(range(len(FINAL_LABELS)), FINAL_LABELS)

for i in range(len(FINAL_LABELS)):
    for j in range(len(FINAL_LABELS)):
        plt.text(j, i, cm_final[i, j], ha="center", va="center")

plt.title("Final Resolved Intent Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_final.png")
plt.close()

print("Saved: confusion_matrix_final.png")

output = pd.DataFrame(
    {"text": X_test.values, "true_label": true_labels, "predicted_label": final_preds}
)

output.to_csv("intent_engine_predictions.csv", index=False)
print("Saved: intent_engine_predictions.csv")

print("\nDONE.")
