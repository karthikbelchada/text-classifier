# =====================================================
# HYBRID NLP INTENT CLASSIFIER
# (SBERT + ML + RULES, Train/Test Split by Files)
# =====================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier

# -----------------------------------------------------
# 1. LOAD TRAIN & TEST DATA
# -----------------------------------------------------

TRAIN_PATH = "train_80.csv"
TEST_PATH = "test_20.csv"

TEXT_COL = "Social Copy"
LABEL_COL = "Content Pillar"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in (train_df, test_df):
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))


# -----------------------------------------------------
# 2. SINGLE LABEL → MULTI-LABEL
# -----------------------------------------------------


def to_multilabel(label):
    return {
        "is_event": int(label == "events"),
        "is_thought_leadership": int(label == "thought leadership"),
        "is_hiring": int(label == "hiring"),
        "is_award": int(label == "awards"),
    }


train_multi = train_df[LABEL_COL].apply(to_multilabel).apply(pd.Series)
test_multi = test_df[LABEL_COL].apply(to_multilabel).apply(pd.Series)

LABELS = train_multi.columns.tolist()

train_df = pd.concat([train_df, train_multi], axis=1)
test_df = pd.concat([test_df, test_multi], axis=1)


# -----------------------------------------------------
# 3. SBERT EMBEDDINGS (SEMANTIC CORE)
# -----------------------------------------------------

print("Loading Sentence-BERT model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

X_train_emb = embedder.encode(
    train_df[TEXT_COL].tolist(), normalize_embeddings=True, show_progress_bar=True
)

X_test_emb = embedder.encode(
    test_df[TEXT_COL].tolist(), normalize_embeddings=True, show_progress_bar=True
)

y_train = train_df[LABELS]
y_test = test_df[LABELS]


# -----------------------------------------------------
# 4. MULTI-LABEL CLASSIFIER
# -----------------------------------------------------

model = OneVsRestClassifier(LogisticRegression(max_iter=3000, class_weight="balanced"))

model.fit(X_train_emb, y_train)


# -----------------------------------------------------
# 5. THRESHOLDS
# -----------------------------------------------------

THRESHOLDS = {
    "is_event": 0.50,
    "is_thought_leadership": 0.50,
    "is_hiring": 0.60,
    "is_award": 0.40,
}

probs = model.predict_proba(X_test_emb)

y_pred = np.zeros_like(probs)
for i, label in enumerate(LABELS):
    y_pred[:, i] = (probs[:, i] >= THRESHOLDS[label]).astype(int)


# -----------------------------------------------------
# 6. RULE BOOSTERS (SAFE RULES ONLY)
# -----------------------------------------------------

AWARD_KEYWORDS = [
    "award",
    "awarded",
    "honored",
    "honoured",
    "recognized",
    "recognised",
    "winner",
    "finalist",
]

HIRING_KEYWORDS = [
    "hiring",
    "we are hiring",
    "join our team",
    "open position",
    "apply now",
]


def contains_any(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


for i, text in enumerate(test_df[TEXT_COL].tolist()):
    if contains_any(text, AWARD_KEYWORDS):
        y_pred[i, LABELS.index("is_award")] = 1
    if contains_any(text, HIRING_KEYWORDS):
        y_pred[i, LABELS.index("is_hiring")] = 1


# -----------------------------------------------------
# 7. MULTI-LABEL METRICS
# -----------------------------------------------------

print("\nMULTI-LABEL CLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

print("\nPER-LABEL ACCURACY")
for i, label in enumerate(LABELS):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{label}: {acc:.3f}")


# -----------------------------------------------------
# 8. TOTAL EXACT-MATCH ACCURACY
# -----------------------------------------------------

exact_match = (y_test.values == y_pred).all(axis=1)
total_correct = exact_match.sum()
total_samples = len(exact_match)

print("\nTOTAL EXACT-MATCH ACCURACY")
print(f"Correctly labelled samples: {total_correct} / {total_samples}")
print(f"Exact-match accuracy: {total_correct / total_samples:.3f}")


# -----------------------------------------------------
# 9. FINAL LABEL RESOLUTION
# -----------------------------------------------------


def resolve_label(prob_row, pred_row):
    if max(prob_row) < 0.45:
        return "others"

    flags = dict(zip(LABELS, pred_row))

    if flags["is_event"] and flags["is_thought_leadership"]:
        return "thought leadership"
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

true_labels = test_df[LABEL_COL].values


# -----------------------------------------------------
# 10. FINAL CONFUSION MATRIX
# -----------------------------------------------------

FINAL_LABELS = ["events", "thought leadership", "hiring", "awards", "others"]

cm = confusion_matrix(true_labels, final_preds, labels=FINAL_LABELS)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.colorbar()
plt.xticks(range(len(FINAL_LABELS)), FINAL_LABELS, rotation=45, ha="right")
plt.yticks(range(len(FINAL_LABELS)), FINAL_LABELS)

for i in range(len(FINAL_LABELS)):
    for j in range(len(FINAL_LABELS)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.title("Hybrid SBERT Intent Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_hybrid_split.png")
plt.close()

print("Saved: confusion_matrix_hybrid_split.png")


# -----------------------------------------------------
# 11. SAVE PREDICTIONS
# -----------------------------------------------------

output = test_df[[TEXT_COL]].copy()
output["true_label"] = true_labels
output["predicted_label"] = final_preds

output.to_csv("intent_engine_predictions_hybrid.csv", index=False)
print("Saved: intent_engine_predictions_hybrid.csv")

print("\nDONE.")
