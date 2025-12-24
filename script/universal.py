import html
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import GridSearchCV

nlp = spacy.load("en_core_web_lg")

COGNITIVE_PROTOTYPE = nlp(
    "insight learning idea perspective reflection analysis discussion deep dive"
)

TRANSACTIONAL_PROTOTYPE = nlp("register signup rsvp ticket booking reserve enroll link")

EXPERIENCE_TERMS = {
    "day",
    "session",
    "summit",
    "conference",
    "meetup",
    "workshop",
    "event",
    "talk",
    "panel",
}


def clean_text(s):
    if pd.isna(s):
        return ""
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"http\S+|www\S+|lnkd\.in\S+", "", s)
    s = re.sub(r"[^a-zA-Z0-9\s,.!?#]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_label(s):
    if pd.isna(s):
        return None
    s = str(s).strip().lower()
    if s in {"", "nan"}:
        return None
    if s.startswith("event"):
        return "events"
    if "thought" in s:
        return "thought leadership"
    return s


def extract_hashtags(text):
    return [tag.lower() for tag in re.findall(r"#\w+", text)]


def extract_spacy_features(text):
    doc = nlp(text)
    nouns = sorted({tok.lemma_.lower() for tok in doc if tok.pos_ in {"NOUN", "PROPN"}})
    verbs = sorted({tok.lemma_.lower() for tok in doc if tok.pos_ == "VERB"})
    orgs = sorted({ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"})
    dates = sorted(
        {ent.text.lower() for ent in doc.ents if ent.label_ in {"DATE", "TIME"}}
    )
    locations = sorted(
        {ent.text.lower() for ent in doc.ents if ent.label_ in {"GPE", "LOC"}}
    )
    return nouns, verbs, orgs, dates, locations


def semantic_match(text, prototype, threshold=0.78):
    doc = nlp(text)
    if not doc.vector_norm:
        return False
    return doc.similarity(prototype) >= threshold


def has_event_metadata(doc):
    return any(ent.label_ in {"DATE", "TIME", "GPE", "LOC"} for ent in doc.ents)


def has_call_to_action(doc):
    for tok in doc:
        if tok.pos_ == "VERB" and tok.dep_ in {"ROOT", "xcomp"}:
            if tok.morph.get("VerbForm") == ["Inf"]:
                return True
    return False


def is_retrospective(doc):
    past_verbs = [
        tok for tok in doc if tok.pos_ == "VERB" and tok.morph.get("Tense") == ["Past"]
    ]
    return len(past_verbs) / max(len(doc), 1) > 0.10


def is_latent_event_experience(doc, text):
    has_experience_word = any(w in text for w in EXPERIENCE_TERMS)
    has_org_or_loc = any(ent.label_ in {"ORG", "GPE", "LOC"} for ent in doc.ents)
    past_heavy = is_retrospective(doc)
    return has_experience_word and has_org_or_loc and past_heavy


def is_pure_idea_post(doc):
    content_verbs = [
        tok for tok in doc if tok.pos_ == "VERB" and tok.lemma_ not in {"be", "have"}
    ]
    return len(content_verbs) <= 1 and len(doc.ents) == 0


def intent_resolver(text, base_label, base_prob):
    doc = nlp(text)

    if base_prob >= 0.85:
        return base_label, "very_high_confidence_model"

    if is_latent_event_experience(doc, text):
        return "events", "latent_event_experience"

    if (
        semantic_match(text, TRANSACTIONAL_PROTOTYPE)
        and has_event_metadata(doc)
        and has_call_to_action(doc)
    ):
        return "events", "strong_semantic_event"

    if is_pure_idea_post(doc):
        return "thought leadership", "pure_idea_expression"

    if (
        semantic_match(text, COGNITIVE_PROTOTYPE)
        and not semantic_match(text, TRANSACTIONAL_PROTOTYPE)
        and not has_event_metadata(doc)
    ):
        return "thought leadership", "strong_semantic_cognitive"

    if is_retrospective(doc) and not has_event_metadata(doc):
        return "thought leadership", "retrospective_override"

    return base_label, "model_fallback"


def per_class_acc(y_true, y_pred, label):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred, index=y_true.index)
    mask = y_true == label
    if mask.sum() == 0:
        return 0.0
    return (y_pred[mask] == y_true[mask]).mean()


train_df = pd.read_csv("train_80.csv")
test_df = pd.read_csv("test_20.csv")

train_df["Content Pillar"] = train_df["Content Pillar"].apply(normalize_label)
test_df["Content Pillar"] = test_df["Content Pillar"].apply(normalize_label)

train_df.dropna(subset=["Content Pillar"], inplace=True)
test_df.dropna(subset=["Content Pillar"], inplace=True)

train_df["Social Copy"] = train_df["Social Copy"].apply(clean_text)
test_df["Social Copy"] = test_df["Social Copy"].apply(clean_text)

y_train = train_df["Content Pillar"]
y_test = test_df["Content Pillar"]

tfidf = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
X_train = tfidf.fit_transform(train_df["Social Copy"])
X_test = tfidf.transform(test_df["Social Copy"])

grid = GridSearchCV(
    LogisticRegression(max_iter=5000, class_weight="balanced"),
    {"C": [0.5, 1, 3]},
    cv=5,
    n_jobs=-1,
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

base_preds = pd.Series(model.predict(X_test), index=y_test.index)
base_probs = model.predict_proba(X_test)
label_to_idx = {label: i for i, label in enumerate(model.classes_)}

final_preds = []
reasons = []

for i, text in enumerate(test_df["Social Copy"]):
    base_label = base_preds.iloc[i]
    base_prob = base_probs[i, label_to_idx[base_label]]
    label, reason = intent_resolver(text, base_label, base_prob)
    final_preds.append(label)
    reasons.append(reason)

final_preds = pd.Series(final_preds, index=y_test.index)

print("\nBASE ACCURACY:", accuracy_score(y_test, base_preds))
print("FINAL ACCURACY:", accuracy_score(y_test, final_preds))
print("\nEVENTS ACCURACY:", per_class_acc(y_test, final_preds, "events"))
print(
    "THOUGHT LEADERSHIP ACCURACY:",
    per_class_acc(y_test, final_preds, "thought leadership"),
)

plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(y_test, base_preds)
plt.tight_layout()
plt.savefig("confusion_matrix_base.png", dpi=120)

plt.figure(figsize=(12, 10))
ConfusionMatrixDisplay.from_predictions(y_test, final_preds)
plt.tight_layout()
plt.savefig("confusion_matrix_final.png", dpi=120)

validation_df = test_df.copy()
validation_df["True Label"] = y_test.values
validation_df["Base Prediction"] = base_preds.values
validation_df["Final Prediction"] = final_preds.values
validation_df["Resolution Reason"] = reasons

hashtags, nouns, verbs, orgs, dates, locations = [], [], [], [], [], []

for text in validation_df["Social Copy"]:
    hashtags.append(extract_hashtags(text))
    n, v, o, d, l = extract_spacy_features(text)
    nouns.append(n)
    verbs.append(v)
    orgs.append(o)
    dates.append(d)
    locations.append(l)

validation_df["Hashtags"] = hashtags
validation_df["Nouns"] = nouns
validation_df["Verbs"] = verbs
validation_df["Organizations"] = orgs
validation_df["Dates"] = dates
validation_df["Locations"] = locations

validation_df["Final Correct"] = (
    validation_df["Final Prediction"] == validation_df["True Label"]
)
validation_df["Corrected"] = (
    validation_df["Base Prediction"] != validation_df["True Label"]
) & (validation_df["Final Prediction"] == validation_df["True Label"])

validation_df.to_csv("intent_validation_results.csv", index=False)

print("\nSaved: intent_validation_results.csv")
print("Saved: confusion_matrix_base.png")
print("Saved: confusion_matrix_final.png")
print("\nDONE.")
