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

COGNITIVE_TERMS = {
    "insight",
    "insights",
    "takeaways",
    "learnings",
    "perspective",
    "perspectives",
    "explore",
    "discussion",
    "discuss",
    "reflect",
    "unpack",
}

PROMO_VERBS = {"register", "sign", "attend", "join", "rsvp", "enroll", "sponsor"}
HOST_VERBS = {"host", "hosting", "organize", "organizing", "launch", "announce"}
COMPANY_PRONOUNS = {"we", "our", "us"}

TRANSACTIONAL_TERMS = {
    "register",
    "registration",
    "sign up",
    "rsvp",
    "reserve",
    "seat",
    "spot",
    "ticket",
    "link",
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


def has_event_metadata(doc):
    return any(ent.label_ in {"DATE", "TIME", "GPE", "LOC"} for ent in doc.ents)


def is_company_hosted(doc):
    return any(tok.text.lower() in COMPANY_PRONOUNS for tok in doc) and any(
        tok.lemma_.lower() in HOST_VERBS for tok in doc
    )


def has_basic_promo(doc):
    return any(tok.lemma_.lower() in PROMO_VERBS for tok in doc)


def has_transactional_intent(text):
    text = text.lower()
    return any(term in text for term in TRANSACTIONAL_TERMS)


def is_idea_focused(text):
    text = text.lower()
    return any(term in text for term in COGNITIVE_TERMS)


def is_retrospective(doc):
    past_verbs = [
        tok for tok in doc if tok.pos_ == "VERB" and tok.morph.get("Tense") == ["Past"]
    ]
    return len(past_verbs) / max(len(doc), 1) > 0.10


def company_intent_override(text):
    doc = nlp(text)
    if is_company_hosted(doc):
        if (
            has_basic_promo(doc)
            and has_event_metadata(doc)
            and has_transactional_intent(text)
        ):
            return "events", "company_promotional_event"
        if is_idea_focused(text) and not has_transactional_intent(text):
            return "thought leadership", "company_insight_event"
    return None, None


def intent_resolver(text, base_label):
    override, reason = company_intent_override(text)
    if override:
        return override, reason
    doc = nlp(text)
    if (
        has_basic_promo(doc)
        and has_event_metadata(doc)
        and has_transactional_intent(text)
    ):
        return "events", "transactional_event_signal"
    if is_retrospective(doc) and not has_event_metadata(doc):
        return "thought leadership", "retrospective_no_logistics"
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

final_preds = []
reasons = []

for text, base in zip(test_df["Social Copy"], base_preds):
    label, reason = intent_resolver(text, base)
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

hashtags = []
nouns = []
verbs = []
orgs = []
dates = []
locations = []

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
print("Saved: confusion_matrix_final.png")
print("\nDONE.")
