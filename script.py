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

CONTROL_HEX_PATTERN = re.compile(r"x00[0-9a-f]{2}", re.IGNORECASE)
TM_NOISE_PATTERN = re.compile(r"\b(tm|tms|tmtm|tmz|tmy|tma)\b", re.IGNORECASE)

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}\b",
    r"\b\d+\s+(?:million|billion|thousand)\b",
    r"\b\d{1,2}:\d{2}\b",
    r"\b\d{1,2}\s*(?:am|pm)\b",
]

DATE_REGEX = re.compile("|".join(DATE_PATTERNS), re.IGNORECASE)
ISOLATED_NUMBER_PATTERN = re.compile(r"\b\d\b")
REPEATED_NUMBER_PATTERN = re.compile(r"\b(\d)(\s+\1){2,}\b")
BROKEN_ALPHA_NUM_PATTERN = re.compile(r"\b(?:[a-z]\s*\d|\d\s*[a-z])\b", re.IGNORECASE)


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"http\S+|www\S+|lnkd\.in\S+", " ", s)
    s = CONTROL_HEX_PATTERN.sub(" ", s)
    s = TM_NOISE_PATTERN.sub(" ", s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    protected = {}

    def protect(match):
        key = f"__NUM_{len(protected)}__"
        protected[key] = match.group(0)
        return key

    s = DATE_REGEX.sub(protect, s)
    s = REPEATED_NUMBER_PATTERN.sub(" ", s)
    s = ISOLATED_NUMBER_PATTERN.sub(" ", s)
    s = BROKEN_ALPHA_NUM_PATTERN.sub(" ", s)
    s = re.sub(r"[^a-zA-Z0-9\s,.!?#]", " ", s)

    for k, v in protected.items():
        s = s.replace(k, v)

    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


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


def extract_verb_tenses(text):
    doc = nlp(text)
    past = sorted(
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ == "VERB" and "Tense=Past" in tok.morph
    )
    present = sorted(
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ == "VERB" and "Tense=Pres" in tok.morph
    )
    return past, present


def has_event_metadata(doc):
    return any(ent.label_ in {"DATE", "TIME", "GPE", "LOC"} for ent in doc.ents)


def is_company_hosted(doc, company_name):
    speaker_terms = {"we", "our", "us"}
    if company_name and str(company_name).strip():
        speaker_terms.add(str(company_name).lower())
    speaker = any(tok.lower_ in speaker_terms for tok in doc)
    present_action = any(
        tok.pos_ == "VERB" and "Tense=Pres" in tok.morph for tok in doc
    )
    has_org = any(ent.label_ == "ORG" for ent in doc.ents)
    return speaker and present_action and has_org


def has_basic_promo(doc):
    imperative = any(tok.pos_ == "VERB" and "Mood=Imp" in tok.morph for tok in doc)
    verb_density = sum(tok.pos_ == "VERB" for tok in doc) / max(1, len(doc))
    return imperative or verb_density > 0.18


def has_transactional_intent(text):
    doc = nlp(text)
    action_root = any(sent.root.pos_ == "VERB" for sent in doc.sents)
    has_object = any(tok.dep_ in {"dobj", "pobj"} for tok in doc)
    has_grounding = any(
        ent.label_ in {"DATE", "TIME", "GPE", "LOC"} for ent in doc.ents
    )
    return action_root and has_object and has_grounding


def is_idea_focused(text):
    doc = nlp(text)
    abstract_terms = sum(tok.pos_ == "NOUN" and tok.ent_type_ == "" for tok in doc)
    promotional_verbs = sum(
        tok.lemma_.lower() in {"buy", "register", "join", "sign"} for tok in doc
    )
    return abstract_terms > 4 and promotional_verbs == 0


def company_intent_override(text, company_name):
    doc = nlp(text)
    if is_company_hosted(doc, company_name):
        if (
            has_basic_promo(doc)
            and has_event_metadata(doc)
            and has_transactional_intent(text)
        ):
            return "events", "company_promotional_event"
        if is_idea_focused(text) and not has_transactional_intent(text):
            return "thought leadership", "company_insight_event"
    return None, None


def intent_resolver(text, base_label, company_name):
    if base_label == "thought leadership" and is_idea_focused(text):
        return "thought leadership", "idea_dominant"
    override, reason = company_intent_override(text, company_name)
    if override:
        return override, reason
    if has_transactional_intent(text):
        return "events", "true_event_signal"
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

train_df.drop_duplicates(subset=["Social Copy"], inplace=True)
test_df.drop_duplicates(subset=["Social Copy"], inplace=True)

train_df = train_df[train_df["Social Copy"].str.len() > 10]
test_df = test_df[test_df["Social Copy"].str.len() > 10]

test_df["Company"] = test_df["Company"].fillna("").astype(str).str.strip().str.lower()

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

ALLOWED_SECOND_LAYER = {"events", "thought leadership"}

final_preds = []
reasons = []

for text, base, company in zip(test_df["Social Copy"], base_preds, test_df["Company"]):
    if base in ALLOWED_SECOND_LAYER:
        label, reason = intent_resolver(text, base, company)
    else:
        label, reason = base, "base_only"
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
past_verbs, present_verbs = [], []

for text in validation_df["Social Copy"]:
    hashtags.append(extract_hashtags(text))
    n, v, o, d, l = extract_spacy_features(text)
    p, pr = extract_verb_tenses(text)
    nouns.append(n)
    verbs.append(v)
    orgs.append(o)
    dates.append(d)
    locations.append(l)
    past_verbs.append(p)
    present_verbs.append(pr)

validation_df["Hashtags"] = hashtags
validation_df["Nouns"] = nouns
validation_df["Verbs"] = verbs
validation_df["Past Tense Verbs"] = past_verbs
validation_df["Present Tense Verbs"] = present_verbs
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
