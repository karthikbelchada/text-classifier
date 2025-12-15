import html
import re
import unicodedata

import pandas as pd
import spacy

nlp = spacy.load("en_core_web_lg")

input_csv = "test_20.csv"
output_csv = "test_20_results.csv"

df = pd.read_csv(input_csv)


def clean_social_copy(text):
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities(text):
    doc = nlp(clean_social_copy(text))

    dates = []
    persons = []
    companies = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "PERSON":
            persons.append(ent.text)
        elif ent.label_ == "ORG":
            companies.append(ent.text)
        elif ent.label_ in {"GPE", "LOC"}:
            locations.append(ent.text)

    return pd.Series(
        {
            "dates": ", ".join(dict.fromkeys(dates)),
            "persons": ", ".join(dict.fromkeys(persons)),
            "companies": ", ".join(dict.fromkeys(companies)),
            "locations": ", ".join(dict.fromkeys(locations)),
        }
    )


entities_df = df["Social Copy"].apply(extract_entities)

final_df = pd.concat([df[["Social Copy", "Company"]], entities_df], axis=1)

final_df.to_csv(output_csv, index=False)

print(f"Saved result CSV to {output_csv}")
