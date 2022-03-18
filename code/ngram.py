from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_json_files
import pandas as pd
from nltk.corpus import wordnet as wn
import random

MAX_DOCS = 10
NGRAM_RANGE = (2,2)

def make_language_files(domain: str):
    data = get_json_files(domain)
    en_docs = []
    de_docs = []
    for doc, texts in data.items():
        if int(doc) > MAX_DOCS:
            break
        en_docs.append(" ".join(texts["en"]))
        de_docs.append(" ".join(texts["de"]))

    return en_docs, de_docs


def get_results(docs: list, language: str):
    if language == "en":
        tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE, stop_words='english')
    else:
        tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE)

    response = tfidf.fit_transform(docs)
    terms = tfidf.get_feature_names()

    results = []
    for i, col in enumerate(response.nonzero()[1]):
        results.append([response.nonzero()[0][i], terms[col], response[0, col]])

    return results


def make_dataframe(results):
    df = pd.DataFrame(results, columns=["Doc", "Word(s)", "TF-IDF"])
    df.sort_values(by=["Doc", "TF-IDF"], inplace=True, ascending=[True, False])

    return df


def change_word(df, docs):
    for i in range(5):
        doc, words, tfidf = df.iloc[i]
        text = docs[i]
        replace_string = ""
        for word in words.split(" "):
            syns = []
            for syn in wn.synsets(word):
                for i in syn.lemmas():
                    if str(i.name()) != word:
                        syns.append(i.name())
            if syns:
                replace_string = replace_string + " " + syns[random.randrange(0, len(syns))]
            else:
                replace_string = replace_string + " " + word

        print(text.replace(words, f"\n\n{replace_string}\n\n"))

def show_results(df):
    docs = set(df["Doc"])
    for doc in docs:
        df2 = df[df["Doc"] == doc]
        print(df2.head())

def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}
        d["en"], d["de"] = make_language_files(domain)
        for language in ["en"]:
            results = get_results(d[f"{language}"], language)
            df = make_dataframe(results)
            change_word(df, d[f"{language}"])

if __name__ == "__main__":
    main()