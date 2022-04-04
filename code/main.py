from util import make_language_files
from anonymize import ner_tfidf
from eval import evaluate_fscore


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}
        d["en"], d["de"] = make_language_files(domain, max_docs=1)
        for language in ["en", "de"]:
            docs = d[f"{language}"]
            new_docs = ner_tfidf(docs, language)
            i = 0
            for doc, new_doc in zip(docs, new_docs):
                print(f"Working on doc {i} in {domain}-{language}")
                evaluate_fscore(doc, new_doc)


if __name__ == "__main__":
    main()
