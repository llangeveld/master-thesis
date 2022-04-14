from util import make_language_files
from anonymize import ner_tfidf
from eval import evaluate_fscore


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {"en": (make_language_files(domain, max_docs=3))[0],
             "de": (make_language_files(domain, max_docs=3))[1]}
        d["en_new"] = ner_tfidf(d["en"], "en")
        d["de_new"] = ner_tfidf(d["de"], "de")
        for language in ["en", "de"]:
            i = 0
            for doc, new_doc in zip(d[f"{language}"], d[f"{language}_new"]):
                print(f"Working on doc {i} in {domain}-{language}")
                evaluate_fscore(doc, new_doc)
                print()
                i += 1


if __name__ == "__main__":
    main()
