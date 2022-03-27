from utils import make_language_files
from ngram import calculate_tfidf
from eval import evaluate_bleu


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}
        d["en"], d["de"] = make_language_files(domain, max_docs=25)
        for language in ["en"]:
            docs = d[f"{language}"]
            new_docs = calculate_tfidf(docs, language)

            for doc, new_doc in zip(docs, new_docs):
                evaluate_bleu(doc, new_doc)


if __name__ == "__main__":
    main()
