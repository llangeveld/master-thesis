from utils import make_language_files
from ngram import calculate_tfidf
from eval import evaluate_fscore


def main():
    for domain in ["EMEA"]:
        d = {}
        d["en"], d["de"] = make_language_files(domain, max_docs=2)
        for language in ["en", "de"]:
            docs = d[f"{language}"]
            new_docs = calculate_tfidf(docs, language, "spacy")
            for doc, new_doc in zip(docs, new_docs):
                # evaluate_bleu(doc, new_doc)
                evaluate_fscore(doc, new_doc)


if __name__ == "__main__":
    main()
