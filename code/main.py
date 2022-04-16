from util import make_language_files, get_all_texts
from anonymize import ner_tfidf
from eval import evaluate_fscore, calculate_bleu
import translate

anon = False
transl = True


def anonymization_pipeline(domain: str):
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


def translation_pipeline(domain: str):
    d = {}
    print("Loading translation files...", end="")
    for language in ["en", "de"]:
        d[f"train_{language}"], d[f"test_{language}"], d[f"valid_{language}"] = get_all_texts(
            domain, language)
    print("Translation files loaded.")

    print("Translating English to German...")
    en2de = translate.en2de(d["test_en"])
    print("Translating German to English...")
    de2en = translate.de2en(d["test_de"])

    print("Calculating English BLEU...")
    en2de_bleu = calculate_bleu(en2de, d["test_de"])
    print("Calculated German BLEU...\n")
    de2en_bleu = calculate_bleu(de2en, d["test_en"])
    print(f"English to German BLEU: {en2de_bleu}")
    print(f"German to English BLEU: {de2en_bleu}")


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        if anon:
            anonymization_pipeline(domain)
        if transl:
            translation_pipeline(domain)


if __name__ == "__main__":
    main()
