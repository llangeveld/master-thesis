import spacy
from utils import get_all_texts
from sacremoses import MosesTokenizer, MosesDetokenizer


print("Loading language models...")
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")
replace_all = ["EVENT", "FAC", "GPE", "LANGUAGE", "LOC", "NORP", "ORG",
               "PERSON", "PRODUCT", "WORK_OF_ART"]
mt_en = MosesTokenizer('en')
mt_de = MosesTokenizer('de')
md_en = MosesDetokenizer('en')
md_de = MosesDetokenizer('de')


def do_analysis(domain: str, language: str):
    print("Loading files...")
    d = {}
    d["train"], d["test"], d["valid"] = get_all_texts(domain, language)

    nlp = nlp_en if language == "en" else nlp_de
    mt = mt_en if language == "en" else mt_de
    md = md_en if language == "en" else md_de

    for phase in ["valid"]:
        print(f"Result for {domain} - {phase} - {language}")

        for s in d[f"{phase}"]:
            toks = mt.tokenize(s)
            detoks = md.detokenize(toks)
            doc = nlp(detoks.strip())
            len_diff = 0
            s_new = detoks
            for ent in doc.ents:
                start = ent.start_char + len_diff
                s_new = s_new[:start] + s_new[start:].replace(ent.text,
                                                      f"<{ent.label_}>", 1)
                len_diff += len(ent.label_) + 2 - len(ent.text)
            print(detoks)
            print(s_new + "\n")


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        for language in ["en", "de"]:
            do_analysis(domain, language)


if __name__ == "__main__":
    main()
