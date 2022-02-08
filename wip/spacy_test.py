import spacy
from utils import get_all_texts, remove_wrong_language

print("Loading language models...")
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")

def make_string(text: list):
    string = ""
    for s in text:
        s = s.strip() + " "
        string = string + s

    return string

def do_analysis(domain: str):
    print("Loading files...")
    train_en, test_en, valid_en = get_all_texts(domain, "en")
    train_de, test_de, valid_de = get_all_texts(domain, "de")

    print("Removing wrong languages...")
    d = dict()
    d["train_en"], d["train_de"] = remove_wrong_language(train_en, train_de)
    d["test_en"], d["test_de"] = remove_wrong_language(test_en, test_de)
    d["valid_en"], d["valid_de"] = remove_wrong_language(valid_en, valid_de)

    with open(f'out-{domain}.txt', 'w') as f:
        for phase in ["valid"]:
            for lang in ["en", "de"]:
                print(f"Result for {domain} - {phase} - {lang}", file=f)
                text = d[f"{phase}_{lang}"]
                nlp = nlp_en if lang == "en" else nlp_de
                for s in text:
                    doc = nlp(s.strip())
                    s_new = s
                    print(s, file=f)
                    for ent in doc.ents:
                        placeholder = "X" * (ent.end_char-ent.start_char)
                        s_new = s_new[:ent.start_char] + placeholder + s_new[ent.end_char:]
                        print(ent.label_, file=f)
                    print(s_new, file=f)
                    print("---------------", file=f)
            print("\n\n-------------------------------------------------\n\n", file=f)
                    #for ent in doc.ents:
                    #    print(f"{ent.text} | {ent.label_}")


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        do_analysis(domain)

if __name__ == "__main__":
    main()