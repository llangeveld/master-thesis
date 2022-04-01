import json
from utils import remove_wrong_language


def get_domain_dict(text):
    doc_num = 0
    en = []
    de = []
    docs = {}

    for i, s in enumerate(text):
        s = s.strip()
        if (s.startswith("# de") and en and de) or (i == len(text) - 1):
            docs[f"{doc_num}"] = {"en": en, "de": de}
            doc_num += 1
            en = []
            de = []
        elif s.startswith("(src)"):
            idx = s.find('>')
            de.append(s[idx + 1:])
        elif s.startswith("(trg)"):
            idx = s.find('>')
            en.append(s[idx + 1:])
        else:
            continue

    return docs


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        f = open(f"../data/opus/test/{domain}.out")
        text = [s for s in f]
        docs = get_domain_dict(text)
        docs_corrected = {}
        for i in range(len(docs)):
            doc = docs[str(i)]
            en, de = remove_wrong_language(doc["en"], doc["de"])
            docs_corrected[i] = {"en": en, "de": de}
            print(f"Corrected {domain}-doc {i}")
        with open(f"../data/opus/json/{domain}.json", "w") as outfile:
            json.dump(docs, outfile, indent=2)


if __name__ == "__main__":
    main()
