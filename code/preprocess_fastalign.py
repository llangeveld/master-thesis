from util import get_all_texts


def make_new_text(text1: list, text2: list) -> list:
    new_text = []
    for s1, s2 in zip(text1, text2):
        new_text.append(f"{s1.strip()} ||| {s2.strip()}\n")

    return new_text


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {"en": {}, "de": {}}
        for language in ["en", "de"]:
            dic = d[f"{language}"]
            dic["train"], dic["test"], dic["valid"] = get_all_texts(domain, language)
        for phase in ["train", "test", "valid"]:
            en = d["en"][f"{phase}"]
            de = d["de"][f"{phase}"]
            en2de = make_new_text(en, de)
            de2en = make_new_text(de, en)
            with open(f"../data/3_anonymized/fastalign/{domain}_{phase}.en-de", "w") as f:
                for line in en2de:
                    f.write(line)
            with open(f"../data/3_anonymized/fastalign/{domain}_{phase}.de-en", "w") as f:
                for line in de2en:
                    f.write(line)


if __name__ == "__main__":
    main()
