import langid
import json


def get_json_files(domain: str):
    f = open(f"/home/lonneke/thesis/local/data/opus/json/{domain}.json")
    data = json.load(f)

    return data


def make_language_files(domain: str, max_docs="max"):
    """
    Pulls up JSON-docs.
    :param domain: Domain whose documents are returned
    :param max_docs: Number of documents to be taken into account, int or "max"
    :return: list, list
    """
    data = get_json_files(domain)
    en_docs = []
    de_docs = []
    for doc, texts in data.items():
        if isinstance(max_docs, int):
            if int(doc) = max_docs:
                break
        en_docs.append(texts["en"])
        de_docs.append(texts["de"])

    return en_docs, de_docs


def get_all_texts(domain: str, language: str):
    """
    Finds the three texts (train, test, valid) for the domain and language
    you specify.
    """
    train = open(f"/home/lonneke/thesis/local/data/"
                 f"retokenized/{domain}/train.{language}")
    test = open(f"/home/lonneke/thesis/local/data/"
                f"retokenized/{domain}/test.{language}")
    valid = open(f"/home/lonneke/thesis/local/data/"
                 f"retokenized/{domain}/valid.{language}")

    train_l = [s for s in train]
    test_l = [s for s in test]
    valid_l = [s for s in valid]

    return train_l, test_l, valid_l


def load_full_files():
    """
    Loads the six full data files, and returns them as lists.
    :return: A dictionary with the six files
            (each file being a list of sentences).
    """
    d = {}
    for domain in ["EMEA", "GNOME", "JRC"]:
        for lang in ["en", "de"]:
            x1, x2, x3 = get_all_texts(domain, lang)
            d[f"{domain}_{lang}"] = x1 + x2 + x3

    return d


def remove_wrong_language(text_en: list, text_de: list):
    new_en, new_de = [], []
    for en, de in zip(text_en, text_de):
        if not en or not de:
            continue
        en_lang, _ = langid.classify(en)
        de_lang, _ = langid.classify(de)
        if en_lang == "en" and de_lang == "de":
            new_en.append(en)
            new_de.append(de)

    return new_en, new_de
