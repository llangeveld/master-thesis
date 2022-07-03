import langid
import json


def get_json_files(domain: str) -> dict:
    """
    Pulls up JSON-files for the right domain
    :param domain: EMEA, GNOME, or JRC
    :return: Dictionary: {doc: {"en": [list of sentences],
                                "de": [list of sentences]}}
    """
    f = open(f"../data/2_opus/json/{domain}.json")
    data = json.load(f)

    return data


def make_language_files(domain: str, max_docs="max"):
    """
    Pulls up JSON-docs.
    :param domain: Domain whose documents are returned
    :param max_docs: Number of documents to be taken into account, int or "max"
    :return: List of English sentences, list of German sentences
    """
    if type(max_docs) != int and max_docs != "max":
        raise ValueError("max_docs should be either an integer or 'max'.")
    data = get_json_files(domain)
    en_docs = []
    de_docs = []
    for doc, texts in data.items():
        # If max_docs is an integer, pull up that many docs
        if isinstance(max_docs, int):
            if int(doc) == max_docs:
                break
        en_docs.append(texts["en"])
        de_docs.append(texts["de"])

    return en_docs, de_docs


def get_all_texts(domain: str, language: str):
    """
    Finds the three texts for the domain and language specified
    :param domain: EMEA, GNOME, or JRC
    :param language: en or de
    :return: three lists of sentences (train, test, valid)
    """
    train = open(f"../data/1_main_data/tokenized/{domain}/train.{language}")
    test = open(f"../data/1_main_data/tokenized/{domain}/test.{language}")
    valid = open(f"../data/1_main_data/tokenized/{domain}/valid.{language}")

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
    """
    Removes sentence pairs where one or both of the sentences are not the
    correct language or empty.
    :param text_en: English text (list of strings)
    :param text_de: German text (list of strings)
    :return: List of new English sentences and list of new German sentences.
    """
    new_en, new_de = [], []
    for en, de in zip(text_en, text_de):
        if str(en).strip() == "" or str(de).strip() == "":
            continue
        en_lang, _ = langid.classify(en)
        de_lang, _ = langid.classify(de)
        if en_lang == "en" and de_lang == "de":
            new_en.append(en)
            new_de.append(de)

    return new_en, new_de


def postprocess_alignment_file(domain: str) -> dict:
    """
    Postprocesses alignment files for language pairs within a domain.
    :param domain: Self-explanatory
    :return: A dictionary with, for each language pair, a list of tuples with
    the alignments made by fastalign.
    """
    d_alignments = {"en-de": [], "de-en": []}
    for language_pair in ["en-de", "de-en"]:
        alignment = open(f"../data/3_anonymized/fastalign/{domain}_train.{language_pair}.align")
        alignments = [s for s in alignment]
        for s in alignments:
            new_s = []
            l = s.split()
            for pair in l:
                new_pair = [int(x) for x in pair.split("-")]
                new_s.append(tuple(new_pair))
            d_alignments[f"{language_pair}"].append(new_s)
    return d_alignments
