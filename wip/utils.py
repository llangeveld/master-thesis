import langid


def load_full_files():
    """
    Loads the six full data files, and returns them as lists.
    :return: The six full data files (as lists)
    """
    EMEA_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/EMEA/full.en")
    EMEA_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/EMEA/full.de")
    GNOME_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/GNOME/full.en")
    GNOME_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/GNOME/full.de")
    JRC_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/JRC/full.en")
    JRC_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/JRC/full.de")

    EMEA_en = [s for s in EMEA_en_f]
    EMEA_de = [s for s in EMEA_de_f]
    GNOME_en = [s for s in GNOME_en_f]
    GNOME_de = [s for s in GNOME_de_f]
    JRC_en = [s for s in JRC_en_f]
    JRC_de = [s for s in JRC_de_f]

    return EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de


def get_all_texts(domain: str, language: str):
    """
    Finds the three texts (train, test, valid) for the domain and language
    you specify.
    :param domain: The domain for which you want the files (EMEA, GNOME, JRC)
    :type str
    :param language: The language for which you want the files (en, de)
    :type str
    :return: Three lists: train, test, and valid
    """
    train = open(f"/home/lonneke/thesis/data/"
                 f"full_sent_datasets/{domain}/train.{language}")
    test = open(f"/home/lonneke/thesis/data/"
                f"full_sent_datasets/{domain}/test.{language}")
    valid = open(f"/home/lonneke/thesis/data/"
                 f"full_sent_datasets/{domain}/valid.{language}")

    train_l = [s for s in train]
    test_l = [s for s in test]
    valid_l = [s for s in valid]

    return train_l, test_l, valid_l


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
