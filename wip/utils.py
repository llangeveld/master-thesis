import langid
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import AnalyzerEngine


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
    train = open(f"/home/lonneke/thesis/local/data/"
                 f"corrected_data/{domain}/train.{language}")
    test = open(f"/home/lonneke/thesis/local/data/"
                f"corrected_data/{domain}/test.{language}")
    valid = open(f"/home/lonneke/thesis/local/data/"
                 f"corrected_data/{domain}/valid.{language}")

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


def create_presidio_engine():
    print("Building engine...")
    # Create configuration containing engine name and models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"},
                    {"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine= provider.create_engine()

    # Pass the created NLP engine and supported_languages to the AnalyzerEngine
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "de"]
    )

    return analyzer
