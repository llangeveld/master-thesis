from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy

nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")


def create_presidio_analyzer():
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"},
                   {"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Pass the created NLP engine and supported_languages to the AnalyzerEngine
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "de"]
    )

    return analyzer


def create_presidio_anonymizer():
    return AnonymizerEngine()


def anonymize_text_spacy(text: list, language: str) -> list:
    nlp = nlp_en if language == "en" else nlp_de

    anonymized_text = []
    for s in text:
        doc = nlp(s)
        len_diff = 0
        s_new = s
        for ent in doc.ents:
            start = ent.start_char + len_diff
            s_new = s_new[:start] + s_new[start:].replace(ent.text,
                                                          f"<{ent.label_}>", 1)
            len_diff += len(ent.label_) + 2 - len(ent.text)

        anonymized_text.append(s_new)

    return anonymized_text


def anonymize_text_presidio(text: list, language: str) -> list:
    analyzer = create_presidio_analyzer()
    anonymizer = create_presidio_anonymizer()
    anonymized_text = []
    for s in text:
        results = analyzer.analyze(text=s, language=language)
        anonymized = anonymizer.anonymize(
            text=s,
            analyzer_results=results
        )
        anonymized_text.append(anonymized)

    return anonymized_text
