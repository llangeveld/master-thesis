from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def create_presidio_analyzer():
    """
    Creates Presidio analyzer that can handle English and German.
    :return: AnalyzerEngine-object
    """
    # Configure SpaCy-models
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_lg"},
                   {"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    # Create NLP engine based on configuration
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    # Pass the created NLP engine and supported_languages to the AnalyzerEngine
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine,
                              supported_languages=["en", "de"])
    return analyzer


def create_presidio_anonymizer():
    """
    Creates a Presidio AnonymizerEngine
    :return: AnonymizerEngine-object
    """
    return AnonymizerEngine()


def anonymize_text_presidio(text: list, language: str) -> list:
    """
    Anonymizes text using the Presidio-engine.
    :param text: List of sentences to be anonymized
    :param language: Language of sentences (en or de)
    :return: List of anonymized sentences
    """
    # Create necessary engines
    analyzer = create_presidio_analyzer()
    anonymizer = create_presidio_anonymizer()
    # Anonymize text sentence by sentence
    anonymized_text = []
    for s in text:
        results = analyzer.analyze(text=s, language=language)
        anonymized = anonymizer.anonymize(
            text=s,
            analyzer_results=results
        )
        anonymized_text.append(anonymized)

    return anonymized_text