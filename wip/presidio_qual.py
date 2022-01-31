from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import langid
import matplotlib.pyplot as plt
import matplotlib as mp

def load_files():
    EMEA_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/EMEA/valid.en")
    EMEA_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/EMEA/valid.de")
    GNOME_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/GNOME/valid.en")
    GNOME_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/GNOME/valid.de")
    JRC_en_f = open("/home/lonneke/thesis/data/full_sent_datasets/JRC/valid.en")
    JRC_de_f = open("/home/lonneke/thesis/data/full_sent_datasets/JRC/valid.de")

    EMEA_en = [s for s in EMEA_en_f]
    EMEA_de = [s for s in EMEA_de_f]
    GNOME_en = [s for s in GNOME_en_f]
    GNOME_de = [s for s in GNOME_de_f]
    JRC_en = [s for s in JRC_en_f]
    JRC_de = [s for s in JRC_de_f]

    return EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de


def create_engine():
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


def analyze_ner(text, analyzer, anonimizer, language):
    sents = []
    for s in text:
        results = analyzer.analyze(text=s, language=language)
        anonymized = anonimizer.anonymize(
            text = s,
            analyzer_results=results
        )
        sents.append(anonymized.text)
    return sents


def do_analysis(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de):
    analyzer = create_engine()
    anonimizer = AnonymizerEngine()

    print("Doing NER-count analysis...")
    EMEA_en_a = analyze_ner(EMEA_en, analyzer, anonimizer, "en")
    EMEA_de_a = analyze_ner(EMEA_de, analyzer, anonimizer, "de")
    GNOME_en_a = analyze_ner(GNOME_en, analyzer, anonimizer, "en")
    GNOME_de_a = analyze_ner(GNOME_de, analyzer, anonimizer, "de")
    JRC_en_a = analyze_ner(JRC_en, analyzer, anonimizer, "en")
    JRC_de_a = analyze_ner(JRC_de, analyzer, anonimizer, "de")

    EMEA_en_f = open("/home/lonneke/thesis/data/presidio/EMEA.en", "w")
    EMEA_de_f = open("/home/lonneke/thesis/data/presidio/EMEA.de", "w")
    GNOME_en_f = open("/home/lonneke/thesis/data/presidio/GNOME.en", "w")
    GNOME_de_f = open("/home/lonneke/thesis/data/presidio/GNOME.de", "w")
    JRC_en_f = open("/home/lonneke/thesis/data/presidio/JRC.en", "w")
    JRC_de_f = open("/home/lonneke/thesis/data/presidio/JRC.de", "w")

    for i in range(len(EMEA_en)-1):
        EMEA_en_f.write(EMEA_en[i] + EMEA_en_a[i] + "\n")
        EMEA_de_f.write(EMEA_de[i] + EMEA_de_a[i] + "\n")
        GNOME_en_f.write(GNOME_en[i] + GNOME_en_a[i] + "\n")
        GNOME_de_f.write(GNOME_de[i] + GNOME_de_a[i] + "\n")
        JRC_en_f.write(JRC_en[i] + JRC_en_a[i] + "\n\n")
        JRC_de_f.write(JRC_de[i] + JRC_de_a[i] + "\n\n")


def main():
    EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de = load_files()
    do_analysis(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de)


if __name__ == "__main__":
    main()