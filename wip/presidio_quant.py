print("Importing modules...")
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig
import langid
import matplotlib.pyplot as plt
import matplotlib as mp
plt.rcParams.update({'font.size': 14})

NUM_SENT = 12153

def load_files():
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

def count_words(data):
    c = 0
    for s in data:
        c += len(s)

    return c

def make_wordcount(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de):
    count = []
    count.append(count_words(EMEA_en))
    count.append(count_words(EMEA_de))
    count.append(count_words(GNOME_en))
    count.append(count_words(GNOME_de))
    count.append(count_words(JRC_en))
    count.append(count_words(JRC_de))

    return count

def language_ID(text, lang):
    correct = 0
    for s in text:
        c_lang, _ = langid.classify(s)
        if c_lang == lang:
            correct += 1

    prop = correct/len(text)
    return prop

def check_languages(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de):
    print("Checking languages...")
    EMEA_en_l = language_ID(EMEA_en, 'en')
    EMEA_de_l = language_ID(EMEA_de, 'de')
    GNOME_en_l = language_ID(GNOME_en, 'en')
    GNOME_de_l = language_ID(GNOME_de, 'de')
    JRC_en_l = language_ID(JRC_en, 'en')
    JRC_de_l = language_ID(JRC_de, 'de')

    print("  Values for EMEA:")
    print(f"    English: {EMEA_en_l}")
    print(f"    German: {EMEA_de_l}")
    print("  Values for GNOME:")
    print(f"    English: {GNOME_en_l}")
    print(f"    German: {GNOME_de_l}")
    print("  Values for JRC:")
    print(f"    English: {JRC_en_l}")
    print(f"    German: {JRC_de_l}")


def sentence_length(text):
    l = 0
    for s in text:
        tok = s.split(' ')
        l += len(tok)

    avg = l / len(text)
    return avg

def check_sentence_length(EMEA_en, EMEA_de, GNOME_en,
                          GNOME_de, JRC_en, JRC_de):
    print("Checking sentence length...")

    lengths = [sentence_length(EMEA_en), sentence_length(EMEA_de),
               sentence_length(GNOME_en), sentence_length(GNOME_de),
               sentence_length(JRC_en), sentence_length(JRC_de)]

    print("  Values for EMEA:")
    print(f"    English: {lengths[0]}")
    print(f"    German: {lengths[1]}")
    print("  Values for GNOME:")
    print(f"    English: {lengths[2]}")
    print(f"    German: {lengths[3]}")
    print("  Values for JRC:")
    print(f"    English: {lengths[4]}")
    print(f"    German: {lengths[5]}")

    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap.from_list("grey",
                                                            ["lightgrey",
                                                             "dimgrey"])

    x_axis = ["EMEA-en", "EMEA-de", "GNOME-en", "GNOME-de", "JRC-en", "JRC-de"]
    plt.bar(x_axis, lengths, align='center',
            color=color_map(data_normalizer(lengths)))
    plt.title("Sentence length per domain and language")
    plt.xticks(rotation=45)
    plt.ylabel("Avg. sentence length in words")
    plt.tight_layout()
    plt.savefig("lengths.pdf")

    return lengths

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

def analyze_ner(text, analyzer, language):
    results = []
    not_empty = 0
    for s in text:
        r = analyzer.analyze(text=s, language=language)
        if r == []:
            continue
        else:
            not_empty += 1
            results.append(r)

    prop = not_empty/len(text)
    return results, prop

def do_analysis(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de, wordcount):
    analyzer = create_engine()

    print("Doing NER-count analysis...")
    EMEA_en_a, EMEA_en_p = analyze_ner(EMEA_en, analyzer, "en")
    EMEA_de_a, EMEA_de_p = analyze_ner(EMEA_de, analyzer, "de")
    GNOME_en_a,GNOME_en_p = analyze_ner(GNOME_en, analyzer, "en")
    GNOME_de_a, GNOME_de_p = analyze_ner(GNOME_de, analyzer, "de")
    JRC_en_a, JRC_en_p = analyze_ner(JRC_en, analyzer, "en")
    JRC_de_a, JRC_de_p = analyze_ner(JRC_de, analyzer, "de")

    analysis = [EMEA_en_a, EMEA_de_a, GNOME_en_a,
                GNOME_de_a, JRC_en_a, JRC_de_a]

    prop = [EMEA_en_p, EMEA_de_p, GNOME_en_p,
            GNOME_de_p, JRC_en_p, JRC_de_p]

    print(analysis)
    count = [len(l)/4 for l in analysis]

    print("  Values for EMEA:")
    print(f"    English: {count[0]}")
    print(f"    German: {count[1]}")
    print("  Values for GNOME:")
    print(f"    English: {count[2]}")
    print(f"    German: {count[3]}")
    print("  Values for JRC:")
    print(f"    English: {count[4]}")
    print(f"    German: {count[5]}")

    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap.from_list("grey",
                                                            ["lightgrey",
                                                             "dimgrey"])

    per_word = []
    for i, c in enumerate(count):
        per_word.append(c/wordcount[i])

    print("Doing NER-count per-word analysis...")
    print("  Values for EMEA:")
    print(f"    English: {per_word[0]}")
    print(f"    German: {per_word[1]}")
    print("  Values for GNOME:")
    print(f"    English: {per_word[2]}")
    print(f"    German: {per_word[3]}")
    print("  Values for JRC:")
    print(f"    English: {per_word[4]}")
    print(f"    German: {per_word[5]}")

    x_axis = ["EMEA-en", "EMEA-de", "GNOME-en", "GNOME-de", "JRC-en", "JRC-de"]
    plt.bar(x_axis, per_word, align='center',
            color=color_map(data_normalizer(per_word)))
    plt.title("Percentage of words with a NER-tag")
    plt.xticks(rotation=45)
    plt.ylabel("% of words")
    plt.tight_layout()
    # plt.show()
    plt.savefig("ner_per_word.pdf")




def main():
    EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de = load_files()
    # check_languages(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de)
    wordcount = make_wordcount(EMEA_en, EMEA_de, GNOME_en,
                               GNOME_de, JRC_en, JRC_de)
    # lengths = check_sentence_length(EMEA_en, EMEA_de, GNOME_en,
                                    # GNOME_de, JRC_en, JRC_de)
    do_analysis(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de, wordcount)

if __name__ == "__main__":
    main()