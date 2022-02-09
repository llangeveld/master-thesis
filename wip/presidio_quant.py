import langid
import matplotlib.pyplot as plt
import matplotlib as mp
from utils import load_full_files, create_presidio_engine
from sacremoses import MosesTokenizer


plt.rcParams.update({'font.size': 14})
mt_en = MosesTokenizer('en')
mt_de = MosesTokenizer('de')


def count_tokens(data: list, language: str) -> int:
    c = 0
    if language == "en":
        mt = mt_en
    else:
        mt = mt_de
    for s in data:
        s = s.rstrip("\n")
        tokens = mt.tokenize(s)
        c += len(tokens)
    return c


def make_wordcount(d: dict) -> list:
    count = [count_tokens(v, k[-2:]) for k, v in d.items()]
    return count


def language_id(text, lang):
    correct = 0
    for s in text:
        c_lang, _ = langid.classify(s)
        if c_lang == lang:
            correct += 1

    prop = correct/len(text)
    return prop


def check_languages(d):
    print("Checking languages...")
    languages = {}
    for k, v in d.items():
        languages[k] = language_id(v, k[-2:])

    for k, v in languages.items():
        print(f"  Results for {k}: {v}")


def s_length_analysis(text: list, language: str) -> dict:
    if language == "en":
        mt = mt_en
    else:
        mt = mt_de
    lengths = []
    l = 0
    for s in text:
        s = s.rstrip("\n")
        tok = mt.tokenize(s)
        lengths.append(len(tok))
        l += len(tok)

    d = {"avg": l/len(text), "min": min(lengths), "max": max(lengths)}
    return d


def check_sentence_length(d: dict):
    print("Checking sentence length...")

    lengths = {}
    for k, v in d.items():
        lengths[k] = s_length_analysis(v, k[-2:])

    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap.from_list("grey",
                                                            ["lightgrey",
                                                             "dimgrey"])

    x_axis = [k for k in d.keys()]
    plt.bar(x_axis, lengths, align='center',
            color=color_map(data_normalizer(lengths)))
    plt.title("Sentence length per domain and language")
    plt.xticks(rotation=45)
    plt.ylabel("Avg. sentence length in words")
    plt.tight_layout()
    plt.savefig("lengths.pdf")

    return lengths


def analyze_ner(text, analyzer, language):
    results = []
    not_empty = 0
    for s in text:
        r = analyzer.analyze(text=s, language=language)
        if not r:
            not_empty += 1
            results.append(r)
        else:
            continue
    return results


def do_analysis(d: dict, wordcount: list):
    analyzer = create_presidio_engine()

    print("Doing NER-count analysis...")
    count = {}
    i = 0
    per_word = {}
    for k, v in d.items():
        analysis = analyze_ner(v, analyzer, k[-2:])
        c = len(analysis)/4
        count[k] = c
        print(f"  Results for {k}: {c}")
        per_word[k] = c/wordcount[i]
        i += 1

    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap.from_list("grey",
                                                            ["lightgrey",
                                                             "dimgrey"])

    print("Doing NER-count per-word analysis...")
    for k, v in per_word.items():
        print(f"  Results for {k}: {v}")

    x_axis = [k for k in d.keys()]
    plt.bar(x_axis, per_word, align='center',
            color=color_map(data_normalizer(per_word)))
    plt.title("Percentage of words with a NER-tag")
    plt.xticks(rotation=45)
    plt.ylabel("% of words")
    plt.tight_layout()
    # plt.show()
    plt.savefig("ner_per_word.pdf")


def main():
    d = load_full_files()
    # check_languages(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de)
    wordcount = make_wordcount(d)
    # lengths = check_sentence_length(EMEA_en, EMEA_de, GNOME_en,
                                    # GNOME_de, JRC_en, JRC_de)
    do_analysis(d, wordcount)


if __name__ == "__main__":
    main()
