import langid
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from matplotlib.colors import Normalize
from utils import load_full_files, create_presidio_engine
from sacremoses import MosesTokenizer
import pandas as pd


plt.rcParams.update({'font.size': 14})
mt_en = MosesTokenizer('en')
mt_de = MosesTokenizer('de')
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")


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


def sl_analysis(s: str, language: str) -> int:
    if language == "en":
        mt = mt_en
    else:
        mt = mt_de
    tok = mt.tokenize(s.rstrip("\n"))
    return len(tok)


def sl_dataframe(d: dict):
    lengths = []
    for k, v in d.items():
        for s in v:
            s_length = sl_analysis(s, k[-2:])
            lengths.append([k, s_length])

    df = pd.DataFrame(lengths, columns=["Text", "Length"])
    return df


def check_sentence_length(d: dict):
    print("Checking sentence length...")

    df = sl_dataframe(d)
    norm = Normalize(vmin=0, vmax=100)
    colors = [plt.cm.Greys(norm(c)) for c in df["Length"]]
    sns.set_style("whitegrid")
    sns.boxplot(x=df["Text"], y=df["Length"],
                palette=colors, showfliers=False)
    plt.xticks(rotation=45)
    plt.xlabel(None)
    plt.tight_layout()
    plt.savefig("../figures/lengths_boxplot.pdf")


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


def analyze_ner_presidio(text, analyzer, language):
    results = 0
    for s in text:
        r = analyzer.analyze(text=s, language=language)
        results += len(r)
    return results


def analyze_ner_spacy(text: list, language: str) -> int:
    results = 0
    nlp = nlp_en if language == "en" else nlp_de
    for s in text:
        doc = nlp(s.rstrip("\n"))
        results += len(doc.ents)
    return results


def do_analysis(d: dict, wordcount: list, tagger="Presidio"):
    if tagger == "Presidio":
        analyzer = create_presidio_engine()

    print("Doing NER-count analysis...")
    i = 0
    per_word = {}
    for k, v in d.items():
        if tagger == "Presidio":
            analysis = analyze_ner_presidio(v, analyzer, k[-2:])
        else:
            analysis = analyze_ner_spacy(v, k[-2:])
        print(f"  Results for {k}: {analysis}")
        per_word[k] = analysis/wordcount[i]
        i += 1

    print("Doing NER-count per-word analysis...")
    for k, v in per_word.items():
        print(f"  Results for {k}: {v}")

    x_axis = [k for k in d.keys()]
    y_axis = [v for v in per_word.values()]
    if tagger == "Presidio":
        norm = Normalize(vmin=0, vmax=0.025)
    else:
        norm = Normalize(vmin=0.012, vmax=0.07)
    colors = [plt.cm.Greys(norm(c)) for c in y_axis]
    sns.barplot(x=x_axis, y=y_axis, palette=colors)
    plt.title("Percentage of words with a NER-tag")
    plt.xticks(rotation=45)
    plt.ylabel("% of words")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"../figures/ner_per_word_{tagger}.pdf")


def main():
    d = load_full_files()
    # check_languages(EMEA_en, EMEA_de, GNOME_en, GNOME_de, JRC_en, JRC_de)
    # check_sentence_length(d)
    wordcount = make_wordcount(d)
    # do_analysis_presidio(d, wordcount)
    do_analysis(d, wordcount, "Spacy")


if __name__ == "__main__":
    main()
