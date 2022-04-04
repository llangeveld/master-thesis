from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import random
import spacy

NGRAM_RANGE = (2, 2)
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")


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


def anonymize_text_spacy(text: list, language: str) -> list:
    """
    Anonymizes text using the SpaCy NER-tagger.
    :param text: List of sentences to be anonymized
    :param language: Language of sentences (en or de)
    :return: List of anonymized sentences
    """
    # Create NLP-engine for the right language
    nlp = nlp_en if language == "en" else nlp_de
    # Anonymize text sentence by sentence
    anonymized_text = []
    for s in text:
        doc = nlp(s)
        len_diff = 0  # Length difference to calculate string replacement starting point
        s_new = s
        for ent in doc.ents:
            start = ent.start_char + len_diff
            # Update string based on starting point
            s_new = s_new[:start] + s_new[start:].replace(ent.text,
                                                          f"<{ent.label_}>", 1)
            # Calculate new starting point in string
            len_diff += len(ent.label_) + 2 - len(ent.text)
        anonymized_text.append(s_new)

    return anonymized_text


def anonymize_text_ner(text: list, language: str, method="spacy") -> list:
    if method == "spacy":
        return anonymize_text_spacy(text, language)
    elif method == "presidio":
        return anonymize_text_presidio(text, language)
    else:
        raise ValueError("NER-anonymization method must be either spacy or"
                         "Presidio.")


def get_results(docs, language: str) -> list:
    """
    Returns list of TF-IDF results for documents.
    :param docs: list of documents to be calculated
    :param language: language in which the documents are written
    :return: list of results: doc-num, words, tf-idf score.
    """
    # Calculate TF-IDF
    if language == "en":
        tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE, stop_words='english')
    else:
        tfidf = TfidfVectorizer(ngram_range=NGRAM_RANGE)
    texts = [" ".join(doc) for doc in docs]
    response = tfidf.fit_transform(texts)
    terms = tfidf.get_feature_names()

    # Put TF-IDF into readable format
    results = []
    for i, col in enumerate(response.nonzero()[1]):
        # [document number, words, TF-IDF-score]
        results.append([response.nonzero()[0][i], terms[col], response[0, col]])

    return results


def make_dataframe(results):
    """
    Builds Pandas dataframe from results, increases legibility
    :param results: list of results
    :return: pandas dataframe
    """
    df = pd.DataFrame(results, columns=["Doc", "Word(s)", "TF-IDF"])
    df.sort_values(by=["Doc", "TF-IDF"], inplace=True, ascending=[True, False])

    return df


def get_replace_strings_wn(words: list) -> list:
    """
    Finds replacement strings using WordNet (only available for English)
    :param words: list of words needing to be replaced
    :return: a list of lists of strings with replacement words per word
    """
    replace_strings = []
    for word in words:
        syns = []
        # Find synonyms for word
        for syn in wn.synsets(word):
            for i in syn.lemmas():
                if str(i.name()) != word:
                    syns.append(i.name())
        # Build possible replacement list
        if syns:
            replace_strings.append(syns)
        else:
            replace_strings.append(word)

    return replace_strings


def most_similar_spacy(word, language, n=5):
    """
    Finds the n most similar words to input word in the given language
    :param word: Word that will get most similar words as output
    :param language: en or de
    :param n: Number of most similar words
    :return: list of n most similar words to input word
    """
    if language == "en":
        sp = nlp_en
    elif language == "de":
        sp = nlp_de
    else:
        raise ValueError("Argument 'language' must be 'en' or 'de'.")
    word = sp.vocab[str(word)]
    queries = [
        w for w in word.vocab
        if w.is_lower == word.is_lower and w.prob >= -30 and np.count_nonzero(w.vector)
    ]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    values = [
        w.lower_ for w in by_similarity[:n + 1]
        if w.lower_ != word.lower_
    ]
    if values:
        return values
    else:
        return [word.lower_]


def get_replace_strings_spacy(words: list, language: str) -> list:
    """
    Finds replacement strings using SpaCy
    :param words: list of words in need of replacement
    :param language: en or de
    :return: a list of lists of strings with replacement words per word
    """
    replace_strings = []
    for word in words:
        syns = [syn for syn in most_similar_spacy(word, language)]
        replace_strings.append(syns)
    return replace_strings


def change_word(words: list, text: list, language: str,
                method="wordnet") -> list:
    """
    Replaces given words by random synonyms in a given text.
    :param words: Words that need replacing
    :param text: Text in which words need to be replaced
    :param language: en or de
    :param method: wordnet or spacy
    :return: List of sentences wherein the words have been replaced
    """
    new_text = text
    for word_set in words:
        words_list = word_set.split(" ")
        if method == "wordnet" and language == "en":
            replace_strings = get_replace_strings_wn(words_list)
        elif method == "spacy":
            replace_strings = get_replace_strings_spacy(words_list, language)
        elif method == "wordnet" and language != "en":
            raise ValueError("If method is wordnet, language must be English")
        else:
            raise ValueError("Method must be either wordnet or spacy"
                             " (default = wordnet)")
        # Replacing old words by new words
        for idx, sentence in enumerate(new_text):
            new_sentence = sentence
            for j in range(sentence.count(word_set)):
                replace_string = ""
                # Build replacement string
                for i in range(len(words_list)):
                    replace_string += str(random.choice(replace_strings[i]))
                    replace_string += " "
                new_sentence = new_sentence.replace(word_set, replace_string.rstrip(), 1)
            # Replacing old sentence by new sentence
            new_text[idx] = new_sentence

    return new_text


def show_results(df, df2):
    """
    Shows the top 5 TF-IDF scores from two different dataframes.
    :param df: Pandas dataframe
    :param df2: Pandas dataframe
    :return: None
    """
    docs = set(df["Doc"])
    for doc in docs:
        new_df1 = df[df["Doc"] == doc]
        new_df2 = df2[df2["Doc"] == doc]
        print(new_df1.head())
        print(new_df2.head())
        print("----------------------------------------------")


def calculate_tfidf(docs: list, language: str, method: str,
                    show_tfidf=False) -> list:
    """
    Calculates TF-IDF score for a "before"-document and the document in which
    the most common n-grams have been replaced.
    :param show_tfidf: Whether or not to show table with TF-IDF-scores
    :param docs: list of documents (document = list of strings)
    :param language: en or de
    :param method: wordnet or spacy
    :return: List of documents in which common n-grams have been replaced
    """
    # Calculate TF-IDF-scores for original texts
    results = get_results(docs, language)
    df = make_dataframe(results)
    new_texts = []
    # Change words in documents
    for i in range(len(docs)):
        words = df[df["Doc"] == i]["Word(s)"].tolist()[:5]
        new_texts.append(change_word(words, docs[i], language, method))
    # Calculate TF-IDF-scores for new texts
    new_results = get_results(new_texts, language)
    new_df = make_dataframe(new_results)
    if show_tfidf:
        # Show changed TF-IDF-scores
        show_results(df, new_df)

    return new_texts


def ner_tfidf(docs: list, language: str, tfidf_method="spacy",
              ner_method="spacy"):
    """
    Combined NER & TF-IDF-anonymization.
    :param docs: List of documents to be anonymized
    :param language: Language (de or en)
    :param tfidf_method: spacy or wordnet
    :param ner_method: spacy or presidio
    :return: List of anonymized documents
    """
    after_tfidf = calculate_tfidf(docs, language, tfidf_method)
    after_spacy = [anonymize_text_ner(doc, language, ner_method)
                   for doc in after_tfidf]

    return after_spacy
