from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import wordnet as wn
import random

NGRAM_RANGE = (2, 2)


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


def change_word(words: list, text: list) -> list:
    """
    Changes list of words in the text to new texts
    :param words: List of words/phrases to be replaced
    :param text: Document in which words need to be replaced
    :return: Document with replaced words
    """
    new_text = text
    for word_set in words:
        words_list = word_set.split(" ")
        replace_strings = []
        # Go over each word in the n-gram
        for idx, word in enumerate(words_list):
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


def calculate_tfidf(docs: list, language: str) -> list:
    # Calculate TF-IDF-scores for original texts
    results = get_results(docs, language)
    df = make_dataframe(results)
    new_texts = []
    # Change words in documents
    for i in range(len(docs)):
        words = df[df["Doc"] == i]["Word(s)"].tolist()[:5]
        new_texts.append(change_word(words, docs[i]))
    # Calculate TF-IDF-scores for new texts
    new_results = get_results(new_texts, language)
    new_df = make_dataframe(new_results)
    # Show changed TF-IDF-scores
    show_results(df, new_df)

    return new_texts
