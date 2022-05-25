from sklearn.feature_extraction.text import TfidfVectorizer
from util import get_all_texts
import pandas as pd
import numpy as np
import random
import spacy

NGRAM_RANGE = (2, 2)
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")


class NER:
    def __init__(self, text, language):
        self.text = text
        self.language = language

    def anonymize(self) -> list:
        """
            Anonymizes text using the SpaCy NER-tagger.
            :return: List of anonymized sentences
            """
        # Create NLP-engine for the right language
        nlp = nlp_en if self.language == "en" else nlp_de
        # Anonymize text sentence by sentence
        anonymized_text = []
        for s in self.text:
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


class TFIDF:
    def __init__(self, docs, language):
        self.docs = docs
        self.language = language

    def calculate_tfidf(self, docs: list) -> list:
        """
        Returns list of TF-IDF results for documents.
        :param docs: list of documents to be calculated
        :return: list of results: doc-num, words, tf-idf score.
        """
        # Calculate TF-IDF
        if self.language == "en":
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

    @staticmethod
    def make_dataframe(results):
        """
        Builds Pandas dataframe from results, increases legibility
        :param results: list of results
        :return: pandas dataframe
        """
        df = pd.DataFrame(results, columns=["Doc", "Word(s)", "TF-IDF"])
        df.sort_values(by=["Doc", "TF-IDF"], inplace=True, ascending=[True, False])

        return df

    def find_most_similar_words(self, word, n=5):
        """
        Finds the n most similar words to input word in the given language
        :param word: Word that will get most similar words as output
        :param n: Number of most similar words
        :return: list of n most similar words to input word
        """
        if self.language == "en":
            sp = nlp_en
        elif self.language == "de":
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

    def get_replace_strings(self, words: list) -> list:
        """
        Finds replacement strings using SpaCy
        :param words: list of words in need of replacement
        :return: a list of lists of strings with replacement words per word
        """
        replace_strings = []
        for word in words:
            syns = [syn for syn in self.find_most_similar_words(word)]
            replace_strings.append(syns)
        return replace_strings

    def change_word(self, words: list, text: list) -> list:
        """
        Replaces given words by random synonyms in a given text.
        :param words: Words that need replacing
        :param text: Text in which words need to be replaced
        :return: List of sentences wherein the words have been replaced
        """
        new_text = text
        for word_set in words:
            words_list = word_set.split(" ")
            replace_strings = self.get_replace_strings(words_list)
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

    @staticmethod
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

    def anonymize(self, show_tfidf=False) -> list:
        """
        Calculates TF-IDF score for a "before"-document and the document in which
        the most common n-grams have been replaced.
        :param show_tfidf: Whether to show table with TF-IDF-scores
        :return: List of documents in which common n-grams have been replaced
        """
        # Calculate TF-IDF-scores for original texts
        results = self.calculate_tfidf(self.docs)
        print("...Done for before anonymization")
        df = self.make_dataframe(results)
        new_texts = []
        # Change words in documents
        for i in range(len(self.docs)):
            words = df[df["Doc"] == i]["Word(s)"].tolist()[:5]
            new_texts.append(self.change_word(words, self.docs[i]))
        if show_tfidf:
            # Show changed TF-IDF-scores
            new_results = self.calculate_tfidf(new_texts)
            new_df = self.make_dataframe(new_results)
            self.show_results(df, new_df)

        return new_texts


def separate_into_documents(text: list, domain: str) -> list:
    """
    Returns text separated into documents based on average number of sentences
    per document for a certain domain.
    :param text: Text to be separated into documents
    :param domain: Domain (needed for avg. number of sentences)
    :return:
    """
    doc_lens = {"EMEA": 152, "GNOME": 166, "JRC": 29}
    docs = []
    i = 0
    new_doc = []
    max_doc_len = doc_lens[domain]
    for sentence in text:
        if i <= max_doc_len:
            new_doc.append(sentence)
            i += 1
        else:
            docs.append(new_doc)
            new_doc = []
            i = 0
    docs.append(new_doc)

    return docs


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        for language in ["en", "de"]:
            print("Loading data...")
            text, _, _ = get_all_texts(domain, language)
            print("Separating into documents...")
            docs = separate_into_documents(text, domain)
            print("Calculating TFIDF...")
            tfidf_anonymizer = TFIDF(docs, language)
            anonymized_tfidf = tfidf_anonymizer.anonymize()
            print("Flattening texts...")
            flat_text = [sentence for doc in anonymized_tfidf for sentence in doc]
            print("Anonymizing NER...")
            ner_anonymizer = NER(flat_text, language)
            anonymized_ner = ner_anonymizer.anonymize()
            print("Done. Saving documents.")
            with open(f"../data/3_anonymized/full/{domain}.{language}", "w") as f:
                for sentence in anonymized_ner:
                    f.write(sentence)


if __name__ == "__main__":
    main()
