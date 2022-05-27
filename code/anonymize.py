from sklearn.feature_extraction.text import TfidfVectorizer
from util import get_all_texts, postprocess_alignment_file
import pandas as pd
import numpy as np
import random
import spacy

NGRAM_RANGE = (2, 2)
tf = False
do_ner = True
qualitative = False
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")


class NER:
    def __init__(self, src_text, trg_text, src_lan, align):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src_lan = src_lan
        self.trg_lan = "en" if self.src_lan == "de" else "de"
        self.align = align

    def replace_in_trg(self, idx: int, indices: list, ent_label: str):
        alignment = self.align[idx]
        replace_ids = find_replace_idx(alignment, indices, flat=True)
        i_start = replace_ids[0]
        sentence = self.trg_text[idx]
        s_new = sentence.split()
        for i in replace_ids:
            if i == i_start:
                s_new[i] = f"<{ent_label}>"
            else:
                s_new[i] = ""
        s_string = " ".join([w for w in s_new if w != ""])
        return s_string

    def anonymize(self) -> list:
        """
        Anonymizes text using the SpaCy NER-tagger.
        :return: List of anonymized sentences
        """
        # Create NLP-engine for the right language
        nlp = nlp_en if self.src_lan == "en" else nlp_de
        # Anonymize text sentence by sentence
        anonymized_text_src = []
        anonymized_text_trg = []
        for idx, s in enumerate(self.src_text):
            doc = nlp(s)
            s_list = s.split()
            s_startchars = []
            new_trg = self.trg_text[idx]
            c = 0
            for i in range(len(s_list) - 1):
                s_startchars.append(c)
                c += len(s_list[i]) + 1
            s_new = s_list.copy()
            indices = []
            for ent in doc.ents:
                i = s_startchars.index(ent.start_char)
                i_start = i
                ent_list = ent.text.split()
                for _ in ent_list:
                    indices.append(i)
                    if i == i_start:
                        s_new[i] = f"<{ent.label_}>"
                    else:
                        s_new[i] = ""
                    i += 1
                new_trg = self.replace_in_trg(idx, indices, ent.label_)
            s_string = " ".join([w for w in s_new if w != ""])
            anonymized_text_src.append(s_string)
            anonymized_text_trg.append(new_trg)
        return anonymized_text_src


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

    def find_most_similar_words(self, word, n=4):
        """
        Finds the n most similar words to input word in the given language
        :param word: Word that will get most similar words as output
        :param n: Number of most similar words
        :return: list of n most similar words to input word
        """
        if self.src_lan == "en":
            sp = nlp_en
        elif self.src_lan == "de":
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
            syns.append(word)
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
        df = self.make_dataframe(results)
        new_texts = []
        # Change words in documents
        for i in range(len(self.docs)):
            words = df[df["Doc"] == i]["Word(s)"].tolist()[:5]
            new_text = self.change_word(words, self.docs[i])
            new_texts.append(new_text)

        if show_tfidf:
            # Show changed TF-IDF-scores
            new_results = self.calculate_tfidf(new_texts)
            new_df = self.make_dataframe(new_results)
            self.show_results(df, new_df)

        return new_texts


def find_replace_idx(alignment: list, indices: list, flat=False):
    replace_ids = []
    for i in indices:
        replace_ids.append([a[1] for a in alignment if a[0] == i])
    if flat:
        flat_replace = [i for x in replace_ids for i in x]
        return flat_replace
    return replace_ids


def divide_into_chunks(lst: list, n: int):
    chunked = []
    for i in range(len(lst), n):
        chunked.append(lst[i:i + n])
    return chunked


def separate_into_documents(text: list, domain: str) -> list:
    """
    Returns text separated into documents based on average number of sentences
    per document for a certain domain.
    :param text: Text to be separated into documents
    :param domain: Domain (needed for avg. number of sentences)
    :return: List of lists with documents
    """
    doc_lens = {"EMEA": 152, "GNOME": 166, "JRC": 29}
    docs = []
    i = 0
    doc_len = doc_lens[domain]
    while i + doc_len < len(text):
        docs.append(text[i:i+doc_len])
        i = i + doc_len
    docs.append(text[i:])

    return docs


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}

        for language in ["en", "de"]:
            d[f"{language}"], _, _ = get_all_texts(domain, language)
        d_alignments = postprocess_alignment_file(domain)
        d_docs = {"en": separate_into_documents(d["en"], domain),
                  "de": separate_into_documents(d["de"], domain)}
        for src, trg in [("en", "de"), ("de", "en")]:
            pass
        if tf:
            tfidf_anonymizer = TFIDF(docs, language)
            anonymized_tfidf = tfidf_anonymizer.anonymize()
            flat_text = [sentence for doc in anonymized_tfidf for sentence in doc]
        if qualitative:
            d[f"{language}"]["anonymized"] = flat_text
        if do_ner:
            ner_anonymizer = NER(d["en"], d["de"], "en", d_alignments["en-de"])
            anonymized_ner = ner_anonymizer.anonymize()
            final_text = []
            for s in anonymized_ner:
                x = s.strip() + "\n"
                final_text.append(x)
            with open(f"../data/3_anonymized/full/{domain}.{language}", "w") as f:
                f.writelines(final_text)

        if qualitative:
            print("Writing to documents...")
            original = d["de"]["original"]
            anon_de = d["de"]["anonymized"]
            anon_en = d["en"]["anonymized"]
            with open(f"../data/3_anonymized/quali/{domain}.out", "w") as f:
                for i in range(len(original)-1):
                    if original[i] != anon_de[i]:
                        f.write(f"German original: {original[i]}")
                        f.write(f"German anonymized: {anon_de[i]}")
                        f.write(f"English anonymized: {anon_en[i]}\n")


if __name__ == "__main__":
    main()
