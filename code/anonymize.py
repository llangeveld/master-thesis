from sklearn.feature_extraction.text import TfidfVectorizer
from util import get_all_texts, postprocess_alignment_file
import pandas as pd
import numpy as np
import random
import spacy

NGRAM_RANGE = (2, 2)
do_ner = False
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

    def anonymize(self) -> tuple:
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

        return anonymized_text_src, anonymized_text_trg


class TFIDF:
    def __init__(self, src_text, trg_text, src_lan, align, domain):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src_lan = src_lan
        self.trg_lan = "en" if self.src_lan == "de" else "de"
        self.align = align
        self.src_docs = separate_into_documents(self.src_text, domain)
        self.trg_docs = separate_into_documents(self.trg_text, domain)

    def calculate_tfidf(self, docs: list) -> list:
        """
        Returns list of TF-IDF results for documents.
        :param docs: list of documents to be calculated
        :return: list of results: doc-num, words, tf-idf score.
        """
        # Calculate TF-IDF
        if self.src_lan == "en":
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

    def find_most_similar_words(self, word, language, n=4):
        """
        Finds the n most similar words to input word in the given language
        :param language: Language for which to find most similar words
        :param word: Word that will get most similar words as output
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

    def get_replace_strings(self, words: list, lan: str) -> list:
        """
        Finds replacement strings using SpaCy
        :param lan: Language
        :param words: list of words in need of replacement
        :return: a list of lists of strings with replacement words per word
        """
        replace_strings = []
        for word in words:
            syns = [syn for syn in self.find_most_similar_words(word, lan)]
            syns.append(word)
            replace_strings.append(syns)
        return replace_strings

    def change_word(self, words: list, text: list, text_trg: list, alignment: list) -> tuple[list, list]:
        """
        Replaces given words by random synonyms in a given text.
        :param text_trg: Target text
        :param words: Words that need replacing
        :param text: Text in which words need to be replaced
        :return: List of sentences wherein the words have been replaced
        """
        # n_text = text.copy()
        n_text = []
        new_text_trg = []
        for word_set in words:
            words_list = word_set.split()
            replace_strings = self.get_replace_strings(words_list, self.src_lan)
            for idx, sentence in enumerate(text):
                s_list = sentence.split()
                new_sentence_src = s_list.copy()
                new_sentence_trg = text_trg[idx].split()
                indices = []
                for i, w in enumerate(s_list):
                    if w in words_list:
                        indices.append(i)
                if indices:
                    indices_divided = divide_into_chunks(indices, len(words_list))
                    if isinstance(indices_divided[0], int):
                        for j, ix in enumerate(indices_divided):
                            new_sentence_src[ix] = str(random.choice(replace_strings[j]))
                    else:
                        for w_set in indices_divided:
                            for j, ix in enumerate(w_set):
                                new_sentence_src[ix] = str(random.choice(replace_strings[j]))
                    s_align = alignment[idx]
                    replace_ids = find_replace_idx(s_align, indices)
                    trg_sentence = text_trg[idx]
                    trg_sentence_lst = trg_sentence.split()
                    replace_words = []
                    for i in flatten_list(replace_ids):
                        replace_words.append(trg_sentence_lst[i])
                    replace_words = [trg_sentence_lst[i] for i in flatten_list(replace_ids)]
                    replace_strings_trg = self.get_replace_strings(replace_words, self.trg_lan)
                    for w_set in replace_ids:
                        for j, ix in enumerate(w_set):
                            new_sentence_trg[ix] = str(random.choice(replace_strings_trg[j]))
                    n_text.append(" ".join(new_sentence_src))
                    new_text_trg.append(" ".join(new_sentence_trg))
                else:
                    n_text.append(sentence)
                    new_text_trg.append(text_trg[idx])

        return n_text, new_text_trg

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

    def anonymize(self, show_tfidf=False) -> tuple:
        """
        Calculates TF-IDF score for a "before"-document and the document in which
        the most common n-grams have been replaced.
        :param show_tfidf: Whether to show table with TF-IDF-scores
        :return: List of documents in which common n-grams have been replaced
        """
        # Calculate TF-IDF-scores for original texts
        results = self.calculate_tfidf(self.src_docs)
        df = self.make_dataframe(results)
        new_texts = []
        new_texts_trg = []
        # Change words in documents
        for i in range(len(self.src_docs)):
            words = df[df["Doc"] == i]["Word(s)"].tolist()[:5]
            new_text_src, new_text_trg = self.change_word(words, self.src_docs[i],
                                                          self.trg_docs[i],
                                                          self.align[i])
            new_texts.append(new_text_src)
            new_texts_trg.append(new_text_trg)

        if show_tfidf:
            # Show changed TF-IDF-scores
            new_results = self.calculate_tfidf(new_texts)
            new_df = self.make_dataframe(new_results)
            self.show_results(df, new_df)

        return new_texts, new_texts_trg


def flatten_list(lst: list) -> list:
    return [i for x in lst for i in x]


def find_replace_idx(alignment: list, indices: list, flat=False):
    replace_ids = []
    for i in indices:
        replace_ids.append([a[1] for a in alignment if a[0] == i])
    if flat:
        return flatten_list(replace_ids)
    return replace_ids


def divide_into_chunks(lst: list, n: int):
    if len(lst) <= n:
        return lst
    chunked = []
    for i in range(0, len(lst), n):
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
        docs.append(text[i:i + doc_len])
        i = i + doc_len
    docs.append(text[i:])

    return docs


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}
        for language in ["en", "de"]:
            d[f"{language}"], _, _ = get_all_texts(domain, language)
        d_alignments = postprocess_alignment_file(domain)
        d_alignments_docs = {"en-de": separate_into_documents(d_alignments["en-de"], domain),
                             "de-en": separate_into_documents(d_alignments["de-en"], domain)}
        for src, trg in [("en", "de"), ("de", "en")]:
            d_tfidf = {}
            d_ner = {}
            tfidf_anonymizer = TFIDF(d[f"{src}"], d[f"{trg}"], src,
                                     d_alignments_docs[f"{src}-{trg}"], domain)
            docs_src, docs_trg = tfidf_anonymizer.anonymize()
            d_tfidf[f"{src}"] = flatten_list(docs_src)
            d_tfidf[f"{trg}"] = flatten_list(docs_trg)
            if do_ner:
                ner_anonymizer = NER(d_tfidf[f"{src}"], d_tfidf[f"{trg}"],
                                     src, d_alignments[f"{src}-{trg}"])
                d_ner[f"{src}"], d_ner[f"{trg}"] = ner_anonymizer.anonymize()
                for lan in [src, trg]:
                    final_text = []
                    for s in d_ner[f"{lan}"]:
                        x = s.strip() + "\n"
                        final_text.append(x)
                    with open(f"../data/3_anonymized/full/{domain}.{src}-{trg}.{lan}", "w") as f:
                        f.writelines(final_text)

            if qualitative and src == "en":
                print("Writing to documents...")
                original = d["de"]
                anon_de = d_tfidf["de"]
                anon_en = d_tfidf["en"]
                with open(f"../data/3_anonymized/quali/{domain}.out", "w") as f:
                    for i in range(len(original) - 1):
                        if original[i] != anon_de[i]:
                            f.write(f"German original: {original[i]}")
                            f.write(f"German anonymized: {anon_de[i]}")
                            f.write(f"English anonymized: {anon_en[i]}\n")


if __name__ == "__main__":
    main()
