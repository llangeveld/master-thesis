from sklearn.feature_extraction.text import TfidfVectorizer
from util import get_all_texts, postprocess_alignment_file
from sacrebleu import BLEU
from sentence_transformers import SentenceTransformer, util
from sacremoses import MosesTokenizer
import pandas as pd
import numpy as np
import random
import spacy

NGRAM_RANGE = (2, 2)
N_SENTS = 200
DO_NER = False
DO_TFIDF = False
SAVE = False
EVAL = True
nlp_en = spacy.load("en_core_web_lg")
nlp_de = spacy.load("de_core_news_lg")
bleu = BLEU(effective_order=True)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
mt_en = MosesTokenizer("en")
mt_de = MosesTokenizer("de")


class Evaluate:
    def __init__(self, text, anon_text):
        self.text = text
        self.anon_text = anon_text

    def evaluate_bleu(self):
        """
            Returns the ratio of sentences that were properly anonymized
            :return: Number of sentences that were not properly anonymized (int)
            """
        found = 0
        # Go over every sentence in the anonymized text
        for idx, sentence in enumerate(self.anon_text):
            max_idx = 0
            max_score = 0
            # Go over every sentence in the regular text
            for i, s in enumerate(self.text):
                score = bleu.sentence_score(sentence, [s]).score
                if score > max_score:
                    max_score = score
                    max_idx = i
            if self.text[max_idx] == self.text[idx]:
                found += 1
            #     print("FOUND!")
            # print(f"Anonymized sentence: {sentence}")
            # print(f"Regular sentence: {self.text[idx]}")
            # print(f"Chosen sentence: {self.text[max_idx]}")
            # print("----------------------------------")
        ratio = found / len(self.text)
        print(f"Found {found} out of {len(self.text)}")
        return 1 - ratio

    def evaluate_semsim(self) -> float:
        """
        Returns average cosine (semantic) similarity of all original-anonymized
        sentence pairs.
        :return: Cosine similarity score
        """
        sim_scores = []
        for before, after in zip(self.text, self.anon_text):
            embed_before = model.encode(before, convert_to_tensor=True)
            embed_after = model.encode(after, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embed_before, embed_after)
            sim_scores.append(cosine_score.item())
            # print(f"Regular sentence: {before}")
            # print(f"Anonymized sentence: {after}")
            # print(f"Similarity score: {cosine_score.item()}")
            # print("---------------------------------------")
        score = sum(sim_scores) / len(self.text)
        return score

    def evaluate_fscore(self) -> None:
        """
        Calculates (harmonic) mean between BLEU- and semantic similarity scores
        :return: Nothing
        """
        bleu_score = self.evaluate_bleu()
        semsim_score = self.evaluate_semsim()
        f1 = 2 * ((bleu_score * semsim_score) / (bleu_score + semsim_score))
        print(f"BLEU-score: {bleu_score}\n"
              f"Semantic score: {semsim_score}\n"
              f"F1-score: {f1}")


class NER:
    def __init__(self, src_text, trg_text, src_lan, align):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src_lan = src_lan
        self.trg_lan = "en" if self.src_lan == "de" else "de"
        self.align = align

    def replace_in_trg(self, idx: int, indices: list, ent_label: str) -> str:
        """
        Replace words in target sentence.
        :param idx: Target sentence index (to get right sentence)
        :param indices: List of indices that need to be replaced
        :param ent_label: Entity label with which to replace text
        :return: New sentence with entity replaced (str)
        """
        # Find the corresponding words in target text
        alignment = self.align[idx]
        replace_ids = find_replace_idx(alignment, indices, flat=True)
        sentence = self.trg_text[idx]
        # Replace entity label
        tok = mt_en if self.trg_lan == "en" else mt_de
        s_new = tok.tokenize(sentence)
        if replace_ids:
            i_start = replace_ids[0]
            for i in replace_ids:
                # Only replace first word
                if i == i_start:
                    s_new[i] = f"<{ent_label}>"
                else:
                    s_new[i] = ""
            s_string = " ".join([w for w in s_new if w != ""])
        else:
            s_string = sentence
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
            tok = mt_en if self.src_lan == "en" else mt_de
            s_list = tok.tokenize(s)
            # Find words in list based on starting characters of words in str
            s_startchars = []
            c = 0
            for i in range(len(s_list)):
                s_startchars.append(c)
                c += len(s_list[i]) + 1
            # Replace words in source & target sentence
            s_new = s_list.copy()
            new_trg = self.trg_text[idx]
            indices = []
            for ent in doc.ents:
                try:  # Differences in tokenization
                    i = s_startchars.index(ent.start_char)
                    i_start = i
                    ent_list = ent.text.split()
                    # Replace words + find indices for target words
                    for _ in ent_list:
                        indices.append(i)
                        if i == i_start:
                            s_new[i] = f"<{ent.label_}>"
                        else:
                            s_new[i] = ""
                        i += 1
                except ValueError:
                    break
                # Replace words in target sentence
                if indices:
                    new_trg = self.replace_in_trg(idx, indices, ent.label_)
                else:
                    new_trg = self.trg_text[idx]
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

    @staticmethod
    def find_most_similar_words(word, language, n=4):
        """
        Finds the n most similar words to input word in the given language
        :param language: Language for which to find most similar words
        :param word: Word that will get most similar words as output
        :param n: Number of most similar words
        :return: list of n most similar words to input word
        """
        # Find correct NLP engine
        if language == "en":
            sp = nlp_en
        elif language == "de":
            sp = nlp_de
        else:
            raise ValueError("Argument 'language' must be 'en' or 'de'.")
        word = sp.vocab[str(word)]  # Convert word to word vector
        # Find words similar to input word
        queries = [
            w for w in word.vocab
            if w.is_lower == word.is_lower and w.prob >= -30 and np.count_nonzero(w.vector)
        ]
        # Sort by how much alike the words are
        by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
        # Make human-readable list
        values = [
            w.lower_ for w in by_similarity[:n + 1]
            if w.lower_ != word.lower_
        ]
        if values:
            return values
        else:  # If there are no similar words
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

    def change_word(self, words: list, text: list, text_trg: list,
                    alignment: list) -> tuple[list, list]:
        """
        Replaces given words by random synonyms in a given text.
        :param alignment: Alignment files for document
        :param text_trg: Document in target language
        :param words: Words that need replacing
        :param text: Document in source language
        :return: List of sentences wherein the words have been replaced
        """
        n_text = text.copy()
        new_text_trg = text_trg.copy()
        # Go over word sets
        for word_set in words:
            words_list = word_set.split()
            # Find replace strings in source language
            replace_strings = self.get_replace_strings(words_list, self.src_lan)
            # Go over every sentence in the text
            for idx, sentence in enumerate(n_text):
                s_list = sentence.split()
                new_sentence_src = s_list.copy()
                new_sentence_trg = text_trg[idx].split()
                indices = []
                # Find indices for words to replace
                for i, w in enumerate(s_list):
                    if w in words_list:
                        indices.append(i)
                if indices:
                    # Replace words in source text
                    indices_divided = divide_into_chunks(indices, len(words_list))
                    if isinstance(indices_divided[0], int):
                        for j, ix in enumerate(indices_divided):
                            new_sentence_src[ix] = str(random.choice(replace_strings[j]))
                    else:
                        for w_set in indices_divided:
                            for j, ix in enumerate(w_set):
                                new_sentence_src[ix] = str(random.choice(replace_strings[j]))
                    # Replace words in target text
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

                    # Add to texts
                    n_text[idx] = " ".join(new_sentence_src)
                    new_text_trg[idx] = " ".join(new_sentence_trg)

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
        print("\tCalculated TFIDF. Replacing words.")
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
    """Flattens 2D-list to 1D-list"""
    return [i for x in lst for i in x]


def find_replace_idx(alignment: list, indices: list, flat=False) -> list:
    """
    Finds replacement indices given an alignment file
    :param alignment: Alignment list of sentence
    :param indices: List of indices in source language that are replaced
    :param flat: If a flat list needs to be returned
    :return: List of indices to replace in target text
    """
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
    if doc_len >= len(text):
        return [text]
    while i + doc_len < len(text):
        docs.append(text[i:i + doc_len])
        i = i + doc_len
    docs.append(text[i:])

    return docs


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        print(f"Working on {domain}...")
        d = {}
        for language in ["en", "de"]:
            d[f"{language}"], _, _ = get_all_texts(domain, language)
        d_alignments = postprocess_alignment_file(domain)
        d_alignments_docs = {"en-de": separate_into_documents(d_alignments["en-de"], domain),
                             "de-en": separate_into_documents(d_alignments["de-en"], domain)}
        print(f"\tProcessed files. Starting anonymization on TFIDF.")
        for src, trg in [("en", "de"), ("de", "en")]:
            print(f"\tStarting first language pair.")
            d_tfidf = {}
            d_ner = {}
            if DO_TFIDF:
                tfidf_anonymizer = TFIDF(d[f"{src}"], d[f"{trg}"], src,
                                         d_alignments_docs[f"{src}-{trg}"], domain)
                docs_src, docs_trg = tfidf_anonymizer.anonymize()
                print("\tReplaced words. Start working on NER.")
                d_tfidf[f"{src}"] = flatten_list(docs_src)
                d_tfidf[f"{trg}"] = flatten_list(docs_trg)
            if DO_NER:
                if DO_TFIDF:
                    ner_anonymizer = NER(d_tfidf[f"{src}"], d_tfidf[f"{trg}"],
                                         src, d_alignments[f"{src}-{trg}"])
                else:
                    ner_anonymizer = NER(d[f"{src}"], d[f"{trg}"], src,
                                         d_alignments[f"{src}-{trg}"])
                d_ner[f"{src}"], d_ner[f"{trg}"] = ner_anonymizer.anonymize()
                print("\tAnonymized NER. Writing to files.")
            for lan in [src, trg]:
                final_text = []
                if DO_NER:
                    text = d_ner[f"{lan}"]
                elif DO_TFIDF:
                    text = d_tfidf[f"{lan}"]
                if SAVE:
                    for s in text:
                        x = s.strip() + "\n"
                        final_text.append(x)
                    if DO_NER and DO_TFIDF:
                        with open(f"../data/3_anonymized/full/{domain}.{src}-{trg}.{lan}", "w") as f:
                            f.writelines(final_text)
                    elif DO_NER:
                        with open(f"../data/3_anonymized/ner/{domain}.{src}-{trg}.{lan}", "w") as f:
                            f.writelines(final_text)
                    elif DO_TFIDF:
                        with open(f"../data/3_anonymized/tfidf/{domain}.{src}-{trg}.{lan}", "w") as f:
                            f. writelines(final_text)
                if EVAL:
                    if not SAVE:
                        # for type in ["full", "ner", "tfidf"]:
                        for type in ["full"]:
                            if type == "full":
                                anon_f = open(f"../data/3_anonymized/tokenized/{domain}.{src}-{trg}.{lan}")
                            else:
                                anon_f = open(f"../data/3_anonymized/{type}/{domain}.{src}-{trg}.{lan}")
                            anon_text = anon_f.readlines()
                            reg_f = open(f"../data/1_main_data/tokenized/{domain}/train.{lan}")
                            reg_text = reg_f.readlines()
                            eval = Evaluate(reg_text[:N_SENTS], anon_text[:N_SENTS])
                            print(f"EVALUATION SCORE FOR {type} | {domain} | {src}-{trg} | {lan}:")
                            eval.evaluate_fscore()


if __name__ == "__main__":
    main()
