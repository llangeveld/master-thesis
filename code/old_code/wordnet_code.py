from nltk.corpus import wordnet as wn

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