from sacremoses import MosesTokenizer
from utils import get_all_texts, remove_wrong_language


def count_tokens(text: list, language: str):
    mt = MosesTokenizer(lang=language)
    tokens = []
    for sentence in text:
        s_tok = mt.tokenize(sentence)
        tokens = tokens + s_tok

    unique = set(tokens)

    return len(tokens), len(unique)


def print_tokens(before_tok: int, before_unq: int,
                 after_tok: int, after_unq: int):
    print(f"      Number of tokens (uncorrected): {before_tok}")
    print(f"      Number of unique tokens (uncorrected): {before_unq}")
    print(f"      Number of tokens (corrected): {after_tok}")
    print(f"      Number of unique tokens (corrected): {after_unq}")


def print_proportion(before_tok: int, before_unq: int,
                     after_tok: int, after_unq: int):
    tok = (after_tok-before_tok)/before_tok
    unq = (after_unq-before_unq)/before_unq
    print(f"      Decrease in tokens: {tok}")
    print(f"      Decrease in unique tokens: {unq}")


def calculate_tokens(text: list, new_text: list,
                     language: str, option="count"):
    before_t, before_u = count_tokens(text, language)
    after_t, after_u = count_tokens(new_text, language)
    if option == "count":
        print_tokens(before_t, before_u, after_t, after_u)
    elif option == "prop":
        print_proportion(before_t, before_u, after_t, after_u)
    else:
        print("Option not recognized. Try either 'count' or 'prop'.")


def print_results_per_type(text_en: list, text_de: list, phase: str):
    print(f"  Results for {phase}:")

    new_en, new_de = remove_wrong_language(text_en, text_de)
    print(f"    Number of sentences (uncorrected): {len(text_en)}")
    print(f"    Number of sentences (corrected): {len(new_en)}")
    print(f"    Decrease in sentences: "
          f"{(len(new_en)-len(text_en))/len(text_en)}")

    print(f"    Results for English:")
    calculate_tokens(text_en, new_en, "en", "prop")

    print(f"    Results for German:")
    calculate_tokens(text_de, new_de, "de", "prop")


def get_results(domain: str):
    en_train, en_test, en_valid = get_all_texts(domain, "en")
    de_train, de_test, de_valid = get_all_texts(domain, "de")
    print_results_per_type(en_train, de_train, "train")
    print_results_per_type(en_test, de_test, "test")
    print_results_per_type(en_valid, de_valid, "valid")


def main():
    print("Results for EMEA:")
    get_results("EMEA")
    print("Results for GNOME:")
    get_results("GNOME")
    print("Results for JRC:")
    get_results("JRC")


if __name__ == "__main__":
    main()

