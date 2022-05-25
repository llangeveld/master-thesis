#!bin/env python
from util import get_all_texts
from eval import calculate_bleu
import translate
import sacremoses
# Note: This import doesn't seem to do anything, but the program WILL NOT
# work without it. Do NOT remove this import.


def translation_pipeline(d: dict):
    print("Translating English to German...", end="")
    en2de = translate.translate_en2de(d["test_en"])
    print("Done.")
    print("Translating German to English...", end="")
    de2en = translate.translate_de2en(d["test_de"])
    print("Done.")

    print("Calculating English BLEU...")
    en2de_bleu = calculate_bleu(en2de, d["test_de"])
    print("Calculated German BLEU...\n")
    de2en_bleu = calculate_bleu(de2en, d["test_en"])
    print(f"English to German BLEU: {en2de_bleu}")
    print(f"German to English BLEU: {de2en_bleu}")


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d = {}
        for language in ["en", "de"]:
            d[f"train_{language}"], d[f"test_{language}"], d[f"valid_{language}"] = get_all_texts(
                domain, language)
        translation_pipeline(d)


if __name__ == "__main__":
    main()
