from thefuzz import process
from utils import load_full_files
import anonymize
import random

engine = anonymize.create_presidio_analyzer()
anonymizer = anonymize.create_presidio_anonymizer()
NUM_SENTS = 150


def do_analysis(text: list, anon_text: list):
    found = 0
    idxs = random.sample(range(len(text)-1), NUM_SENTS)
    new_text = [text[i] for i in idxs]
    for i in idxs:
        hyp, _ = process.extractOne(anon_text[i], new_text)
        if hyp == text[i]:
            found += 1

    return found


def main():
    d = load_full_files()
    for k, v in d.items():
        sp_text = anonymize.anonymize_text_spacy(v, k[-2:])
        f = do_analysis(v, sp_text)
        ratio = f/NUM_SENTS
        print(f"Results for {k}: {ratio}")


if __name__ == "__main__":
    main()
