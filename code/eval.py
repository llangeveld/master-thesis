from utils import load_full_files
from sacrebleu import BLEU
import anonymize
import random

print("Initializing engines...")
engine = anonymize.create_presidio_analyzer()
anonymizer = anonymize.create_presidio_anonymizer()
bleu = BLEU(effective_order=True)
NUM_SENTS = 150


def do_analysis(text: list, anon_text: list) -> int:
    found = 0
    idxs = random.sample(range(len(text)-1), NUM_SENTS)
    new_text = [text[i] for i in idxs]
    for i in idxs:
        max_idx = 0
        max_score = 0
        for idx, s in enumerate(new_text):
            score = bleu.sentence_score(anon_text[i], [s]).score
            if score > max_score:
                max_score = score
                max_idx = idx

        if new_text[max_idx] == text[i]:
            found += 1

    return found


def main():
    print("Starting program...")
    d = load_full_files()
    for k, v in d.items():
        sp_text = anonymize.anonymize_text_spacy(v.strip(), k[-2:])
        f = do_analysis(v, sp_text)
        ratio = f/NUM_SENTS
        print(f"Results for {k}: {ratio}")


if __name__ == "__main__":
    main()
