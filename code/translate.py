import torch
import sacremoses
# NOTE: This may seem like an unused statement, but it won't work without it.
# Do NOT remove the import-statement!
from util import get_all_texts
from sacrebleu import BLEU

bleu = BLEU(effective_order=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English to German model
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe').to(device)


def calculate_bleu(hyps: list, refs: list) -> float:
    """
    Calculates corpus BLEU for translated text
    :param hyps: List of translations made by machine
    :param refs: List of the original sentences
    :return: BLEU-score (float)
    """
    result = bleu.corpus_score(hyps, [refs])
    return result.score


def main():
    for domain in ["EMEA", "GNOME", "JRC"]:
        d_og = {}
        print(f"Working on {domain}...")
        for language in ["en", "de"]:
            d_og[f"{language}"] = open(f"/data/s3225143/tests/wmt/{language}.txt", "r").read()
        for src, trg in [("en", "de")]:
            translated_text = []
            for sentence in d_og[f"{src}"]:
                translated_text.append(en2de.translate(sentence))
            score = calculate_bleu(translated_text, d_og[f"{trg}"])
            print(f"Score for {domain}-{src}-{trg}: {score}")


if __name__ == "__main__":
    main()
