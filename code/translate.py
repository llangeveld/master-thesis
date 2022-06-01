import torch
import sacremoses
# NOTE: This may seem like an unused statement, but it won't work without it.
# Do NOT remove the import-statement!
from util import get_all_texts
from sacrebleu import BLEU

bleu = BLEU(effective_order=True)


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
            _, d_og[f"{language}"], _ = get_all_texts(domain, language)
        for src, trg in [("en", "de"), ("de", "en")]:
            model = torch.hub.load(f"/data/s3225143/models/finetune/"
                                   f"{src}-{trg}/{domain}/", "checkpoint_best.pt",
                                   source="local")
            translated_text = []
            for sentence in d_og[f"{src}"]:
                translated_text.append(model.translate(sentence))
            score = calculate_bleu(translated_text, d_og[f"{trg}"])
            print(f"Score for {domain}-{src}-{trg}: {score}")


if __name__ == "__main__":
    main()
