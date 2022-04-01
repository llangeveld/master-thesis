from sacrebleu import BLEU
from sentence_transformers import SentenceTransformer, util

bleu = BLEU(effective_order=True)
model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


def evaluate_bleu(text: list, anon_text: list) -> float:
    """
    Returns the ratio of sentences that were properly anonymized
    :param text: List of unanonymized sentences
    :param anon_text: Parallel list of anonymized sentences
    :return: Number of sentences that were not properly anonymized (int)
    """
    found = 0
    empty = 0
    # Go over every sentence in the anonymized text
    for idx, sentence in enumerate(anon_text):
        max_idx = 0
        max_score = 0
        # Go over every sentence in the regular text
        for i, s in enumerate(text):
            if not s:
                empty += 1
                continue
            score = bleu.sentence_score(sentence, [s]).score
            if score > max_score:
                max_score = score
                max_idx = i
        if text[max_idx] == text[idx]:
            found += 1

    ratio = found / (len(text) - empty)
    return 1 - ratio


def evaluate_semsim(text: list, anon_text: list) -> float:
    sim_scores = []
    for before, after in zip(text, anon_text):
        embed_before = model.encode(before, convert_to_tensor=True)
        embed_after = model.encode(after, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embed_before, embed_after)
        sim_scores.append(cosine_score.item())
    score = sum(sim_scores) / len(text)
    return score


def evaluate_fscore(text: list, anon_text: list) -> None:
    bleu_score = evaluate_bleu(text, anon_text)
    semsim_score = evaluate_semsim(text, anon_text)
    # f1 = 2 * ((bleu_score * semsim_score) / (bleu_score + semsim_score))
    avg = (bleu_score+semsim_score)/2
    print(f"BLEU-score: {bleu_score}\n"
          f"Semantic score: {semsim_score}\n"
          f"F1-score: {avg}")
