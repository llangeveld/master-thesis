from sacrebleu import BLEU


def do_analysis(text: list, anon_text: list, bleu) -> int:
    found = 0
    # Go over every sentence in the anonymized text
    for idx, sentence in enumerate(anon_text):
        max_idx = 0
        max_score = 0
        # Go over every sentence in the regular text
        for i, s in enumerate(text):
            score = bleu.sentence_score(sentence, [s]).score
            if score > max_score:
                max_score = score
                max_idx = i
        if text[max_idx] == text[idx]:
            found += 1

    return found


def evaluate_bleu(text: list, anon_text: list):
    bleu = BLEU(effective_order=True)
    f = do_analysis(text, anon_text, bleu)
    ratio = f/len(text)
    print(f"BLEU-results: {ratio}")
