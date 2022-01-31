from sacrebleu.metrics import BLEU

fe = open("valid.en", "r")
fd = open("valid.de", "r")

fe_new = [s for s in fe]
fe_test = [fe_new]
bleu = BLEU()
fd_new = [[s for s in fd]]

result = bleu.corpus_score(fe_new, fe_test)
print(f"Result: {result}")