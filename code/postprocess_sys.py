from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang="de")

sys_out = [s for s in open("gen.out.sys")]
detoks = [md.detokenize(s.split()) for s in sys_out]
with open("gen.out.sys.detok", "w") as f:
    for s in detoks:
        f.write(s + "\n")
