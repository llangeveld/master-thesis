from sacremoses import MosesDetokenizer
import argparse
md = MosesDetokenizer(lang="de")
parser = argparse.ArgumentParser()

parser.add_argument("--input", "--i", type=str, help="Path to input file")
parser.add_argument("--output", "--o", type=str, help="Path to output file")

args = parser.parse_args()

sys_out = [s for s in open(str(args.input))]
detoks = [md.detokenize(s.split()) for s in sys_out]
with open(args.output, "w") as f:
    for s in detoks:
        f.write(s + "\n")
