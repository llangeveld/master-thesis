from sacremoses import MosesDetokenizer


def main():
    for language in ["en", "de"]:
        md = MosesDetokenizer(lang=f"{language}")
        for domain in ["EMEA", "GNOME", "JRC"]:
            for phase in ["test", "train", "valid"]:
                f = open(f"../data/1_main_data/{domain}/{phase}.{language}")
                f_list = [s for s in f]
                detoks = [md.detokenize(s.split()) for s in f_list]
                with open(f"../data/1_main_data/detok/{domain}/{phase}.{language}", "w") as f:
                    for s in detoks:
                        f.write(s + "\n")
            for pair in ["de-en", "en-de"]:
                x = open(f"../data/3_anonymized/full/{domain}.{pair}.{language}")
                x_list = [s for s in x]
                anon_detoks = [md.detokenize(s.split()) for s in x_list]
                with open(f"../data/3_anonymized/detok/{domain}.{pair}.{language}", "w") as z:
                    for s in anon_detoks:
                        z.write(s)


if __name__ == "__main__":
    main()
