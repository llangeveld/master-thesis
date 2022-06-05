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


if __name__ == "__main__":
    main()