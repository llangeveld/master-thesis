

def main():
    d = {}
    with open("translations.out", "r") as f:
        for s in f:
            s_list = s.split("\t")
            num = int(s_list[0].split("-")[1])
            d[num] = s_list[2:]
    sorted_d = dict(sorted(d.items()))
    translations = [v for v in sorted_d.values()]
    with open("translations_sorted.out", "w") as f:
        for s in translations:
            f.write(s[0])

if __name__ == "__main__":
    main()
