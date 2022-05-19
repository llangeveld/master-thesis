p = "/data/s3225143/data/4_translation/processed"
NUM = 42020

for domain in ["EMEA", "JRC", "GNOME"]:
    for pair in ["en-de", "de-en"]:
        for language in ["en", "de"]:
            new_list = []
            sp = f"{p}/{domain}.{pair}"
            f = open(f"{sp}/dict.{language}.txt")
            text = [s for s in f]
            for line in text:
                new_list.append(line)
            i = 0
            while len(new_list) < NUM:
                new_list.append(f"fakeword{i} 0\n")
                i += 1
            with open(f"{sp}/dict_new.{language}.txt", "w") as nf:
                for line in new_list:
                    nf.write(line)
            print(f"DONE: {domain} | {pair} | {language}")
