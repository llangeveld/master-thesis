f = open("../data/opus-test/gnome.out")

doc_num = 0
en = []
de = []
docs = {}
prev = ""
for s in f:
    s = s.strip()
    if not s:
        continue
    elif s[0] == "#":
        if prev != "#":
            docs[f"{doc_num}"] = {"en": en, "de" : de}
            print(en)
            print(de)
            doc_num += 1
            en = []
            de = []
    elif s[0] == "=":
        continue
    else:
        if s.startswith("(src)"):
            idx = s.find('>')
            de.append(s[idx+1:])
        elif s.startswith("(trg)"):
            idx = s.find('>')
            en.append(s[idx+1:])
    prev = s[0]