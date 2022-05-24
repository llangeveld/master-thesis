from util import make_language_files
import numpy as np

for domain in ["EMEA", "JRC", "GNOME"]:
    docs, _ = make_language_files(domain)
    c = 0
    lens = [len(doc) for doc in docs]
    q25, q75 = np.percentile(lens, [25,75])
    ran = q75-q25
    max_len = q75 + (1.5*ran)
    min_len = q25 - (1.5*ran)
    lens_new = []
    for doc in docs:
        x = len(doc)
        if min_len <= x <= max_len:
            lens_new.append(x)
    print(f"Average without removing outliers for {domain}: {sum(lens)/len(docs)}")
    print(f"Average after removing outliers for {domain}: {sum(lens_new)/len(docs)}")
