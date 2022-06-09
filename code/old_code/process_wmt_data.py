from bs4 import BeautifulSoup, SoupStrainer


en = open("/home/lonneke/Downloads/ende.en.sgm", "r")
en_data = en.read()
en_soup = BeautifulSoup(en_data)
en_sents = en_soup.findAll('seg')
with open("/home/lonneke/thesis/Data/tests/wmt/en.txt", "w") as f:
    for sent in en_sents:
        f.write(sent.text + "\n")

de = open("/home/lonneke/Downloads/ende.de.sgm", "r")
de_data = de.read()
de_soup = BeautifulSoup(de_data)
de_sents = de_soup.findAll('seg')
with open("/home/lonneke/thesis/Data/tests/wmt/de.txt", "w") as f:
    for sent in de_sents:
        f.write(sent.text + "\n")
