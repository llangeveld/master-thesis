from util import get_all_texts
import torch
import sacremoses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English to German model
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe').to(device)

# German to English model
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe').to(device)

path = "../data/4_translation"
for domain in ["EMEA", "JRC", "GNOME"]:
    d_en2de = {}
    d_de2en = {"en": {}, "de": {}}
    d_path = f"{path}/{domain}"
    for language in ["de", "en"]:
        d = {}
        d["train"], d["test"], d["valid"] = get_all_texts(domain, language)
        for p in ["train", "test", "valid"]:
            text = d[f"{p}"]
            bpe_en2de = [en2de.apply_bpe(s) for s in text]
            with open(f"{d_path}/en2de/{p}.{language}", "w") as f:
                for line in bpe_en2de:
                    f.write(line)
            bpe_de2en = [de2en.apply_bpe(s) for s in text]
            with open(f"{d_path}/de2en/{p}.{language}", "w") as f:
                for line in bpe_de2en:
                    f.write(line)
