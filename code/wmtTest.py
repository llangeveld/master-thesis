import torch

# English to German translation
en2de = torch.load('/data/s3225143/wmt19.en-de.joined-dict.ensemble.tar.gz',
                   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                   tokenizer='moses', bpe='fastbpe')
en2de.translate("Machine learning is great!")

# German to English translation
de2en = torch.load('/data/s3225143/wmt19.de-en.joined-dict.ensemble.tar.gz',
                   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                   tokenizer='moses', bpe='fastbpe')
de2en.translate("Maschinelles Lernen ist großartig!")
