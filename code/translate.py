import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English to German model
en2de = torch.load('/data/s3225143/wmt19.en-de.joined-dict.ensemble.tar.gz',
                   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                   tokenizer='moses', bpe='fastbpe').to(device)

# German to English model
de2en = torch.load('/data/s3225143/wmt19.de-en.joined-dict.ensemble.tar.gz',
                   checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                   tokenizer='moses', bpe='fastbpe').to(device)


def translate_en2de(text: list) -> list:
    """
    Translates English text into German text
    :param text: List of sentences
    :return: List of translated sentences
    """
    translated_text = []
    i = 0
    for sentence in text:
        translated_text.append(en2de.translate(sentence))
        if i % 100 == 0:
            print(f"\tTranslated up to sentence {i/100}")
    return translated_text


def translatede2en(text: list):
    """
    Translates German text into English text
    :param text: List of sentences
    :return: List of translated sentences
    """
    translated_text = []
    i = 0
    for sentence in text:
        translated_text.append(de2en.translate(sentence))
        if i % 100 == 0:
            print(f"\tTranslated up to sentence {i/100}")
    return translated_text
