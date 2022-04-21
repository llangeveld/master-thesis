import torch
import sacremoses
# Note: This import doesn't seem to do anything, but the program WILL NOT
# work without it. Do NOT remove this import.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English to German model
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe', verbose=False).to(device)

# German to English model
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                       tokenizer='moses', bpe='fastbpe', verbose=False).to(device)


def translate_en2de(text: list) -> list:
    """
    Translates English text into German text
    :param text: List of sentences
    :return: List of translated sentences
    """
    translated_text = []
    for sentence in text:
        translated_text.append(en2de.translate(sentence))
    return translated_text


def translate_de2en(text: list):
    """
    Translates German text into English text
    :param text: List of sentences
    :return: List of translated sentences
    """
    translated_text = []
    for sentence in text:
        translated_text.append(de2en.translate(sentence))
    return translated_text
