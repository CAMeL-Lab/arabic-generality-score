from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar, normalize_teh_marbuta_ar, normalize_alef_maksura_ar
from camel_tools.utils.charsets import UNICODE_PUNCT_CHARSET


def preprocess_text(text, normalize = False):   ## preprocess to calculate unigram probabilities using CAMeL Frequency Lists

    if text is None:
      return None
    for char in text:
        if (char in UNICODE_PUNCT_CHARSET):
            text = text.replace(char, '')
    #remove tatweel
    text = text.replace('\u0640', '')
    text = dediac_ar(text)

    if normalize:
      text = normalize_alef_maksura_ar(text)
      text = normalize_alef_ar(text)
      text = normalize_teh_marbuta_ar(text)

    text = text.strip()
    return text
