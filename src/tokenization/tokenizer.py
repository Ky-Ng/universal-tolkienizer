import re
import string
from typing import Counter, Mapping

CJK_RANGE = r"\u4E00-\u9FFF"
SEGMENT = re.compile(
    fr"[{CJK_RANGE}]+|[^\s{CJK_RANGE}{string.punctuation}]+|[{string.punctuation}]+")
SPACE = re.compile(r"\s+")

EOS = "</w>"
# Helper Functions for Sennrich et al.


def segment_words(text: str) -> list[str]:
    """
    Helper to split sentences on whitespace, punctuation, or script boundary

    For whitespace delimited languages: Converts a sentence/continuous text into words
    i.e. "This sentence is short" => ["this", "sentence", "is", "short"]

    For non-whitespce delimited languages: Converts a sentence/continuous text into runs
    i.e "今天好熱！但是因為裡面有空調，所以還好" => ["今天好熱", "！", "但是因為裡面有空調", "，", "所以還好"]
    """
    return SEGMENT.findall(text)


print(segment_words("今天好熱！但是因為裡面有空調，所以還好"))

print(segment_words("This sentence is short."))


def _word_to_symbols(word: str) -> list[str]:
    """
    Converts a word to a list of symbols with the EOS token
    e.g. "john" => ["j", "o", "h", "n", <EOS>] 
    e.g. "我們" => ["我", "們", <EOS>]
    """
    return list(word) + [EOS]


def _get_bigram(vocab_counts: Mapping[str, int]) -> dict[tuple[str, str], int]:
    """
    Calculate the co-occurence of tokens based on their weight in the vocab

    e.g. "low lower mid low"
    vocab_counts = {"l o w <EOS>": 2, "l o w e r <EOS>": 1, "m i d <EOS>": 1} 
    bigram = _get_bigram(vocab_counts)
    > bigram => {("l", "o"): 2, ("w" , <EOS>): 1, ("e", "r"): 1, ...} 
    """
    pairs = Counter()
    for word_str, freq in vocab_counts.items():
        symbols = word_str.split()
        for l, r in zip(symbols, symbols[1:]):
            pairs[(l, r)] += freq
    return pairs


print(_get_bigram(vocab_counts={"l o w <EOS>": 2,
      "l o w e r <EOS>": 1, "m i d <EOS>": 1}))


def _merge_token(pair: tuple[str, str],
                vocab_counts: Mapping[str, int]) -> dict[str, int]:
    """
    Merges all instances of `pair` in the vocab_counts and returns an updated vocab
    
    e.g. pair = ("l", "o")
    vocab_counts = {"l o w <EOS>": 2, "l o w e r <EOS>": 1, "m i d <EOS>": 1} 
    _merge_token = vocab_counts = {"lo w <EOS>": 2, "lo w e r <EOS>": 1, "m i d <EOS>": 1} 
    """
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab: dict[str, int] = {}
    for word_str, freq in vocab_counts.items():
        merged = p.sub("".join(pair), word_str)
        new_vocab[merged] = new_vocab.get(merged, 0) + freq
    return new_vocab

print(_merge_token(pair=("l","o"), vocab_counts={"l o w <EOS>": 2, "l o w e r <EOS>": 1, "m i d <EOS>": 1} ))