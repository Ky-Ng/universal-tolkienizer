from dataclasses import dataclass, field
import re
import string
from typing import Counter, Iterable, Mapping

CJK_RANGE = r"\u4E00-\u9FFF"
SEGMENT = re.compile(
    fr"[{CJK_RANGE}]+|[^\s{CJK_RANGE}{string.punctuation}]+|[{string.punctuation}]+")
SPACE = re.compile(r"\s+")

EOW = "</w>"
UNK = "<unk>"
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
    Converts a word to a list of symbols with the EOW token
    e.g. "john" => ["j", "o", "h", "n", <EOW>] 
    e.g. "我們" => ["我", "們", <EOW>]
    """
    return list(word) + [EOW]


def _get_bigram(vocab_counts: Mapping[str, int]) -> dict[tuple[str, str], int]:
    """
    Calculate the co-occurence of tokens based on their weight in the vocab

    e.g. "low lower mid low"
    vocab_counts = {"l o w <EOW>": 2, "l o w e r <EOW>": 1, "m i d <EOW>": 1} 
    bigram = _get_bigram(vocab_counts)
    > bigram => {("l", "o"): 2, ("w" , <EOW>): 1, ("e", "r"): 1, ...} 
    """
    pairs = Counter()
    for word_str, freq in vocab_counts.items():
        symbols = word_str.split()
        for l, r in zip(symbols, symbols[1:]):
            pairs[(l, r)] += freq
    return pairs


print(_get_bigram(vocab_counts={"l o w <EOW>": 2,
      "l o w e r <EOW>": 1, "m i d <EOW>": 1}))


def _merge_token(pair: tuple[str, str],
                 vocab_counts: Mapping[str, int]) -> dict[str, int]:
    """
    Merges all instances of `pair` in the vocab_counts and returns an updated vocab

    e.g. pair = ("l", "o")
    vocab_counts = {"l o w <EOW>": 2, "l o w e r <EOW>": 1, "m i d <EOW>": 1} 
    _merge_token = vocab_counts = {"lo w <EOW>": 2, "lo w e r <EOW>": 1, "m i d <EOW>": 1} 
    """
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab: dict[str, int] = {}
    for word_str, freq in vocab_counts.items():
        merged = p.sub("".join(pair), word_str)
        new_vocab[merged] = new_vocab.get(merged, 0) + freq
    return new_vocab


print(_merge_token(pair=("l", "o"), vocab_counts={
      "l o w <EOW>": 2, "l o w e r <EOW>": 1, "m i d <EOW>": 1}))


@dataclass
class BPETokenizer:
    merges: list[tuple[str, str]] = field(default_factory=list)
    merge_ranks: dict[tuple[str, str], int] = field(default_factory=dict)
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    specials: list[str] = field(default_factory=lambda: [UNK])

    def train(self,
              texts: Iterable[str],
              num_merges: int,
              ):
        """
        Learn BPE merges and merge ranks 
        """
        # 1) Build vocab_counts from corpus
        vocab_counts: Counter[str] = Counter()
        for text in texts:
            # Split text on punctuation/script change ["this", "is", "."]
            for w in segment_words(text):
                # Split word to individual characters and <EOW>
                # e.g. ["t", "h", "i", <EOW>] or ["你", “好”, <EOW>]
                sym_seq = " ".join(_word_to_symbols(w))
                vocab_counts[sym_seq] += 1

        # 2) Merge high frequency co-occuring characters
        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            bigram = _get_bigram(vocab_counts)
            if not bigram:
                # All words merged as tokens, no possible merges left
                break
            best = max(bigram, key=bigram.get)
            if bigram[best] < 1:
                # most frequent co-occuring tokens are likely not generalizeable
                break
            vocab_counts = _merge_token(best, vocab_counts)
            merges.append(best)

            print(merges)

        # Save merges and merge order for tokenization
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # 3) Create Token Vocabulary
        token_set = set()
        for word in vocab_counts.keys():
            token_set.update(word.split())

        for special_token in self.specials + [EOW]:
            token_set.add(special_token)

        # 4) Add token to id mapping
        token_list = sorted(token_set, key=lambda t: (len(t), t))
        self.token_to_id = {token: idx for idx, token in enumerate(token_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(token_list)}


tokenizer = BPETokenizer()
tokenizer.train(
    texts=["low lower", "lowest 我的天", "我的好吃的蘋果"], num_merges=5
)
