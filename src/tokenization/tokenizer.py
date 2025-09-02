from dataclasses import dataclass, field
import re
import string
from typing import Iterable, Mapping
from collections import Counter

CJK_RANGE = r"\u4E00-\u9FFF"
SEGMENT = re.compile(
    fr"[{CJK_RANGE}]+|[^\s{CJK_RANGE}{string.punctuation}]+|[{string.punctuation}]+")
SPACE = re.compile(r" +")

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


@dataclass
class BPETokenizer:
    merges: list[tuple[str, str]] = field(default_factory=list)
    merge_ranks: dict[tuple[str, str], int] = field(default_factory=dict)
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)
    specials: list[str] = field(default_factory=lambda: [UNK, EOW])

    def train(self,
              texts: Iterable[str],
              num_merges: int,
              ):
        """
        Learn BPE merges and merge ranks 
        """
        # 1) Build vocab_counts from corpus
        vocab_counts: Counter[str] = Counter()

        # Add all symbols in original vocab to generalize tokenization
        base_symbols: set[str] = set(self.specials)
        for text in texts:
            # Split text on punctuation/script change ["this", "is", "."]
            for w in segment_words(text):
                # Split word to individual characters and <EOW>
                # e.g. ["t", "h", "i", <EOW>] or ["你", “好”, <EOW>]
                symbols = _word_to_symbols(w)
                sym_seq = " ".join(_word_to_symbols(w))
                vocab_counts[sym_seq] += 1
                base_symbols.update(symbols)

        # 2) Merge high frequency co-occuring characters
        merges: list[tuple[str, str]] = []
        merged_tokens: set[str] = set()
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

            L, R = best
            merged_tokens.add(L+R)

        # Save merges and merge order for tokenization
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # 3) Create Token Vocabulary
        token_set = set()
        for word in vocab_counts.keys():
            token_set.update(word.split())

        # Add all intermediate merge tokens and base tokens
        token_set.update(base_symbols, merged_tokens)

        # 4) Add token to id mapping
        token_list = sorted(token_set, key=lambda t: (len(t), t))
        self.token_to_id = {token: idx for idx, token in enumerate(token_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(token_list)}

    def _encode_word(self, word: str) -> list[str]:
        """
        Converts word from a string to a list of token strings w/ <EOW> token

        Case: Priority Merge
        e.g. word="the"
        word => symbols = ["t", "h", "e", <EOW>] 
        let merge_ranks = {("t","h"): 0, ("h", "e"): 1}
        symbols = ["th", "e", <EOW>] # Note; "th" blocks the "he" merge

        Case: Empty merge_ranks
        e.g. word="the"
        word => symbols = ["t", "h", "e", <EOW>] 
        let merge_ranks = {}
        symbols unchanged since no words available

        Case: symbols unseen to merge
        e.g. word="the"
        word => symbols = ["t", "h", "e", <EOW>] 
        let merge_ranks = {("w", "e"): 4}
        symbols unchanged since unable to apply any merges
        """
        # 1) Map word to symbols
        symbols = _word_to_symbols(word)

        # 2) Check if learned merges are available
        if not self.merge_ranks:
            return symbols
        # 3) Apply merges; make copy of `symbols`
        # for safety in case _word_to_symbols returns an array
        # not intended to be mutated
        symbols = symbols[:]
        while True:
            # Generate all consecutive token pairs and look for tokens to merge
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]
            ranked_merges = [(self.merge_ranks[p], p, i)
                             for i, p in enumerate(pairs) if p in self.merge_ranks]

            if not ranked_merges:
                # No more possible merge matches, return the string as is and use fallbacks if needed
                break

            # find highest priority merge
            _, (L, R), i = min(ranked_merges, key=lambda x: x[0])
            symbols = symbols[:i] + [L+R] + symbols[2+i:]
        return symbols

    def encode(self, text: str) -> list[int]:
        """
        Convert continuous text to a list of corresponding token ids
        e.g. "lower to tomorrow" => ["lo", "wer", "<EOW>", "to", "<EOW>", "to", "mor", "row", "<EOW>"] => [3, 4, 90, 20, 90, 20, 13, 41, 32, 90]
        e.g. "今天很好" => ["今天 很 好", "<EOW>"] => [35, 90]
        """
        tokens = []
        # Split continuous text into segments on whitespace/punctuation
        for word in segment_words(text):
            # Split word into subtokens
            tokenized_word = self._encode_word(word)

            # Add tokens one-by-one
            for token in tokenized_word:
                tokens.append(token)

        # Turn each token into its corresponding ID
        ids: list[int] = []
        unk_id = self.token_to_id[UNK]
        for token in tokens:
            # <UNK> token fallback method
            token_id = self.token_to_id.get(token, unk_id)

            ids.append(token_id)

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Convert list of token ids into string; EOW turn into spaces
        """
        token_strs = [self.id_to_token[id] for id in ids]
        text = "".join(token_strs)
        text = text.replace(EOW, " ")
        return SPACE.sub(" ", text).strip()
