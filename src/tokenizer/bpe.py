from collections import Counter, defaultdict
import re

def init_vocab(sentence: str) -> list[str]:
    """
    Splits a string on whitespace and appends `</w>` to end
    """
    # 1) Split string on whitespace;
    # ex "the happy happy cat" => ["t h e </w>", "h a p p y </w>", ...]
    vocab = [" ".join(word) + " </w>" for word in sentence.split()]
    return vocab


def count_cooccurence(vocab: list[str]) -> dict[tuple[str, str], int]:
    """
    Calculates biagram of each word in the vocab weighted by its frequency
    """
    # ["t h e </w>", "h a p p y </w>", ...] => {"t h e </w>": 1, "h a p p y </w>": 2, "c a t </w>": 1}
    word_freq = Counter(vocab)
    cooccurence_freq = defaultdict(int)

    for word, freq in word_freq.items():
        symbols = word.split()
        for l, r in zip(symbols, symbols[1:]):
            cooccurence_freq[(l, r)] += freq
    return cooccurence_freq


def create_merged_vocab(to_merge: tuple[str, str], orig_vocab: list[str]):
    """
    Merges the substring `to_merge` into a single token and updates the vocab accordingly
    """
    # Prepare regex to find and merge
    new_token = re.escape(" ".join(to_merge))
    p = re.compile(r"(?<!\S)" + new_token + r"(?!\S)")

    # Create new vocab
    new_vocab = []

    for word in orig_vocab:
        new_vocab .append(p.sub("".join(to_merge), word))
    return new_vocab


vocab = init_vocab("the happy happy cat")

bigram = count_cooccurence(vocab)
best_bigram = max(bigram, key=bigram.get)

vocab = create_merged_vocab(to_merge=best_bigram, orig_vocab=vocab)
print(vocab)