from collections import Counter, defaultdict
import re


def init_english_vocab(sentence: str) -> list[str]:
    """
    Splits a string on whitespace and appends `</w>` to end
    """
    # 1) Split string on whitespace;
    # ex "the happy happy cat" => ["t h e </w>", "h a p p y </w>", ...]
    vocab = [" ".join(word) + " </w>" for word in sentence.split()]
    return vocab


def init_chinese_vocab(sentence: str) -> list[str]:
    """
    Finds all chinese characters in a string breaking on punctuation/non-chinese words and appends `</w>` to the end`
    """
    CJK_UNICODE_RANGE = r"\u4E00-\u9FFF"
    CJK_PATTERN = re.compile(f"[{CJK_UNICODE_RANGE}]+")
    chinese_words = CJK_PATTERN.findall(sentence)

    vocab = [" ".join(word) + " </w>" for word in chinese_words]
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


def train_tokenizer(num_merges: int, vocab: str):
    for _ in range(num_merges):
        bigram = count_cooccurence(vocab)
        best_bigram = max(bigram, key=bigram.get)

        vocab = create_merged_vocab(to_merge=best_bigram, orig_vocab=vocab)
        print(vocab)


english_vocab = init_english_vocab("lower farther wider far low widow")
chinese_vocab = init_chinese_vocab("高麗菜的價格真的低迷到極點，大家都這麼努力消化它了，價格依然還是在低點，曬成高麗菜乾吧，收藏起來可以吃蠻久的，很多人對於高麗菜乾怎麼烹煮有疑慮 ，今天ching就用高麗菜乾來煮古早味的菜湯，加上排骨和薑，搭配白胡椒粉和鹽巴的調味，簡單煮就能吃出湯的美味。")
# TODO: Tokenizer encoding/decoding
# TODO: OOV, punctuation on real data (Aya Collection)

train_tokenizer(num_merges=10, vocab=english_vocab)
print(30*"=")
train_tokenizer(num_merges=10, vocab=chinese_vocab)