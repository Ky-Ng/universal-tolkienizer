# minimal driver for tokenizer.py

from src.tokenization.tokenizer import BPETokenizer, _get_bigram, _merge_token, segment_words

# Test Helper functions

print(50*"=")
print("Testing Helper Functions:")

chinese_string = "今天好熱！但是因為裡面有空調，所以還好"
print(f"Segmenting chinese string | {chinese_string} |: ", segment_words(
    chinese_string))

english_string = "This sentence, is a demo!"
print(f"Segmenting english string | {english_string} |: ", segment_words(
    english_string))

vocab_counts = {
    "l o w <EOW>": 2,
    "l o w e r <EOW>": 1,
    "m i d <EOW>": 1
}

print("Getting bigram frequencies of weighted vocab",
      vocab_counts, ": \n\t", _get_bigram(vocab_counts))

pair = ("l", "o")
print(f"merging pair {pair}: \n\t", _merge_token(
    pair=pair, vocab_counts=vocab_counts))

print(50*"=")
print("Testing Tokenizer Class")

tokenizer = BPETokenizer()
texts = ["low lower", "lowest 我的天", "我的好吃的蘋果"]

print("Training tokenizer on dataset: \n\t", texts)
tokenizer.train(
    texts=texts, num_merges=5
)

print("tokenizing the word 'lottery'", tokenizer._encode_word(word="lottery"))
print("tokenizing the multilingual phrase 'low 我的 test:",
      tokenizer.encode("low 我的 est"))
print("decoding tokens for 'lottery':",
      tokenizer.decode(tokenizer.encode("lottery")))
