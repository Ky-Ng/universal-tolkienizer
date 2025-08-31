import re
import string

CJK_RANGE = r"\u4E00-\u9FFF"
SEGMENT = re.compile(fr"[{CJK_RANGE}]+|[^\s{CJK_RANGE}{string.punctuation}]+|[{string.punctuation}]+")
SPACE = re.compile(r"\s+")

# Helper Functions for Sennrich et al.

def segment_words(text:str) -> list[str]:
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