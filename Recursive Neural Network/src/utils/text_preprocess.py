import re
from pyvi import ViTokenizer

class TextProcessor:
    def __init__(self):
         pass

    def preprocess(self, text):
        "Clean text and segment words"
        text = text.lower().strip()
        text = " ".join(text.split())
        text = ViTokenizer.tokenize(text)
        return text

if __name__ == "__main__":
    processor = TextProcessor()
    sample = "Slide giáo trình rất đầy đủ,10 điểm, giảng viên nhiệt tình! "
    print(f"Original: {sample}")
    print(f"Processed: {processor.preprocess(sample)}")