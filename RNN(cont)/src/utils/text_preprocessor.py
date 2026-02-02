from pyvi import ViTokenizer

class TextProcessor:
    def __init__(self):
        pass

    def preprocess(self, text):
        text = text.lower().strip()
        text = " ".join(text.split())
        text = ViTokenizer.tokenize(text)
        return text

if __name__ == "__main__":
    processor = TextProcessor()
    sample =  "On August 14th , 1947 , a woman in Bombay goes into labor as the clock ticks towards midnight."
    print(f"Original: {sample}")
    print(f"Processed: {processor.preprocess(sample)}")