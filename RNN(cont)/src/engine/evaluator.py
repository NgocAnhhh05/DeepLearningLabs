import torch
from rouge_score import rouge_scorer

class Evaluator:
    """
    Handles translation inference and ROUGE-L score calculation.
    """
    def __init__(self, model, device, tgt_vocab):
        self.model = model
        self.device = device
        self.tgt_vocab = tgt_vocab
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def translate(self, src_tensor, max_len=50):
        self.model.eval()
        with torch.no_grad():
            hidden, cell = self.model.encoder(src_tensor)

        # Start with <SOS>
        batch_size = src_tensor.shape[0]
        input = torch.tensor([self.tgt_vocab.word2idx["<SOS>"]] * batch_size).to(self.device)

        translated_indices = []
        for _ in range(max_len):
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(input, hidden, cell)

            input = output.argmax(1)
            translated_indices.append(input.cpu().numpy())

        # Convert to list of sentences
        translated_indices = list(zip(*translated_indices)) # Transpose
        results = [self.tgt_vocab.decode(idx) for idx in translated_indices]
        return [" ".join(res) for res in results]

    def calculate_rouge(self, preds, targets):
        scores = []
        for p, t in zip(preds, targets):
            s = self.scorer.score(t, p)
            scores.append(s['rougeL'].fmeasure)
        return sum(scores) / len(scores) if scores else 0