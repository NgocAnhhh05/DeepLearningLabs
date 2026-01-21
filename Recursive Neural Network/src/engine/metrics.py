"""
Define f1 evaluate for these assingments
"""
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seq_f1_score

class MetricManager:
    "Manage metrics calcualtion for Classification and NER tasks"
    @staticmethod
    def calculated_f1(preds, targets, task_type='classification', idx2label=None):
        if task_type == 'classification':
            return f1_score(targets, preds, average='weighted')
        elif task_type == "ner":
            # Convert indices to label strings for seqeval
            # seqeval expects List[List[str]]
            true_tags = [[idx2label[t] for t in sent if t != -100] for sent in targets]
            pred_tags = []
            for i, sent in enumerate(preds):
                p_tags = [idx2label[p] for j, p in enumerate(sent) if idx2label[i][j] != -100]
                pred_tags.append(p_tags)
            return seq_f1_score(true_tags, pred_tags)
        else:
            raise ValueError("Task must be neither 'classification' nor 'ner'")