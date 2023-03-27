import numpy as np
import pandas as pd
import json
import hashlib


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


d = json.load(open("out/model/T4/4.json", "r", encoding='UTF-8'))
sha256sum("out/model/T4/4.json")

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import auc


def get_evaluation(label: list, pred: list, pro_cutoff: float = None):
    fpr, tpr, thresholds = roc_curve(label, pred)
    if pro_cutoff is None:
        best_one_optimal_idx = np.argmax(tpr - fpr)
        pro_cutoff = thresholds[best_one_optimal_idx]
    pred_l = [1 if i >= pro_cutoff else 0 for i in pred]
    confusion_matrix_1d = confusion_matrix(label, pred_l).ravel()
    confusion_dict = {N: n for N, n in zip(['tn', 'fp', 'fn', 'tp'], list(
        confusion_matrix_1d * 2 / np.sum(confusion_matrix_1d)))}
    evaluation = {
        "accuracy": accuracy_score(label, pred_l),
        "precision": precision_score(label, pred_l),
        "f1_score": f1_score(label, pred_l),
        "mmc": matthews_corrcoef(label, pred_l),
        "auc": auc(fpr, tpr),
        "specificity": confusion_dict['tn'] / (confusion_dict['tn'] + confusion_dict['fp']),
        "sensitivity": confusion_dict['tp'] / (confusion_dict['tp'] + confusion_dict['fn']),
        "confusion_matrix": confusion_dict,
        "_roc_Data": {'fpr': list(fpr), 'tpr': list(tpr)},
        'pro_cutoff': pro_cutoff
    }

    return evaluation

    import pandas as pd


df = pd.DataFrame({
    "pred": d['test_pred'],
    'target': d['test_target'],
})
performance = get_evaluation(
    label=d['test_target'],
    pred=d['test_pred'],
)
performance.pop('_roc_Data')
performance.pop('confusion_matrix')
print(json.dumps(performance, indent=4))
