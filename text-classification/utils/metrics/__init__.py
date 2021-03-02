# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2020 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Text classification tasks evaluation utils
"""

from sklearn.metrics import f1_score, precision_score, recall_score


SENTIMENT_LABELS = ["positive", "negative", "neutral"]

POETRY_LABELS = ["شعر حر", "شعر التفعيلة", "عامي", "موشح", "الرجز",
                 "الرمل", "الهزج", "البسيط", "الخفيف", "السريع",
                 "الطويل", "الكامل", "المجتث", "المديد", "الوافر",
                 "الدوبيت", "السلسلة", "المضارع", "المقتضب", "المنسرح",
                 "المتدارك", "المتقارب", "المواليا"]

MADAR_26_LABELS = ["KHA", "TUN", "MOS", "CAI", "BAG", "ALE", "DOH", "ALX",
                   "SAN", "BAS", "TRI", "ALG", "MSA", "FES", "BEN", "SAL",
                   "JER", "BEI", "SFX", "MUS", "JED", "RIY", "RAB", "DAM",
                   "ASW", "AMM"]

MADAR_6_LABELS = ["TUN", "CAI", "DOH", "MSA", "BEI", "RAB"]

MADAR_TWITTER_LABELS = ["Algeria", "Bahrain", "Djibouti", "Egypt", "Iraq",
                        "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania",
                        "Morocco", "Oman", "Palestine", "Qatar",
                        "Saudi_Arabia", "Somalia", "Sudan", "Syria",
                        "Tunisia", "United_Arab_Emirates", "Yemen"]

NADI_COUNTRY_LABELS = ["Algeria", "Bahrain", "Djibouti", "Egypt",
                       "Iraq", "Jordan", "Kuwait", "Lebanon",
                       "Libya", "Mauritania", "Morocco", "Oman",
                       "Palestine", "Qatar", "Saudi_Arabia",
                       "Somalia", "Sudan", "Syria", "Tunisia",
                       "United_Arab_Emirates", "Yemen"]


def acc_and_f1_poetry(preds, labels):
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, average='macro')

    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def acc_and_f1_sentiment(preds, labels):
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average=None, labels=[0, 1, 2])
    precision = precision_score(y_true=labels, y_pred=preds, average=None, labels=[0, 1, 2])
    recall = recall_score(y_true=labels, y_pred=preds, average=None, labels=[0, 1, 2])

    f1_macro = float(f1[0] + f1[1] + f1[2]) / 3.0
    f1_pn = float(f1[0] + f1[1]) / 2.0
    precision_macro = float(precision[0] + precision[1] + precision[2]) / 3.0
    precision_pn = float(precision[0] + precision[1]) / 2.0
    recall_macro = float(recall[0] + recall[1] + recall[2]) / 3.0
    recall_pn = float(recall[0] + recall[1]) / 2.0

    return {
        "acc": acc,
        "f1": f1_macro,
        "f1_pn": f1_pn,
        "precision": precision_macro,
        "precision_pn": precision_pn,
        "recall": recall_macro,
        "recall_pn": recall_pn
    }

def acc_and_f1_DID(preds, labels, labels_str):
    # gold_labels = list(set(labels))
    # f1 = f1_score(labels, preds, labels=gold_labels, average=None) * 100
    # recall = recall_score(labels, preds, labels=gold_labels, average=None) * 100
    # precision = precision_score(labels, preds, labels=gold_labels, average=None) * 100
    # print(f1, flush=True)
    # print(recall, flush=True)
    # print(precision, flush=True)
    # print(list(labels), flush=True)
    # print(2 in list(labels), flush=True)
    # print(list(preds), flush=True)
    # print(2 in list(preds), flush=True)
    # print(gold_labels, flush=True)
    # individual_scores = {}
    # precisions = {}
    # recalls = {}
    # f_scores = {}

    # for x in gold_labels:
    #     precisions[labels_str[x]] = precision[x]
    #     recalls[labels_str[x]] = recall[x]
    #     f_scores[labels_str[x]] = f1[x]

    # individual_scores['INDIVIDUAL PRECISION SCORE'] = precisions
    # individual_scores['INDIVIDUAL RECALL SCORE'] = recalls
    # individual_scores['INDIVIDUAL F1 SCORE'] = f_scores

    ## computes overall scores (accuracy, f1, recall, precision)
    accuracy = (preds == labels).mean() * 100
    f1 = f1_score(labels, preds, average="macro") * 100
    recall = recall_score(labels, preds, average="macro") * 100
    precision = precision_score(labels, preds, average="macro") * 100

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": accuracy
        # "INDIVIDUAL SCORES": individual_scores
        }

def write_predictions(path_dir, task_name, preds):
    predictions_file = open(path_dir, mode='w')
    if task_name == "arabic_did_madar_twitter":
        for pred in preds:
            predictions_file.write(MADAR_TWITTER_LABELS[pred])
            predictions_file.write('\n')

    elif task_name == "arabic_did_nadi_country":
        for pred in preds:
            predictions_file.write(NADI_COUNTRY_LABELS[pred])
            predictions_file.write('\n')

    predictions_file.close()

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name == "arabic_sentiment":
        return acc_and_f1_sentiment(preds, labels)

    elif task_name == "arabic_poetry":
        return acc_and_f1_poetry(preds, labels)

    elif task_name == "arabic_did_madar_26":
        return acc_and_f1_DID(preds, labels, labels_str=MADAR_26_LABELS)

    elif task_name == "arabic_did_madar_6":
        return acc_and_f1_DID(preds, labels, labels_str=MADAR_6_LABELS)

    elif task_name == "arabic_did_madar_twitter":
        return acc_and_f1_DID(preds, labels, labels_str=MADAR_TWITTER_LABELS)

    elif task_name == "arabic_did_nadi_country":
        return acc_and_f1_DID(preds, labels, labels_str=NADI_COUNTRY_LABELS)

