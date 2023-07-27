from os import listdir
from os.path import join, abspath, dirname, isfile
import json
from pprint import pprint
import operator


fn = dirname( __file__ )

for dataset in ["RRecipe","WoT"]:
    fn_d = join(fn, dataset)
    reports = [f for f in listdir(fn_d)]
    print("___ Dataset: " + dataset + " ___")
    rank_accuracy, rank_f1_score_weighted, rank_f1_score_micro, rank_precision, rank_recall, rank_hamming_loss = \
    dict(), dict(), dict(), dict(), dict(), dict()
    for report in reports:        
        metrics_fn = [f for f in listdir(join(fn_d, report)) if "metrics" in f and "training" not in f]
        if dataset == "RRecipe":
            if "few_shot" in metrics_fn[0]:
                continue
        elif dataset == "WoT":
            if "sampled" in metrics_fn[0]:
                continue
        with open(join(fn_d, report, metrics_fn[0]), 'r') as ja:
            metrics = json.load(ja)        
        rank_accuracy[report] = metrics["test_Accuracy"]
        rank_f1_score_weighted[report] = metrics["test_F1 Score Weighted"]
        rank_f1_score_micro[report] = metrics["test_F1 Score Micro"]
        rank_precision[report] = metrics["test_Precision"]
        rank_recall[report] = metrics["test_Recall"]
        rank_hamming_loss[report] = metrics["test_Hamming loss"]
    print("Accuracy:")
    for k, v in dict(sorted(rank_accuracy.items(), key=operator.itemgetter(1), reverse=True)).items():
        print(k.ljust(32) + str(v))
    print("\nF1_score_weighted:")
    for k, v in dict(sorted(rank_f1_score_weighted.items(), key=lambda item: item[1], reverse=True)).items():
        print(k.ljust(32) + str(v))
    print("\nF1_score_micro:")
    for k, v in dict(sorted(rank_f1_score_micro.items(), key=lambda item: item[1], reverse=True)).items():
        print(k.ljust(32) + str(v))
    print("\nPrecision:")
    for k, v in dict(sorted(rank_precision.items(), key=lambda item: item[1], reverse=True)).items():
        print(k.ljust(32) + str(v))
    print("\nRecall:")
    for k, v in dict(sorted(rank_recall.items(), key=lambda item: item[1], reverse=True)).items():
        print(k.ljust(32) + str(v))
    print("\nHaming loss:")
    for k, v in dict(sorted(rank_hamming_loss.items(), key=lambda item: item[1], reverse=False)).items():
        print(k.ljust(32) + str(v))
    print()