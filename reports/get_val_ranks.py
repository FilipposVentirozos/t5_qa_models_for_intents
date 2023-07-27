from os import listdir
from os.path import join, abspath, dirname, isfile
import json
from pprint import pprint
import operator

fn = join(abspath(join(dirname( __file__ ), ".." ,"..")), "models")


for dataset in ["RRecipe","WoT"]:
    fn_d = join(fn, dataset)
    models = [f for f in listdir(fn_d)]
    print("___ Dataset: " + dataset + " ___")
    rank_accuracy, rank_f1_score_weighted, rank_f1_score_micro, rank_precision, rank_recall, rank_hamming_loss = \
    dict(), dict(), dict(), dict(), dict(), dict()
    for model in models:        
        with open(join(fn_d, model, "eval_training_metrics.json"), 'r') as ja:
            metrics = json.load(ja)        
        rank_accuracy[model] = metrics[metrics[-1]["max of: Accuracy"] - 1]["Accuracy"]
        rank_f1_score_weighted[model] = metrics[metrics[-1]["max of: F1 Score Weighted"] - 1]["F1 Score Weighted"]
        rank_f1_score_micro[model] = metrics[metrics[-1]["max of: F1 Score Micro"] - 1]["F1 Score Micro"]
        rank_precision[model] = metrics[metrics[-1]["max of: Precision"] - 1]["Precision"]
        rank_recall[model] = metrics[metrics[-1]["max of: Recall"] - 1]["Recall"]
        rank_hamming_loss[model] = metrics[metrics[-1]["min of: Hamming loss"] - 1]["Hamming loss"]
    print("Accuracy:")
    pprint(dict(sorted(rank_accuracy.items(), key=operator.itemgetter(1), reverse=True)), sort_dicts=False)
    print("F1_score_weighted:")
    pprint(dict(sorted(rank_f1_score_weighted.items(), key=lambda item: item[1], reverse=True)), sort_dicts=False)
    print("F1_score_micro:")
    pprint(dict(sorted(rank_f1_score_micro.items(), key=lambda item: item[1], reverse=True)), sort_dicts=False)
    print("Precision:")
    pprint(dict(sorted(rank_precision.items(), key=lambda item: item[1], reverse=True)), sort_dicts=False)
    print("Recall:")
    pprint(dict(sorted(rank_recall.items(), key=lambda item: item[1], reverse=True)), sort_dicts=False)
    print("Haming loss:")
    pprint(dict(sorted(rank_hamming_loss.items(), key=lambda item: item[1], reverse=False)), sort_dicts=False)
    print()

