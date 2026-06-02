import csv
import itertools
import evaluator

# Open a CSV reader on the predictions file and skip the header
prediction_file = open("data/predictions_validate.csv", "r")
prediction_reader = csv.reader(prediction_file)
next(prediction_reader)

# Open a CSV reader on the label file and skip the header
truth_file = open("data/labels_validate.csv", "r")
truth_reader = csv.reader(truth_file)
next(truth_reader)

# What are all the (predicted, truth) combinations?
count_keys = itertools.product(["depots", "laundromats"], repeat=2)

# (prediction, actual) -> occurences
count_dict = {}
for k in count_keys:
    count_dict[k] = 0

# Count them all (and gather probabilities in a list)
for pred_row in prediction_reader:
    truth_row = next(truth_reader)

    # Just a sanity check
    pshop_id = int(pred_row[0])
    tshop_id = int(truth_row[0])
    if pshop_id != tshop_id:
        print("Error: {pshop_id} != {cshop_id}")
        continue

    # Increment entry in count dict
    prediction = pred_row[2]
    truth = truth_row[1]
    count_dict[(prediction, truth)] += 1

# Print and sum
print("Data for confusion matrix:")
for (p, t), n in count_dict.items():
    print(f'\tPredicted "{p}", Truth "{t}": {n}')

stats = evaluator.gather_stats(count_dict)

# Print the stats
print(f"Total data points: {stats['n']}")
print(f"Accuracy: {stats['accuracy'] * 100.0:.1f}%")
print(f"Recall of depots: {stats['recall'] * 100.0:.1f}%")
print(f"Precision of depots: {stats['precision'] * 100.0:.1f}%")
print(f"F1: {stats['f1']:.3f}")
