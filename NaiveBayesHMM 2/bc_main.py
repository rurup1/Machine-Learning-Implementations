import numpy as np
import csv
from make_parameters import make_parameters
from Classifier import Classifier

# The classes and their priors
classes = ["depots", "laundromats"]
priors = [0.2, 1 - 0.2]

# Read the file and return the numpy arrays representing
# each column
def read_csv(filename) -> tuple:
    with open(filename, "r") as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader)

        # Gather the data in lists
        areas_list = []
        visitors1_list = []
        visitors2_list = []
        felons_list = []
        for row in reader:
            areas_list.append(float(row[1]))
            visitors1_list.append(int(row[2]))
            visitors2_list.append(int(row[3]))
            if int(row[4]) == 1:
                felons_list.append(True)
            else:
                felons_list.append(False)

        # Convert to numpy arrays
        areas = np.array(areas_list, dtype=np.float64)
        visitors1 = np.array(visitors1_list, dtype=np.int32)
        visitors2 = np.array(visitors2_list, dtype=np.int32)
        felons = np.array(felons_list, dtype=bool)

    return (areas, visitors1, visitors2, felons)


# Read in the data and create the parameters
parameters = []
for c in classes:
    columns = read_csv(f"data/{c}.csv")
    class_parameters = make_parameters(*columns)
    print(f"For {c}:")
    for key, value in class_parameters.items():
        print(f"\t{key:>9} {value:f}")
    parameters.append(class_parameters)

# Make a classifier
classifier = Classifier(priors, parameters)


# Open the output file
outpath = f"data/predictions_validate.csv"
outfile = open(outpath, "w")
print("shop_id,pdepot,prediction", file=outfile)

# Step through the unknowns
inpath = f"data/unlabeled_validate.csv"
print(f"Reading {inpath}")
with open(inpath, "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        # Get the data in the row in the right types
        shop_id = int(row[0])
        area = float(row[1])
        v1 = int(row[2])
        v2 = int(row[3])
        felon = int(row[4]) == 1

        # Compute the probability
        ps = classifier.predict((area, v1, v2, felon))

        # Threshold at 50% confidence
        p = ps[0]
        if p > 0.5:
            prediction = "depots"
        else:
            prediction = "laundromats"

        print(f"{shop_id},{p:.4f},{prediction}", file=outfile)

outfile.close()
print(f"Wrote {outpath}.")
