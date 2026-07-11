"""
Part 1: KNN from scratch
Run: python part1_knn_scratch.py
This will:
 - load the HAR dataset (expecting the "UCI HAR Dataset" folder in the current dir)
 - run the scratch KNN from util.py for k = 4,6
 - display the confusion matrix and accuracy
"""
import time
from utils import load_har_dataset, ScratchKNN, display_confusion_matrix_and_accuracy

def main():
    print("Part 1: KNN from scratch")
    
    #Your code goes here

    # 1. First, we need to load the dataset.
    X_train, y_train, X_test, y_test = load_har_dataset(base_path="UCI HAR Dataset")

    # 2. Test the classifier on k = 4, 6 
    k = [4,6]
    for val in k:
        # a. Instantiate
        classifier = ScratchKNN(val)

        # b. Fit to the training data
        classifier.fit(X_train, y_train)

        # c. Predict
        y_pred = classifier.predict(X_test)

        #  d. Display metrics 
        display_confusion_matrix_and_accuracy(val, y_test, y_pred)

if __name__ == '__main__':
    main()
