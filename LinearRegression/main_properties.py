import sys

import regression

# Check command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <csvfile>")
    exit(-1)

# Get the command line argument
infilename = sys.argv[1]

# Read in the training data and labels
X, y, labels = regression.read_csv_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Find the coefficients for the linear regression using matrix inversion
print("*** Linear Regression Using Matrix Inversion ***")
b = regression.matrix_inversion_fit(X, y)
print(regression.format_prediction(b, labels))
R2 = regression.score(b, X, y)
print(f"R2 = {R2:f}")

# Find the coefficients for the linear regression using gradient desent
print("*** Linear Regression Using Gradient Descent with explicit differentiation ***")
b = regression.gradient_descent_fit(X, y)
print(regression.format_prediction(b, labels))
R2 = regression.score(b, X, y)
print(f"R2 = {R2:f}")

# Find the coefficients for the linear regression using gradient desent
print("*** Linear Regression Using Gradient Descent with autodiff ***")
b = regression.autodiff_descent_fit(X, y)
print(regression.format_prediction(b, labels))
R2 = regression.score(b, X, y)
print(f"R2 = {R2:f}")
