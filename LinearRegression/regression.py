import jax
import jax.numpy as jnp
import pandas as pd
from jax import grad, jit
from jax import Array
from jax.typing import ArrayLike
from time import perf_counter

# Read in the CSV file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_csv_data(infilename):
    df = pd.read_csv(infilename, index_col=0)
    n, d = df.values.shape
    d = d - 1  # The price column doesn't count
    X = df.values[:, :-1]
    labels = list(df.columns[:-1])
    y = jnp.array(df.values[:, -1])
    X = jnp.hstack([jnp.ones((n, 1)), X])
    return X, y, labels

# Returns a vector of weights
# X of shape (519, 6)
# y of shape (519)
def matrix_inversion_fit(X: ArrayLike, y: ArrayLike) -> Array:
    # y = Xbeta
    # normal equation
    return jnp.linalg.solve(X.T @ X, X.T @ y)


    

# Helper function to standardize data
def standardize(X: ArrayLike) -> tuple[Array, Array, Array]:
    # Returns:
    #   Xp: The standardized data
    #   means: The mean of each column
    #   std: The standard deviation of each column
    means = jnp.mean(X, axis=0)
    std = jnp.std(X, axis=0)

    # First column remains
    means = means.at[0].set(0.0)
    std = std.at[0].set(1.0)
    Xp = (X - means) / std

    return (Xp, means, std)

# Helper function to un-standardize coefficients
def params_for_unstandardized(beta: ArrayLike, means: ArrayLike, std: ArrayLike) -> Array:
    # for 0 < j <= d
    new_beta = beta / std
    # for 0
    shift = jnp.sum(beta * means / std)
    new_beta = new_beta.at[0].set(beta[0] - shift)
    return new_beta
    

# Returns a vector of weights
def gradient_descent_fit(X: ArrayLike, y: ArrayLike) -> Array:
    # Standardize the data (except for the first column!)
    # Get the mean and standard deviation for each column
    Xp, means, std = standardize(X)

    # Iteratively use gradient descent to get
    # a good estimate of the parameters
    # Start with all zeros as your first guess
    # Stop when the change in the parameters gets very small
    # Experiment to find a good learning rate
    # You should expect a few hundred iterations
    iter_count = 0
    start = perf_counter()

    learning_rate = 0.1
    beta = jnp.zeros(X.shape[1])
    while True:
        grad = (-2/X.shape[0]) * Xp.T @ (y - Xp @ beta)
        if jnp.linalg.norm(grad) < 1.0:
            break

        beta = beta + (-learning_rate) * grad
        iter_count += 1

    print(f"Took {iter_count} iterations to converge: {perf_counter() - start:.6f} seconds")

    # Un-standardize the coefficients before you return them
    return params_for_unstandardized(beta, means, std)

# Helper function for autodiff
@jax.jit
def loss_function(beta: ArrayLike, X: ArrayLike, y: ArrayLike) -> float:
    # Return the mean squared error
    residuals  = y - X @ beta
    return jnp.mean(residuals ** 2)

# Returns a vector of weights using JAX autodiff
def autodiff_descent_fit(X: ArrayLike, y: ArrayLike) -> Array:
    # Standardize the data
    Xp, means, std = standardize(X)
    # Have JAX create a function that returns the gradient of the loss function
    gradient_function = grad(loss_function)

    # Iteratively use gradient descent
    iter_count = 0
    start = perf_counter()
    
    beta = jnp.zeros(X.shape[1])
    learning_rate = 0.1
    while True:
        g = gradient_function(beta, Xp, y)
        if jnp.linalg.norm(g) < 1.0:
            break

        beta = beta + (-learning_rate) * g
        iter_count += 1


    print(f"Took {iter_count} iterations to converge: {perf_counter() - start:.6f} seconds")

    # Un-standardize
    return params_for_unstandardized(beta, means, std)

# Make it pretty
def format_prediction(beta, labels):
    str = f"predicted price = ${beta[0]:,.2f} + "
    d = len(labels)
    for i in range(d):
        b = beta[i + 1]
        label = labels[i]
        str += f"(${b:,.2f} x {label})"
        if i < d - 1:
            str += " + "
    return str

# Return the R2 score for coefficients B
# Given inputs X and outputs y
def score(beta, X, y):
    prediction = X @ beta
    numerator = jnp.sum((y - prediction) ** 2)
    y_mean = jnp.mean(y)
    denominator = jnp.sum((y - y_mean) ** 2)

    return 1 - numerator / denominator
