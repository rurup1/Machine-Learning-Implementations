# I predicted house sale prices given property features in 3 ways: closed-form normal equation, gradient descent with hand-derived gradients, and then gradient descent with JAX's automatic differentiation, and then verified that all three converged to the same $`R^2`$.

---

## Table of Contents

---

- Overview
- The Dataset
- Background: How Linear Regression Works
- Matrix Inversion
- Background: What is Gradient Descent?
- Explicit Gradient Descent
- Gradient Descent using JAX autodiff
- Results
- What I Learned

---

## Overview

---

The problem was to estimate prices that a house will sell for. I have a CSV (recent_sales.csv) with features and prices that have been recently sold. The goal is to use the spreadsheet and linear regression to create a formula for predicting the sale of any house.

Running main_properties.py will read the CSV and make a formula that estimates house prices. It will create approximately the same formula three times: once using matrix inversion, once using explicit gradient descent, and once using autodiff. It will then print out the $`R^2`$ score for each.

---

## The Dataset

---

Our data is from a spreadsheet (recent_sales.csv). This file has features and prices of 519 houses. The features that we will use to predict prices are:

- sqft_hvac: Indoor square footage
- sqft_yard: Outdoor square footage
- bedrooms: Number of bedrooms in the house
- bathrooms: Number of bathrooms in the house
- miles_to_school: Number of miles children would need to walk to the nearest elementary school

---

## How Linear Regression Works

---

The goal of linear regression is: given some input X, we want to learn the features such that we can predict y with some line. I've seen this in middle school math before, but in ML, we want to *learn* the right line from the data.

For general inputs $`x \in \mathbb{R}^d`$, the linear model is:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \dots + \beta_dx_d = \beta^Tx
$$

In matrix form, this becomes:

$$
\hat{y} = X\beta
$$

A loss function assigns a scalar cost to the choice of parameters, which measures how far our predictions are from the true labels. The most natural idea is L1 Loss. For this project, I use L2 loss, which penalizes large errors disproportionally:

$$
L_2(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \|y - X\beta\|^2
$$

Notice that L2 Loss is differentiable. This will be important for gradient descent. But, with this in hand, recognize that we want to minimize the loss, so we want:

$$
\beta^* = \arg\min_{\beta} \|y - X\beta\|^2
$$

If we expand the squared norm, the unique solution is:

$$
\beta^* = (X^\top X)^{-1} X^\top y
$$

This is extremely important and remarkable. For linear regression, L2 loss we can solve the optimization problem exactly in closed form. But, this situation depends on the loss being quadratic in the parameters, which is a special structure which most models do not have. Therefore, we must resort to an iterative approach.

---

## Matrix Inversion

---

As stated earlier, I first used matrix operations to find the coefficients. This was done in matrix_inversion_fit().

Because of the identity in the normal equation, there is a simple one-line solution:

```python
return jnp.linalg.solve(X.T @ X, X.T @ y)
```

---

## Background: What is Gradient Descent?

---

Newton's method is a precursor to gradient descent. In 1D, it approximates $`f`$ near the current point $`x_t`$ by a quadratic (using both first and second derivatives) and jumps to the minimum of the quadratic:

$$
x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}
$$

Newton's method converges extremely quickly near a minimum, but computing the second-derivative matrix of a function with $`d`$ parameters is $`O(d^2)`$ to store and $`O(d^3)`$ to invert. For models with even a modest number of parameters, this becomes infeasible. The fix is just to drop the second derivative entirely using $`\alpha`$, a step size that we choose manually:

$$
x_{t+1} = x_t - \alpha f'(x_t)
$$

In multiple dimensions, $`f'`$ becomes the gradient $`\nabla_{\theta} L(\theta)`$, and we get gradient descent:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} L(\theta_t)
$$

The idea is to start from an initial guess $`\theta_0`$ and repeat until the gradient is sufficiently small. The scalar $`\alpha > 0`$ is the learning rate.

Because the gradient points in the direction of the steepest ascent, moving opposite to it descends the loss surface. We can think of the loss as a mountain range, and gradient descent is always walking downhill. For a perfectly bowl-shaped mountain (convex loss), every downhill walk leads to the bottom. On a real mountain range, we might end up in a valley that is not the lowest point, which is the case for non-convex loss.

For our linear regression, the gradient is: $`\nabla_{\beta} L = -2X^\top (y - X\beta)`$, so each step is:

$$
\beta_{t+1} = \beta_t + 2\alpha X^\top (y - X\beta_t)
$$

and gradient descent converges to the $`\beta^*`$ we found in closed form.

### Side Note

Gradient descent is simple but tricky in practice. There are three main challenges that arise:

1. Ill-Conditioning

   The gradient may be very informative in some parameter direction and uninformative in others. A loss landscape that is narrow and elongated causes the gradient to zigzag inefficiently rather than walk towards the minimum. The main fix is **feature standardization:** before training, rescale each feature to zero mean and unit standard deviation:

   $$
   x_i = \frac{x_i - \mu_i}{\sigma_i}
   $$

   Where $`\mu_i`$ and $`\sigma_i`$ are computed from the training set. We must apply the same $`\mu_i`$ and $`\sigma_i`$ at test time as well.

2. Computational cost

   Computing $`\nabla_{\theta} L(\theta)`$ exactly requires summing over every training example. When this is in the millions or billions, this is very expensive. The solution is **mini-batch gradient descent**, which randomly samples a small subset at each step and uses the mini-batch gradient as a noisy estimate of the true gradient.

3. Choosing a learning rate

   If the learning rate is too large, the update might overshoot the minimum, and parameters bounce back and forth and might just diverge entirely. Too small and training makes little progress per step, and takes an impractical amount of time.

---

## Explicit Gradient Descent

---

For this approach, I used gradient descent to minimize the squared error. The result is nearly the same as the result from matrix_inversion_fit.

The features are on very different scales, so converging would take too long if we do not standardize the features beforehand. This is simple, per the formula I gave earlier:

```python
Xp = (X - means) / std
```

However, we should not standardize the first column, as it is for the bias term. If we plug this into the formula, its std is 0, so we would divide by zero. The fix is to overwrite before computing so that the first column goes through untouched:

```python
means = means.at[0].set(0.0)
std = std.at[0].set(1.0)
```

Now that the features are standardized, I can iteratively use gradient descent to get a good estimate of the parameters. I started with all zeros for my first guess $`(\beta_0)`$. Then, I stopped when the change in the parameters gets very small.

So, my learning rate was 0.1. The change in parameters is defined by the gradient, so I stopped when the gradient was lower than some tolerance. In this case, I chose 1.0 as my tolerance.

Now, we need to unstandardize our data so we can make predictions on the raw data, as that is what our callers use, not the standardized data. After some expanding and sorting, we get that the new $`\beta`$ is:

$$
b_0 = b_0' - \sum_{j \ge 1} b_j' \frac{m_j}{s_j}
$$

We can finally use the unstandardized $`\beta`$ to make predictions!

---

## Gradient Descent using JAX autodiff

---

For explicit gradient descent, we had to manually run the calculus to find the gradient of the error. In modern machine learning, we rarely do this by hand. Instead, we use *automatic differentiation.* JAX provides a function called jax.grad. If we provide it a function that computes the loss, jax.grad will return a new function that computes the gradient of that loss.

Therefore, I used a helper loss function: loss_function to return the mean squared error. As shown earlier, it is simply:

```python
return jnp.mean((y - X @ beta) ** 2)
```

Then, we use grad_function = jax.grad(loss_function) to create a function that calculates the gradient, and then run the same descent loop as before, just using grad_function to find the gradient instead of doing the math myself.

---

## Results

---

I got that $`R^2 = 0.853819`$, identical across all three methods. This means that the five features linearly explain about 85% of the variation in house prices. The remaining 15% is noise that the model cannot capture, like condition, renovations, or other non-linear effects.
