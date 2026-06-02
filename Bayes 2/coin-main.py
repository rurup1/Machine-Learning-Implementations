import sys
import os
import math
import matplotlib.pyplot as plt

from CoinEstimator import CoinEstimator

# Check number of command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <probfile>")
    sys.exit(-1)

# Read the command line arg
probpath = sys.argv[1]

# Open the problem file
infile = open(probpath, "r")

# Read the first line
cred = float(infile.readline())

# Read the experiments
experiments = []
for line in infile:
    numstrings = line.split(" ")
    if len(numstrings) < 2:
        break
    r = int(numstrings[0])
    k = int(numstrings[1])
    experiments.append((r, k))

print(f"Loaded {len(experiments)} experiments.")

# Instantiate the class
coin_estimator = CoinEstimator()

# Give it the experiments
coin_estimator.prepare_to_estimate(experiments)

# Ask it for the most probable value for theta
map = coin_estimator.max_a_posteriori()
print(f"Maximum a posteriori estimation of theta: {map:.4f}")

# Get the desired credibility interval
ci = coin_estimator.credibility_interval(cred)
print(f"{100 * cred}% credibility interval: {ci[0]:.4f} - {ci[1]:.4f}")

# For the plots, it is nice to know how high the curve will go
maxdensity = coin_estimator.probability_density(map)

# Graph the PDF
thetas = [x / 100 for x in range(100)]
densities = [coin_estimator.probability_density(x) for x in thetas]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(thetas, densities)
ax.set_title(f"PDF for {os.path.basename(probpath)}, {100 * cred}% CI")
ax.set_xlabel("$\\theta$")
ax.set_xlim(0, 1)
ax.set_ylabel("Probability Density")
ax.set_ylim(0, maxdensity * 1.05)
ax.vlines(map, 0, maxdensity, "r", linestyle="dashed", lw=0.5)
ax.vlines(ci[0], 0, maxdensity, "g", linestyle="dashed", lw=0.5)
ax.vlines(ci[1], 0, maxdensity, "g", linestyle="dashed", lw=0.5)
fig.savefig("pdf.png")

# Where is the solution file?
probdir = os.path.dirname(probpath)
probbase = os.path.basename(probpath)
solutionpath = os.path.join(probdir, "sol" + probbase.removeprefix("coin"))

# Does it exist?
if os.path.exists(solutionpath):
    good_enough = True
    # Open the solution file
    solutionfile = open(solutionpath, "r")
    solution_map = float(solutionfile.readline())
    if math.fabs(solution_map - map) > 5e-3:
        good_enough = False
        print(f"INCORRECT: Maximum a posteriori estimation: Yours {map} != Solution {solution_map}")

    ci_strings = solutionfile.readline().strip().split(" ")
    solution_ci = [float(x) for x in ci_strings]

    if math.fabs(solution_ci[0] - ci[0]) > 5e-3 or math.fabs(solution_ci[1] - ci[1]) > 5e-3:
        good_enough = False
        print(f"INCORRECT: Credible interval: Yours [{ci[0]}, {ci[1]}] != Solution [{solution_ci[0]}, {solution_ci[1]}]")
    if good_enough:
        print("CORRECT")

else:
    print("Solution file not found")
