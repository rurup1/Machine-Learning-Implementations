from importlib import import_module
import sys
import os
import jax.numpy as jnp

# Check number of command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <probfile> [long]")
    sys.exit(1)

# Constants
LETTER_STR_LEN = 8

# Read the command line args
probpath = sys.argv[1]

if len(sys.argv) == 2:
    classname = "Belief"
else:
    classname = "LongBelief"

# Open the problem file
infile = open(probpath, "r")

# Read the first line
numstrings = infile.readline().split(" ")
die_count = int(numstrings[0])
die_sides = int(numstrings[1])
round_count = int(numstrings[2])

# Let the user know what we are doing
print(
    f"{classname} analyzing problem with {die_count} dice, each with {die_sides} sides."
)

# Load the module and find the class
belief_module = __import__(classname)
belief_class = getattr(belief_module, classname)

# Instantiate the class
belief_analyzer = belief_class(die_count, die_sides)

# Go through the file line by line gathering the results in list
results = []
for _ in range(round_count):
    # Read the line from the file as a string
    letter_string = infile.readline().strip()

    # Convert the string to a list of letters
    letters = list(letter_string)

    # To make it pretty, truncate long strings in the output
    if len(letter_string) <= LETTER_STR_LEN:
        letter_label = letter_string
    else:
        letter_label = letter_string[: LETTER_STR_LEN - 3] + "..."

    # Run the analysis
    posterior = belief_analyzer.analyze(letters)

    # Sanity checks
    assert posterior.shape == (
        die_count * die_sides + 1,
    ), f"Posterior shape mismatch:{posterior.shape} != (die_count * die_sides + 1)"
    assert jnp.all(
        posterior >= 0.0
    ), f"Posterior probabilities must be non-negative:{letter_label}, {posterior}"
    assert jnp.isclose(
        1.0, jnp.sum(posterior), atol=1e-6
    ), f"Posterior probabilities must sum to 1:{letter_label}, {posterior}"

    # Show the results to the user
    belief_analyzer.show_array(posterior, f"P(d|{letter_label})")

    # Put them in the results array
    results.append(posterior)

# Always be closing
infile.close()

# Where is the solution file?
probdir = os.path.dirname(probpath)
probbase = os.path.basename(probpath)
solutionpath = os.path.join(probdir, "sol" + probbase.removeprefix("dice"))

# Does it exist?
if os.path.exists(solutionpath):
    # Open the solution file
    solutionfile = open(solutionpath, "r")
    for i in range(len(results)):
        # The line as a list of strings
        float_strings = solutionfile.readline().strip().split(" ")

        # Convert the strings to floats and put them in the solution array
        solution = [0.0] * die_count
        for j in range(len(float_strings)):
            solution.append(float(float_strings[j]))

        # Check if the solution matches the results
        jsolution = jnp.array(solution)

        assert jnp.all(
            jnp.isclose(jsolution, results[i], atol=1e-6)
        ), f"Solution mismatch for row {i}: {solution} != {results[i]}"
    print(f"PASSED: Matched solution in {solutionpath}")
else:
    # This is for the instructor to easily create missing solution files
    outfile = open(solutionpath, "w")
    for result in results:
        for i in range(die_count, die_count * die_sides + 1):
            print(f"{result[i]:e} ", file=outfile, end="")
        _ = outfile.write("\n")
    outfile.close()
    print(f"Solution created in {solutionpath}")