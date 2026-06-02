import glob
import subprocess

# Do the small problems with Belief first
problem_paths = glob.glob("problems/dice*small.txt")
problem_paths.sort()
for problem_path in problem_paths:
    print(f"Running {problem_path}")
    subprocess.run(["python3", "dice-main.py", problem_path])

# Then do all the problems with LongBelief
problem_paths = glob.glob("problems/dice*.txt")
problem_paths.sort()
for problem_path in problem_paths:
    print(f"Running {problem_path}")
    subprocess.run(["python3", "dice-main.py", problem_path, "long"])

# Finally do all the problems with coinss
problem_paths = glob.glob("problems/coin*.txt")
problem_paths.sort()
for problem_path in problem_paths:
    print(f"Running {problem_path}")
    subprocess.run(["python3", "coin-main.py", problem_path])
