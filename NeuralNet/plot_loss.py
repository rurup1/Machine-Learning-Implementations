import matplotlib.pyplot as plt
import pickle

# Read the loss log from the pickle file
with open("losslog.pkl", "rb") as f:
    loss_log = pickle.load(f)

# Plot it
fig, axis = plt.subplots(figsize=(6, 6))
axis.plot(loss_log)
axis.set_title("Loss")
axis.set_xlabel("Epoch")
axis.set_ylabel("Mean Cross-Entropy Loss")
fig.savefig("loss.png")
fig.show()
