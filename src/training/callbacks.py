import matplotlib.pyplot as plt
import os
import numpy as np

class PlottingCallback:
    """Matplotlib-based visualizer for RL training curves."""
    def __init__(self, returns, model_dir):
        self.returns = returns
        self.model_dir = model_dir

    def save_plot(self):
        steps = np.arange(len(self.returns))
        plt.plot(steps, self.returns)
        plt.xlabel("Evaluation Step")
        plt.ylabel("Average Return")
        plt.title("Training Curve")
        path = os.path.join(self.model_dir, "training_curve.png")
        plt.savefig(path)
        plt.close()
