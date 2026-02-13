import pandas as pd
import matplotlib.pyplot as plt
import os

class MetricsTracker:
    def __init__(self, run_name, log_dir="logs"):
        self.run_name = run_name

        # root: .../ is three levels up from src/common/metrics.py
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        self.log_dir = os.path.join(src_dir, log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.data = []

    def log(self, epoch, metrics_dict):
        """
        Log a dictionary of metrics for a specific epoch.
        """
        metrics_dict["epoch"] = epoch
        self.data.append(metrics_dict)

    def save(self):
        """
        Save the collected metrics to a CSV file.
        """
        if not self.data: 
            return
        df = pd.DataFrame(self.data)
        filepath = os.path.join(self.log_dir, f"{self.run_name}_metrics.csv")
        df.to_csv(filepath, index=False)
        print(f"Metrics saved to {filepath}")

    def plot(self, x_label="Epoch"):
        """
        Generate and save separate plots for each tracked metric.
        """
        if not self.data: return
        df = pd.DataFrame(self.data)
        metrics = [c for c in df.columns if c != "epoch"]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df[metric], label=metric)
            plt.title(f"{self.run_name} - {metric}")
            plt.xlabel(x_label) #  variable x label
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            safe_metric_name = metric.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(self.log_dir, f"{safe_metric_name}.png"))
            plt.close()