import matplotlib.pyplot as plt
import numpy as np

# plot the loss curve from train_log.txt
# format is "  step    500 | val_loss 3.7193"

def plot_loss_curve(log_file):
    steps = []
    val_losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'val_loss' in line:
                parts = line.split('|')
                step_part = parts[0].strip()
                val_loss_part = parts[1].strip()
                step = int(step_part.split()[1]) * 20736 # tokens per step
                val_loss = float(val_loss_part.split()[1])
                steps.append(step)
                val_losses.append(val_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, val_losses)
    plt.title('Validation Loss Curve')
    plt.xlabel('Tokens Processed')
    plt.ylabel('Validation Loss')
    plt.grid()
    plt.savefig(log_file.replace('.txt', '.png'))
    plt.show()
    # save the plot as loss_curve.png

if __name__ == "__main__":
    # parse the name of the log file from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot loss curve from log file')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    args = parser.parse_args()
    plot_loss_curve(args.log_file)