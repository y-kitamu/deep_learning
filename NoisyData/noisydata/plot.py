import pandas as pd
import matplotlib.pyplot as plt


def plot_csv(csv_filename):
    df = pd.read_csv(csv_filename)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].plot(df.epoch, df.train_loss, label="Training loss")
    axes[0].plot(df.epoch, df.val_loss, label="Validation loss")
    axes[0].legend()

    axes[1].plot(df.epoch, df.train_acc, label="Training Accuracy")
    axes[1].plot(df.epoch, df.val_acc, label="Validation Accuracy")
    axes[1].legend()

    axes[2].plot(df.epoch, df.lr, label="learning rate")
    axes[2].set_yscale('log')
