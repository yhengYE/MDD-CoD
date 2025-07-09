import json
import os
from DataProvider.config import TaskConfig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
config = TaskConfig()
def smooth_curve(values, smoothing_factor=0.6):

    smoothed_values = []
    for i, value in enumerate(values):
        if i == 0:
            smoothed_values.append(value) 
        else:
            smoothed_value = smoothing_factor * smoothed_values[-1] + (1 - smoothing_factor) * value
            smoothed_values.append(smoothed_value)
    return smoothed_values


def save_data(weights, auroc, epochs, filename='plot_data.json'):

    data = {
        'weights': weights,
        'auroc': auroc,
        'epochs': epochs
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data(filename='.\plot_data.json'):

    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def plot_weights_and_auroc(history, start_epoch=71, smoothing_factor=0.6, filename='CRC_plot_data.json'):

    if 'weights' not in history or 'all_label_auroc' not in history:
        print("Error: History data is incomplete.")
        return


    weights = history['weights'][start_epoch - 1:]  
    auroc = history['all_label_auroc'][start_epoch - 1:]  
    epochs = list(range(start_epoch, start_epoch + len(weights)))  


    save_data(weights, auroc, epochs, filename)

    weights_s = [w['s_Label'] for w in weights if w is not None]
    weights_i = [w['i_Label'] for w in weights if w is not None]
    weights_all = [w['all_Label'] for w in weights if w is not None]
    weights_s_smooth = smooth_curve(weights_s, smoothing_factor)
    weights_i_smooth = smooth_curve(weights_i, smoothing_factor)
    weights_all_smooth = smooth_curve(weights_all, smoothing_factor)

    auroc_smooth = smooth_curve(auroc, smoothing_factor)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Task Weights', color='tab:blue')
    ax1.plot(epochs, weights_s_smooth, label='s_Label Weight (smoothed)', color='tab:blue', linestyle='--')
    ax1.plot(epochs, weights_i_smooth, label='i_Label Weight (smoothed)', color='tab:cyan', linestyle='--')
    ax1.plot(epochs, weights_all_smooth, label='all_Label Weight (smoothed)', color='tab:pink', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('all_Label AUROC', color='tab:red')
    ax2.plot(epochs, auroc_smooth, label='all_Label AUROC (smoothed)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    plt.title(f'Task Weights and all_Label AUROC Over Epochs (Smoothed, from Epoch {start_epoch})')
    plt.show()




def plot_loss_curve(history, save_path="loss_curve.png"):

    train_losses = history.get('joint_train_losses', [])
    val_losses = history.get('joint_val_losses', [])

    if not train_losses or not val_losses:
        print("没有联合训练阶段的损失数据可绘制")
        return

    epochs = list(range(0, len(train_losses)))  

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')

    best_val_loss = min(val_losses)
    best_epoch_idx = val_losses.index(best_val_loss)
    best_epoch = epochs[best_epoch_idx]
    plt.plot(best_epoch, best_val_loss, 'ro', markersize=10,
             label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')

    plt.xlabel('Joint Training Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Joint Training Loss Curves\nTask: {config.task_name}', fontsize=16)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().set_facecolor('#f8f8f8')


    max_loss = max(max(train_losses), max(val_losses))
    min_loss = min(min(train_losses), min(val_losses))
    plt.ylim(min_loss * 0.9, max_loss * 1.1)


    plt.xlim(-0.5, len(epochs) - 0.5)  
    plt.xticks(np.arange(0, len(epochs), max(1, len(epochs) // 10)))  

    plt.legend(fontsize=12, loc='upper right')

    for i, (trn, val) in enumerate(zip(train_losses, val_losses)):
        if i == len(train_losses) - 1 or i % 5 == 0:  
            plt.text(i, trn, f'{trn:.2f}', fontsize=9, ha='center', va='bottom')
            plt.text(i, val, f'{val:.2f}', fontsize=9, ha='center', va='top')


    plt.tight_layout()
    plt.show()


