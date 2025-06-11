"""
Utility functions for the TinyLM project.

This module contains helper functions that can be reused across different notebooks
and parts of the project.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention_heatmap(attention: torch.Tensor, words=None):
    """
    Plot a heatmap visualization of attention weights.
    
    Args:
        attention (torch.Tensor): A 2D tensor containing attention weights
        words (list, optional): List of words/tokens for axis labels. 
                               If None, uses generic "Token i" labels.
    
    Returns:
        None: Displays the plot
    """
    if words is None:
        words = [f"Token {i}" for i in range(attention.size(0))]

    attention_np = attention.detach().cpu().numpy()

    fig, ax = plt.subplots()
    cax = ax.matshow(attention_np, cmap='Oranges')
    plt.colorbar(cax)

    # Set axis ticks and labels
    ax.set_xticks(np.arange(len(words)))
    ax.set_yticks(np.arange(len(words)))
    ax.set_xticklabels(words, rotation=90)
    ax.set_yticklabels(words)

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(attention_np):
        ax.text(j, i, f"{val:.2f}", va='center', ha='center')

    plt.title("Self-Attention Weights", pad=20)
    plt.tight_layout()
    plt.show()
