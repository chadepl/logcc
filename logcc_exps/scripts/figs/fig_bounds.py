import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    xs = np.linspace(0, 1, 1000)
    ys_lemma1 = np.cos(2*np.arccos(xs))
    ys_lemma2 = np.cos(np.arccos(xs) + 2*np.arccos(0.99))

    fig, ax = plt.subplots(figsize=(4,2))
    ax.plot(xs, ys_lemma1, linestyle="-", color="#FF7F0E", linewidth=2, label="lemma 1")
    ax.plot(xs, ys_lemma2, linestyle="-", color="#1F77B4", linewidth=2, label="lemma 2")
    ax.axvline(0.9602, linestyle="--", color="red")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(-0.5, 1)
    # plt.legend()
    # plt.show()
    fig.savefig("/Users/chadepl/Downloads/cor_bounds.png", dpi=300)