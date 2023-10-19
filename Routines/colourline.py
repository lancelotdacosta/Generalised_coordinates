import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_cool(x, y, lw=0.5,alpha=1):
    d= np.arange(len(x))
    c= cm.cool_r((d - np.min(d)) / (np.max(d) - np.min(d)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw,alpha=alpha)
    plt.show()
    return

def plot_cmap(x, y, cmap=cm.get_cmap('binary'), lw=0.5,alpha=1,crev=False):
    l= np.arange(len(x))
    c= cmap((l - np.min(l)) / (np.max(l) - np.min(l)))
    if crev:
        c=np.flip(c,axis=0) #reverse colourmap
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw,alpha=alpha)
    plt.show()
    return