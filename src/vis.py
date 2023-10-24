# visualisation related functions

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from umap import UMAP
import os
from matplotlib.colors import ListedColormap
from typing import List

def custom_cmap(n):
    """Create customised colormap for scattered latent plot of n categories.
    Returns colormap object and colormap array that contains the RGB value of the colors.
    See official matplotlib document for colormap reference:
    https://matplotlib.org/examples/color/colormaps_reference.html
    """
    # first color is grey from Set1, rest other sensible categorical colourmap
    cmap_array = sns.color_palette("Set1", 9)[-1:] + sns.husl_palette(n - 1, h=.6, s=0.7)
    cmap = colors.LinearSegmentedColormap.from_list('mmdgm_cmap', cmap_array)
    return cmap, cmap_array

def hamed_cmap(n):
    # cmap_array = ['#ff00ff', '#ffff00', '#990000', '#00ffff', '#0000ff',
    #                     '#black', '#009900', '#999900', '#00ff00', '#009999'][0:n]
    cmap_array = ['#5c5cd6', '#cc0033', '#80ff00'][0:n]
    cmap = ListedColormap(cmap_array)
    return cmap, cmap_array
    
def embed_umap(data):
    """data should be on cpu, numpy"""
    embedding = UMAP(metric='euclidean',
                     n_neighbors=40,
                     # angular_rp_forest=True,
                     # random_state=torch.initial_seed(),
                     transform_seed=torch.initial_seed())
    return embedding.fit_transform(data)


def plot_embeddings(emb, emb_l, labels, filepath):
    # cmap_obj, cmap_arr = custom_cmap(n=len(labels))
    cmap_obj, cmap_arr = hamed_cmap(n=len(labels))

    plt.figure()
    plt.scatter(emb[:, 0], emb[:, 1], c=emb_l, cmap=cmap_obj, s=25, alpha=0.25, edgecolors='none')
    l_elems = [Line2D([0], [0], marker='o', color=cm, label=l, alpha=0.5, linestyle='None')
               for (cm, l) in zip(cmap_arr, labels)]
    plt.legend(frameon=False, loc=2, handles=l_elems)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

def tensor_to_df(tensor, ax_names=None):
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df

def plot_kls_df(df, filepath):
    _, cmap_arr = custom_cmap(df[df.columns[0]].nunique() + 1)
    with sns.plotting_context("notebook", font_scale=2.0):
        g = sns.FacetGrid(df, height=12, aspect=2)
        g = g.map(sns.boxplot, df.columns[1], df.columns[2], df.columns[0], palette=cmap_arr[1:],
                  order=None, hue_order=None)
        g = g.set(yscale='log').despine(offset=10)
        plt.legend(loc='best', fontsize='22')
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

def mkdir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def custom_plot_loss_2x2(loss_dict, keys_to_plot: List[str], title: str, name_to_save: str,  runPath: str): 
    has_all_keys = all(elem in list(loss_dict.keys()) for elem in keys_to_plot)
    if not has_all_keys:
        print('invalid key for loss')
        print(list(loss_dict.keys()))
        print(keys_to_plot)
        return
    
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=[12, 7], dpi=100)

    losses_names = keys_to_plot
    plot_indx = 0
    
    for col in ax:
        for row in col:
            loss_name = losses_names[plot_indx]
        
            lowest_loss_x = np.argmin(np.array(loss_dict[loss_name]))
            lowest_loss_y = loss_dict[loss_name][lowest_loss_x]
            
            row.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
            row.plot(loss_dict[loss_name], '-x', label=f'{loss_name}', markevery = [lowest_loss_x])

            row.set_xlabel(xlabel='iterations/epochs')
            row.set_ylabel(ylabel=f'{loss_name}')

            row.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
            row.legend()
            # row.label_outer()

            plot_indx += 1

    fig.suptitle(f'{title}')

    mkdir(runPath)
    plt.savefig(f'{runPath}/{name_to_save}.jpg')
    plt.clf()

def custom_plot_loss(loss_dict, keys_to_plot: List[str], title: str, name_to_save: str,  runPath: str): 
    has_all_keys = all(elem in list(loss_dict.keys()) for elem in keys_to_plot)
    if not has_all_keys:
        print('invalid key for loss')
        print(list(loss_dict.keys()))
        print(keys_to_plot)
        return
    
    '''Maximum Two plots, # TODO change to 4 plots
    '''
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=[12, 7], dpi=100)
    
    losses_names = keys_to_plot
    plot_indx = 0
    
    for col in ax:
        loss_name = losses_names[plot_indx]
    
        lowest_loss_x = np.argmin(np.array(loss_dict[loss_name]))
        lowest_loss_y = loss_dict[loss_name][lowest_loss_x]
        
        col.annotate("{:.4f}".format(lowest_loss_y), [lowest_loss_x, lowest_loss_y])
        col.plot(loss_dict[loss_name], '-x', label=f'{loss_name}', markevery = [lowest_loss_x])

        col.set_xlabel(xlabel='iterations/epochs')
        col.set_ylabel(ylabel=f'{loss_name}')

        col.grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        col.legend()
        # col.label_outer()

        plot_indx += 1

    fig.suptitle(f'{title}')

    mkdir(runPath)
    plt.savefig(f'{runPath}/{name_to_save}.jpg')
    plt.clf()
