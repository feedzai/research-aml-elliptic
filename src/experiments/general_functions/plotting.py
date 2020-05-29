import sys
import os

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import matplotlib.pyplot as plt
from experiments.general_functions.elliptic_data_preprocessing import calc_occurences_per_timestep
import seaborn as sns
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_performance_per_timestep(model_metric_dict, last_train_time_step=34,last_time_step=49, model_std_dict=None, fontsize=23, labelsize=18, figsize=(20, 10),
                                  linestyle=['solid', "dotted", 'dashed'], linecolor=["green", "orange", "red"],
                                  barcolor='lightgrey', baralpha=0.3, linewidth=1.5, savefig_path=None):
    occ = calc_occurences_per_timestep()
    illicit_per_timestep = occ[(occ['class'] == 1) & (occ['time_step'] > 34)]

    timesteps = illicit_per_timestep['time_step'].unique()
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    i = 0
    for key, values in model_metric_dict.items():
        if key != "XGBoost":
            key = key.lower()
        ax1.plot(timesteps, values, label=key, linestyle=linestyle[i], color=linecolor[i], linewidth=linewidth)
        if model_std_dict != None:
            ax1.fill_between(timesteps, values + model_std_dict[key], values - model_std_dict[key],
                             facecolor='lightgrey', alpha=0.5)
        i += 1

    ax2.bar(timesteps, illicit_per_timestep['occurences'], color=barcolor, alpha=baralpha, label='\# illicit')
    ax2.get_yaxis().set_visible(True)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.grid(False)

    ax1.set_xlabel('Time step', fontsize=fontsize)
    ax1.set_ylabel('Illicit F1', fontsize=fontsize)
    ax1.set_xticks(range(last_train_time_step+1,last_time_step+1))
    ax1.set_yticks([0,0.25,0.5,0.75,1])
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    ax1.legend(lines, labels, fontsize=fontsize, facecolor="#EEEEEE")

    ax1.tick_params(direction='in')

    ax2.set_ylabel('Num. samples', fontsize=fontsize)

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(os.path.join(ROOT_DIR, savefig_path), bbox_inches='tight', pad_inches=0)


def plot_UMAP_projection(embedding_df, hue_on='class', labelsize=20, fontsize=22, palette=['cadetblue', 'coral'],
                         linewidth=0.000001, savefig_path=None):
    fig_dims = (10, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.scatterplot(x='dim_0', y='dim_1', hue=hue_on, style=hue_on, markers=['.', 'X'], size=hue_on,
                    sizes=[150,170], linewidth=linewidth, palette=palette, data=embedding_df, ax=ax)
    ax.set_xlabel('Dimension 1', fontsize=fontsize)
    ax.set_ylabel('Dimension 2', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
    L = ax.legend(prop={'size': labelsize}, facecolor="#EEEEEE", handletextpad=-0.5)
    if hue_on == 'class':
        L.get_texts()[0].set_text('True label')
    else:
        L.get_texts()[0].set_text('Predicted label')
    L.get_texts()[1].set_text("licit")
    L.get_texts()[2].set_text("illicit")
    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
    fig.show()
