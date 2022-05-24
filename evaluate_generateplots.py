import os
import sys
import time
import scipy
import warnings
import settings
import datetime
import matplotlib

import numpy   as np
import pandas  as pd
import seaborn as sns
import nibabel as nib

import tensorflow as tf 
import matplotlib.pyplot as plt

from scipy       import stats
from nilearn     import plotting
from sklearn     import metrics
from tensorflow  import keras as K
from prepdataset import prepdata
from dataloader  import DatasetGenerator
from time        import perf_counter
from skimage     import morphology

from skimage.transform import resize

import matplotlib.image as mpimg

cmap = matplotlib.colors.ListedColormap([plt.cm.get_cmap('viridis',3)(1), 'red'])
cp   = sns.color_palette("husl", 4)
cp_g =['#BBBBBB','#DDDDDD']
fig_font = 'monospace'

def generate_boxplots_datasets(scores_df1,evalFigures_path):
    
    figure1 = plt.figure(figsize=(15,10))
    
    ax1 = figure1.add_subplot(2,3,1)
    
    ax1 = sns.stripplot(x="L|R", y="dice", hue="dataset", jitter=True, dodge=True, marker='o', 
                   palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    
    ax1 =sns.boxplot(x="L|R", y="dice", hue="dataset", data=scores_df1,order=["L","R"], 
                palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))
    
    ax1.get_legend().remove()
    
    xlabels = ['Left','Right']
    ax1.set_xticklabels(xlabels)
    
    ax1.set_ylabel("Dice Scores", fontsize=12,fontname=fig_font)
    ax1.set_xlabel("Hippocampus", fontsize=12,fontname=fig_font)
    ax1.set_title("Dice Scores", fontsize=13 ,fontname=fig_font, fontweight='heavy')
    ax1.set_ylim(bottom=0.75, top=1)
    
    
    ax2 = figure1.add_subplot(2,3,2)
    
    ax2 = sns.stripplot(x="L|R", y="Jacc", hue="dataset", jitter=True, dodge=True, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    ax2 =sns.boxplot(x="L|R", y="Jacc", hue="dataset", data=scores_df1,order=["L","R"], 
                palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))
    
    ax2.get_legend().remove()
    

    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Jaccard Index", fontsize=12,fontname=fig_font)
    ax2.set_xlabel("Hippocampus"  , fontsize=12,fontname=fig_font)
    ax2.set_title("Jaccard Index" , fontsize=13,fontname=fig_font, fontweight='heavy')
    ax2.set_ylim(bottom=0.75, top=1)

    ax3 = figure1.add_subplot(2,3,3)

    ax3 = sns.stripplot(x="L|R", y="Kapp", hue="dataset", jitter=True, dodge=True, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    ax3 =sns.boxplot(x="L|R", y="Kapp", hue="dataset", data=scores_df1,order=["L","R"], 
                    palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))
    
    ax3.get_legend().remove()
    
    ax3.set_xticklabels(xlabels)
    ax3.set_ylabel("Kappa Index", fontsize=12,fontname=fig_font)
    ax3.set_xlabel("Hippocampus", fontsize=12,fontname=fig_font)
    ax3.set_title("Kappa Index", fontsize=13,fontname=fig_font, fontweight='heavy')
    ax3.set_ylim(bottom=0.75, top=1)

    
    ax4 = figure1.add_subplot(2,3,4)
    
    ax4 = sns.stripplot(x="L|R", y="Accu", hue="dataset", jitter=True, dodge=True, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    ax4 =sns.boxplot(x="L|R", y="Accu", hue="dataset", data=scores_df1,order=["L","R"], 
                    palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))

    ax4.get_legend().remove()
    
    ax4.set_xticklabels(xlabels)
    ax4.set_ylabel("Accuracy", fontsize=12,fontname=fig_font)
    ax4.set_xlabel("Hippocampus", fontsize=12,fontname=fig_font)
    ax4.set_title("Accuracy", fontsize=13,fontname=fig_font, fontweight='heavy')
    ax4.set_ylim(bottom=0.99, top=1)

    
    ax5 = figure1.add_subplot(2,3,5)

    ax5 = sns.stripplot(x="L|R", y="Recl", hue="dataset", jitter=True, dodge=True, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    ax5 =sns.boxplot(x="L|R", y="Recl", hue="dataset", data=scores_df1,order=["L","R"], 
                    palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))
    ax5.get_legend().remove()
    ax5.set_xticklabels(xlabels)
    ax5.set_ylabel("Recall", fontsize=12,fontname=fig_font)
    ax5.set_xlabel("Hippocampus", fontsize=12,fontname=fig_font)
    ax5.set_title("Recall (Sensitivity)", fontsize=13,fontname=fig_font, fontweight='heavy')
    ax5.set_ylim(bottom=0.75, top=1)

    
    ax6 = figure1.add_subplot(2,3,6)
    
    ax6 = sns.stripplot(x="L|R", y="Prec", hue="dataset", jitter=True, dodge=True, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray")
    ax6 =sns.boxplot(x="L|R", y="Prec", hue="dataset", data=scores_df1,order=["L","R"], 
                    palette=cp, fliersize=2, linewidth=1,boxprops=dict(alpha=.6))
    handles, _ = ax6.get_legend_handles_labels()
    
    ax6.set_xticklabels(xlabels)
    ax6.set_ylabel("Precision", fontsize=12,fontname=fig_font)
    ax6.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax6.set_title("Precision (Specificity)", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax6.set_ylim(bottom=0.75, top=1)


    plt.legend(handles, ["HAD", "ADB","ADNI","CC"], bbox_to_anchor=(1.05, 2.15), loc='upper left',title='Dataset')
    
    figure1.tight_layout()
    figure1.subplots_adjust(hspace = .4)
    figure1.savefig(os.path.join(evalFigures_path,'fig1_boxplots-datasets.png'),bbox_inches='tight')
    plt.close()
    
def generate_boxplots_average(scores_df1,evalFigures_path):
    
    figure1 = plt.figure(figsize=(15,10))
    box_width=0.3
    
    ax1 = figure1.add_subplot(2,3,1)

    ax1 = sns.stripplot(x="L|R", y="dice", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax1 =sns.boxplot(x="L|R", y="dice", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width)
    ax1.get_legend().remove()
    xlabels = ['Left','Right']
    
    ax1.set_xticklabels(xlabels)
    ax1.set_ylabel("Dice Score", fontsize=12,fontname=fig_font)
    ax1.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax1.set_title("Dice Scores", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax1.set_ylim(bottom=0.75, top=1)
    
    medians = np.round(scores_df1.groupby(["L|R"])["dice"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["dice"].mean()  ,4)
    offset_med = scores_df1['dice'].median() * 0.03 # offset from median for display
    offset_avg = offset_med+0.015
    for xtick in ax1.get_xticks():
        ax1.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax1.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')


    
    ax2 = figure1.add_subplot(2,3,2)
    ax2 = sns.stripplot(x="L|R", y="Jacc", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax2 =sns.boxplot(x="L|R", y="Jacc", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width)
    ax2.get_legend().remove()
    
    ax2.set_xticklabels(xlabels)
    ax2.set_ylabel("Jaccard Score", fontsize=12,fontname=fig_font)
    ax2.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax2.set_title("Jaccard Scores", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax2.set_ylim(bottom=0.75, top=1)

    medians = np.round(scores_df1.groupby(["L|R"])["Jacc"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["Jacc"].mean()  ,4)
    offset_med = scores_df1['Jacc'].median() * 0.055 # offset from median for display
    offset_avg = offset_med+0.017
    for xtick in ax2.get_xticks():
        ax2.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax2.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')



    ax3 = figure1.add_subplot(2,3,3)
    ax3 = sns.stripplot(x="L|R", y="Kapp", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax3 =sns.boxplot(x="L|R", y="Kapp", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width)
    ax3.get_legend().remove()
    ax3.set_xticklabels(xlabels)
    ax3.set_ylabel("Kappa Value", fontsize=12,fontname=fig_font)
    ax3.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax3.set_title("Kappa Values", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax3.set_ylim(bottom=0.75, top=1)

    medians = np.round(scores_df1.groupby(["L|R"])["Kapp"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["Kapp"].mean()  ,4)
    offset_med = scores_df1['Kapp'].median() * 0.03 # offset from median for display
    offset_avg = offset_med+0.017
    for xtick in ax3.get_xticks():
        ax3.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax3.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')




    ax4 = figure1.add_subplot(2,3,4)
    ax4 = sns.stripplot(x="L|R", y="Accu", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax4 =sns.boxplot(x="L|R", y="Accu", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width)
    ax4.get_legend().remove()
  
    ax4.set_xticklabels(xlabels)
    ax4.set_ylabel("Accuracy", fontsize=12,fontname=fig_font)
    ax4.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax4.set_title("Accuracy", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax4.set_ylim(bottom=0.99, top=1)


    medians = np.round(scores_df1.groupby(["L|R"])["Accu"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["Accu"].mean()  ,4)
    offset_med = scores_df1['Accu'].median() * 0.0013 # offset from median for display
    offset_avg = offset_med+0.0005
    for xtick in ax4.get_xticks():
        ax4.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax4.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')


    ax5 = figure1.add_subplot(2,3,5)
    ax5 = sns.stripplot(x="L|R", y="Recl", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax5 =sns.boxplot(x="L|R", y="Recl", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width)
    ax5.get_legend().remove()
    
    ax5.set_xticklabels(xlabels)
    ax5.set_ylabel("Recall", fontsize=12,fontname=fig_font)
    ax5.set_xlabel("Hippocampus", fontsize=12, fontname=fig_font)
    ax5.set_title("Recall", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax5.set_ylim(bottom=0.75, top=1)

    medians = np.round(scores_df1.groupby(["L|R"])["Recl"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["Recl"].mean()  ,4)
    offset_med = scores_df1['Recl'].median() * 0.06 # offset from median for display
    offset_avg = offset_med+0.018
    for xtick in ax5.get_xticks():
        ax5.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax5.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')


    ax6 = figure1.add_subplot(2,3,6)
    ax6 = sns.stripplot(x="L|R", y="Prec", hue="dataset", jitter=True, dodge=False, marker='o', 
                       palette=cp, linewidth=1, data=scores_df1,edgecolor="gray",size=5,alpha=0.7)
    ax6 =sns.boxplot(x="L|R", y="Prec", data=scores_df1,order=["L","R"], 
                    palette=cp_g, fliersize=0.7, linewidth=1.5,boxprops=dict(alpha=0.8),width=box_width,
                    showmeans=True,meanprops={"marker": "o",
                                              "markerfacecolor": "white","markeredgecolor": "red",
                                              "markersize": "5","linewidth":"5"})
    handles, _ = ax6.get_legend_handles_labels()
    ax6.legend(handles, ["HAD", "ADB","ADNI","CC"],loc='lower left', title='Dataset')
    
    ax6.set_xticklabels(xlabels)
    ax6.set_ylabel("Precision", fontsize=12,fontname=fig_font)
    ax6.set_xlabel("Precision", fontsize=12, fontname=fig_font)
    ax6.set_title("Precision", fontsize=13, fontname=fig_font, fontweight='heavy')
    ax6.set_ylim(bottom=0.75, top=1)
    
    medians = np.round(scores_df1.groupby(["L|R"])["Prec"].median(),4)
    means   = np.round(scores_df1.groupby(["L|R"])["Prec"].mean()  ,4)
    offset_med = scores_df1['Prec'].median() * 0.068 # offset from median for display
    offset_avg = offset_med+0.01
    for xtick in ax6.get_xticks():
        ax6.text(xtick, means[xtick] + offset_avg, str("μ = ")+str(means[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')
        ax6.text(xtick,medians[xtick] + offset_med, str("med = ")+str(medians[xtick]),
                horizontalalignment='left',size='small',color='k',weight='regular')

 
    plt.legend(handles, ["HAD", "ADB","ADNI","CC"], bbox_to_anchor=(1.05, 2.15), loc='upper left',title='Dataset')
    
    figure1.tight_layout()
    figure1.subplots_adjust(hspace = .4)
    figure1.savefig(os.path.join(evalFigures_path,'fig2_boxplots-dataset_avg.png'),bbox_inches='tight')
    plt.close()
    
def create_confusionMatrix(confmat_df,evalFigures_path):
    
    confmat_avg = np.mean(confmat_df[3:7])

    [tn, fp, fn, tp] = [confmat_avg[0],confmat_avg[1],confmat_avg[2],confmat_avg[3]]
    cf_matrix = np.asarray([tn, fp,fn, tp]).reshape(2,2)
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts      = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [ "{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    boundaries = [value for value in cf_matrix.flatten().tolist()]
    list.sort(boundaries)

    colors = ['darksalmon','indianred','darkturquoise','skyblue']

    norm   = matplotlib.colors.BoundaryNorm(boundaries=boundaries + [boundaries[-1]], ncolors=256)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    
    confMat_fig = sns.heatmap(cf_matrix, annot=labels, 
                          annot_kws={"size":14,"family":'monospace',"fontweight":'light'},
                          fmt="", cmap=cmap, norm=norm, cbar=False,
                          cbar_kws={'format': '%.0f%%', 'ticks': boundaries, 'drawedges': True},
                          linewidths=1.5, linecolor='white', xticklabels=False, yticklabels=False)
    
    figure3 = plt.figure(figsize=(7,7))
    figure3 = confMat_fig.get_figure()
    figure3.savefig(os.path.join(evalFigures_path,'fig3_confusionMatrix.png'))
    plt.close(figure3)
    
    
def plot_confMat_distributions(distrib_df,evalFigures_path):
    
    figure4_1 = plt.figure(figsize=(20,4))
    
    fullrange = np.around(np.transpose(np.arange(0,1.01,0.01)),2)
    dist_avg  = np.sum(distrib_df.iloc[:,3:207])

    tn = np.hstack((dist_avg[ 0: 50], np.zeros(50) ))
    fn = np.hstack((dist_avg[50:100], np.zeros(50) ))

    fp = np.hstack((np.zeros(50), dist_avg[100:150]))
    tp = np.hstack((np.zeros(50), dist_avg[150:200]))
    
    ax = sns.barplot(x=fullrange[0:100], y=tn ,label = 'True Negatives' , color='skyblue');
    ax = sns.barplot(x=fullrange[0:100], y=fn ,label = 'False Negatives', color='darksalmon');

    ax = sns.barplot(x=fullrange[0:100], y=fp, label = 'False Positives', color='indianred');
    ax = sns.barplot(x=fullrange[0:100], y=tp ,label = 'True Positives' , color='darkturquoise');

    ax.legend(ncol = 2, loc = 'upper right', fontsize=10) ;

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",fontsize=7) ;
    
    figure4_1 = ax.get_figure()
    figure4_1.savefig(os.path.join(evalFigures_path,'fig4-1_confmat-distributions_all.png'))
    plt.close(figure4_1)

    
    figure4_2 = plt.figure(figsize=(10,4))

    dist_fp = dist_avg[100:150]
    dist_tp = dist_avg[150:200]

    tp = np.transpose(dist_tp)
    fp = np.transpose(dist_fp)

    ax2 = sns.barplot(x=fullrange[51:101], y=tp ,label = 'True Positives'  , color='darkturquoise') ;
    ax2 = sns.barplot(x=fullrange[51:101], y=fp, label = 'False Positives' , color='indianred') ;

    ax2.legend(ncol = 2, loc = 'upper right', fontsize=10) ;

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right",fontsize=7) ;
    
    figure4_2 = ax2.get_figure()
    figure4_2.savefig(os.path.join(evalFigures_path,'fig4-2_confmat-distributions_positives.png'))
    plt.close(figure4_2)
    
    
    
    figure4_3 = plt.figure(figsize=(10,4))
    
    dist_tn = dist_avg[0: 50]
    dist_fn = dist_avg[50:100]
    
    tn = np.transpose(dist_tn)
    fn = np.transpose(dist_fn)
    
    ax3 = sns.barplot(x=fullrange[0:50], y=tn ,label = 'True Negatives' , color='skyblue');
    ax3 = sns.barplot(x=fullrange[0:50], y=fn ,label = 'False Negatives', color='darksalmon');
    
    ax3.legend(ncol = 2, loc = 'upper right', fontsize=10) ;
    
    ax3.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right",fontsize=7) ;
    
    figure4_3 = ax3.get_figure()
    figure4_3.savefig(os.path.join(evalFigures_path,'fig4-3_confmat-distributions_negatives.png'))
    plt.close(figure4_3)
    
    
def plot_volumeCorrelations(volumes_df,evalFigures_path):
    
    figure5 = plt.figure(figsize=(20,4))
    
    X = volumes_df['Volumes: Truth']
    Y = volumes_df['Volumes: Predictions']

    ax = sns.jointplot(x=X, y=Y, kind='reg', color='mediumpurple')
    r, p = stats.pearsonr(X, Y)

    ax.ax_joint.annotate(f'$\\rho = {r:.3f}$',xy=(0.1, 0.9), xycoords='axes fraction',
                        ha='left', va='center', bbox={'boxstyle': 'round', 'fc': 'thistle', 
                                                      'ec': 'mediumpurple'},fontsize=15)

    ax.ax_joint.plot([0,1], [0,1], ':y', transform=ax.ax_joint.transAxes,color='mediumpurple')

    ax.ax_joint.scatter(X, Y, color='thistle', edgecolor='mediumpurple')
    ax.set_axis_labels(xlabel= 'Truth Volumes (mm$\mathregular{^{3}}$)', 
                       ylabel= 'Predicted (mm$\mathregular{^{3}}$)', size=14)
    
    figure5 = ax
    figure5.savefig(os.path.join(evalFigures_path,'fig5_volumeCorrelations.png'))