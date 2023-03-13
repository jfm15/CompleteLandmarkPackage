import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def radial_error_vs_ere_graph(radial_errors, eres, save_path, n_bin=36, save=True):

    # Bin the ere and calculate the radial error for each bin
    binned_eres = []
    binned_errors = []
    sorted_indices = np.argsort(eres)
    for l in range(int(len(eres) / n_bin)):
        binned_indices = sorted_indices[l * n_bin: (l + 1) * n_bin]
        binned_eres.append(np.mean(np.take(eres, binned_indices)))
        binned_errors.append(np.mean(np.take(radial_errors, binned_indices)))
    correlation = np.corrcoef(binned_eres, binned_errors)[0, 1]

    if save:
        # Plot graph
        plt.rcParams["figure.figsize"] = (6, 6)
        fig, ax = plt.subplots(1, 1)
        ax.grid(zorder=0)
        plt.xlabel('Expected Radial Error (ERE)', fontsize=14)
        plt.ylabel('True Radial Error', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.text(0.5, 0.075, "CORRELATION={:.2f}".format(correlation), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large', transform=ax.transAxes)
        ax.scatter(binned_eres, binned_errors, c='lime', edgecolors='black', zorder=3)
        plt.savefig(save_path)
        plt.close()

    return correlation


def roc_outlier_graph(radial_errors, eres, save_path, outlier_threshold=2.0):
    outliers = radial_errors > outlier_threshold

    fpr, tpr, thresholds = roc_curve(outliers, eres)
    auc = roc_auc_score(outliers, eres)

    first_idx = np.min(np.argwhere(tpr > 0.5))
    proposed_threshold = thresholds[first_idx]

    # Plot graph
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    plt.xlabel("False Positive Rate (FPR)", fontsize=14)
    plt.ylabel("True Positive Rate (TPR)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)

    plt.plot(fpr, tpr, c="blue")
    plt.text(0.42, 0.075, 'Area Under Curve={:.2f}'.format(auc), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large',
             transform=ax.transAxes)
    plt.savefig(save_path)
    plt.close()

    return proposed_threshold

# At the moment this assumes all images have the same resolution
def reliability_diagram(radial_errors, correct_thresholds, mode_probabilities, save_path,
                        pixel_size, n_of_bins=10, save=True): 
    
    # mode probablilites is an array of all probabilities for the hottest point for each output heatmap 
    # (in other words the probability of the predicted landmark point)

    x_max = math.floor(np.max(mode_probabilities) / 0.001) * 0.001 
    bins = np.linspace(0, x_max, n_of_bins + 1)
    bins[-1] = 1.1
    widths = x_max / n_of_bins

    if pixel_size[0][1] == pixel_size[0][0]:
        pixel_size = pixel_size[0][1]
    else:
        raise ValueError('pixel size is not isotropic check image cache metadata')
    
    #iterate for all thresholds and store ece values to list as well as save plot
    ece = []

    for threshold in correct_thresholds:
        #find number of correct predictions 
        radius = math.sqrt(((pixel_size*threshold)**2) / math.pi) 
        correct_predictions = radial_errors < radius

        # a 10 length array with values adding to 19
        count_for_each_bin, _ = np.histogram(mode_probabilities, bins=bins)

        # total confidence in each bin
        total_confidence_for_each_bin, _, bin_indices \
            = stats.binned_statistic(mode_probabilities, mode_probabilities, 'sum', bins=bins)

        no_of_correct_preds = np.zeros(len(bins) - 1)

        for bin_idx, pred_correct in zip(bin_indices, correct_predictions):
            if pred_correct==True:
                no_of_correct_preds[bin_idx - 1] += 1
        print(no_of_correct_preds)
        
        # get confidence of each bin
        avg_conf_for_each_bin = total_confidence_for_each_bin / count_for_each_bin.astype(float)
        avg_conf_for_each_bin[np.isnan(avg_conf_for_each_bin)] = 0.0
        avg_acc_for_each_bin = no_of_correct_preds / count_for_each_bin.astype(float)
        avg_acc_for_each_bin[np.isnan(avg_acc_for_each_bin)] = 0.0

        n = float(np.sum(count_for_each_bin))
        ece_tmp = 0.0
        for i in range(len(bins) - 1):
            ece_tmp += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
        
        ece_tmp *= 100

        ece=np.append(ece,ece_tmp)

        if save:
            # save plot
            plt.rcParams["figure.figsize"] = (6, 6)
            fig, ax = plt.subplots(1, 1)
            ax.grid(zorder=0)

            plt.subplots_adjust(left=0.15)
            plt.xlabel('Confidence', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(zorder=0)
            plt.xlim(0.0, x_max)
            plt.ylim(0.0, np.max(avg_acc_for_each_bin))
            plt.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
            plt.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
                    label='Gap', zorder=3)
            plt.legend(fontsize=20, loc="upper left", prop={'size': 16})
            plt.text(0.71, 0.075, 'ECE={:.2f}'.format(ece_tmp), backgroundcolor='white', fontsize='x-large', transform=ax.transAxes)

            save_path_tmp = save_path[:-4] + 'AccThreshold'+str(int(threshold))
            plt.savefig(save_path_tmp)
            plt.close()

    return ece

# # At the moment this assumes all images have the same resolution
# def reliability_diagram_w_norm_beg(radial_errors, mode_probabilities, save_path,
#                         pixel_size, n_of_bins=10, save=True): 
    
#     # mode probablilites is an array of all probabilities for the hottest point for each output heatmap 
#     # (in other words the probability of the predicted landmark point)

#     x_max = 1
#     bins = np.linspace(0, x_max, n_of_bins + 1)
#     widths = x_max / n_of_bins

#     mode_prob_norm = np.array([])
#     for m in mode_probabilities:
#         n = m/mode_probabilities.sum()
#         mode_prob_norm = np.append(mode_prob_norm,n) 

#     mode_probabilities = mode_prob_norm

#     if pixel_size[0][1] == pixel_size[0][0]:
#         pixel_size = pixel_size[0][1]
#     else:
#         raise ValueError('pixel size is not isotropic check image cache metadata')
    
#     #find number of correct predictions
#     radius = math.sqrt((pixel_size**2) / math.pi) #area of disc in pixels around gt
#     correct_predictions = radial_errors < radius

#     # a 10 length array with values adding to 19
#     count_for_each_bin, _ = np.histogram(mode_probabilities, bins=bins)

#     # total confidence in each bin
#     total_confidence_for_each_bin, _, bin_indices \
#         = stats.binned_statistic(mode_probabilities, mode_probabilities, 'sum', bins=bins)

#     no_of_correct_preds = np.zeros(len(bins) - 1)

#     for bin_idx, pred_correct in zip(bin_indices, correct_predictions):
#         if pred_correct==True:
#             no_of_correct_preds[bin_idx - 1] += 1
#     print(no_of_correct_preds)
    
#     # get confidence of each bin
#     avg_conf_for_each_bin = total_confidence_for_each_bin / count_for_each_bin.astype(float)
#     avg_conf_for_each_bin[np.isnan(avg_conf_for_each_bin)] = 0.0
#     avg_acc_for_each_bin = no_of_correct_preds / count_for_each_bin.astype(float)
#     avg_acc_for_each_bin[np.isnan(avg_acc_for_each_bin)] = 0.0

#     n = float(np.sum(count_for_each_bin))
#     ece = 0.0
#     for i in range(len(bins) - 1):
#         ece += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
#     ece *= 100

#     if save:
#         # save plot
#         plt.rcParams["figure.figsize"] = (6, 6)
#         fig, ax = plt.subplots(1, 1)
#         ax.grid(zorder=0)

#         plt.subplots_adjust(left=0.15)
#         plt.xlabel('Confidence Normalized', fontsize=14)
#         plt.ylabel('Accuracy Normalized', fontsize=14)
#         plt.xticks(fontsize=14)
#         plt.yticks(fontsize=14)
#         plt.grid(zorder=0)
#         plt.xlim(0.0, x_max)
#         plt.ylim(0.0, np.max(avg_acc_for_each_bin))
#         plt.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
#         plt.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
#                 label='Gap', zorder=3)
#         plt.legend(fontsize=20, loc="upper left", prop={'size': 16})
#         plt.text(0.71, 0.075, 'ECE={:.2f}'.format(ece), backgroundcolor='white', fontsize='x-large', transform=ax.transAxes)

#         plt.savefig(save_path)
#         plt.show()
#         plt.close()

#     return ece

def reliability_diagram_norm(radial_errors, correct_thresholds, mode_probabilities, save_path,
                        pixel_size, n_of_bins=10, save=True): 
    
    # mode probablilites is an array of all probabilities for the hottest point for each output heatmap 
    # (in other words the probability of the predicted landmark point)

    x_max = math.floor(np.max(mode_probabilities) / 0.001) * 0.001 
    bins = np.linspace(0, x_max, n_of_bins + 1)
    bins[-1] = 1.1
    widths = x_max / n_of_bins

    if pixel_size[0][1] == pixel_size[0][0]:
        pixel_size = pixel_size[0][1]
    else:
        raise ValueError('pixel size is not isotropic check image cache metadata')
    
    ece=[]
    #find number of correct predictions
    for threshold in correct_thresholds:
        #find number of correct predictions 
        radius = math.sqrt(((pixel_size*threshold)**2) / math.pi) #area of disc in pixels around gt
        
        #radial error in mm and radius in mm
        correct_predictions = radial_errors < radius

        # a 10 length array with values adding to 19
        count_for_each_bin, _ = np.histogram(mode_probabilities, bins=bins)

        # total confidence in each bin
        total_confidence_for_each_bin, _, bin_indices \
            = stats.binned_statistic(mode_probabilities, mode_probabilities, 'sum', bins=bins)

        no_of_correct_preds = np.zeros(len(bins) - 1)

        for bin_idx, pred_correct in zip(bin_indices, correct_predictions):
            if pred_correct==True:
                no_of_correct_preds[bin_idx - 1] += 1
        #print(no_of_correct_preds)
        
        # get confidence of each bin
        avg_conf_for_each_bin = total_confidence_for_each_bin / count_for_each_bin.astype(float)
        avg_conf_for_each_bin[np.isnan(avg_conf_for_each_bin)] = 0.0
        avg_acc_for_each_bin = no_of_correct_preds / count_for_each_bin.astype(float)
        avg_acc_for_each_bin[np.isnan(avg_acc_for_each_bin)] = 0.0

        n = float(np.sum(count_for_each_bin))
        ece_tmp = 0.0
        for i in range(len(bins) - 1):
            ece_tmp += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
        
        ece_tmp *= 100

        ece=np.append(ece,ece_tmp)
        
        if save:
            # save plot norm
            x_max = 1
            bins = np.linspace(0, x_max, n_of_bins + 1)
            widths = x_max / n_of_bins

            avg_acc_for_each_bin_norm = [a/sum(avg_acc_for_each_bin) for a in avg_acc_for_each_bin]
            avg_conf_for_each_bin_norm = [c/sum(avg_conf_for_each_bin) for c in avg_conf_for_each_bin]

            plt.rcParams["figure.figsize"] = (6, 6)
            fig, ax = plt.subplots(1, 1)
            ax.grid(zorder=0)

            plt.subplots_adjust(left=0.15)
            plt.xlabel('Confidence Normalized', fontsize=14)
            plt.ylabel('Accuracy Normalized', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(zorder=0)
            plt.xlim(0.0, x_max)
            plt.ylim(0.0, x_max) #np.max(avg_acc_for_each_bin))
            plt.bar(bins[:-1], avg_acc_for_each_bin_norm, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
            plt.bar(bins[:-1], avg_conf_for_each_bin_norm, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
                    label='Gap', zorder=3)
            plt.legend(fontsize=20, loc="upper left", prop={'size': 16})
            plt.text(0.71, 0.075, 'ECE={:.2f}'.format(ece_tmp), backgroundcolor='white', fontsize='x-large', transform=ax.transAxes)

            
            save_path_tmp = save_path[:-4] + 'AccThreshold'+str(int(threshold))
            plt.savefig(save_path_tmp)
            #plt.show()
            #plt.close()

    return ece