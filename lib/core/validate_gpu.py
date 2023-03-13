import os
import torch
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import pandas as pd

from lib.utils import get_stats
from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_and_target_points
from lib.core.evaluate import get_sdr_statistics

from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.visualisations import display_box_plot
from lib.visualisations import radial_error_vs_ere_graph
from lib.visualisations import reliability_diagram
from lib.visualisations import reliability_diagram_norm
from lib.visualisations import roc_outlier_graph

from lib.measures import measure
from lib.measures import diagnose_set


def validate_over_set(ensemble, loader, final_layer, loss_function, visuals, cfg_validation, save_path, save_txt=False,
                      logger=None, training_mode=False, temperature_scaling_mode=False):
    
    print('save path:', save_path)
    predicted_points_per_model = []
    eres_per_model = []
    modes_per_model = []
    dataset_target_points = []
    scaled_predicted_points_per_model = []
    dataset_target_scaled_points = []
    losses = []

    #Create folders for images
    for visual_name in visuals:
        visual_dir = os.path.join(save_path, visual_name)
        if not os.path.exists(visual_dir):
            os.makedirs(visual_dir)

    # In gpu mode we run through all images in each model first
    for model_idx, model in enumerate(ensemble):

        model = model.cuda()
        model.eval()

        model_predicted_points = []
        model_predicted_scaled_points = []
        dataset_target_points = []
        dataset_target_scaled_points = []
        model_eres = []
        model_modes = []

        for idx, (image, channels, meta) in enumerate(loader):

            # allocate
            image = image.cuda()
            channels = channels.cuda()
            meta['landmarks_per_annotator'] = meta['landmarks_per_annotator'].cuda()
            meta['pixel_size'] = meta['pixel_size'].cuda()

            output = model(image.float())
            output = final_layer(output)
            loss = loss_function(output, channels)
            losses.append(loss.item())

            predicted_points, target_points, eres, modes, scaled_predicted_points, scaled_target_points \
                = get_predicted_and_target_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])
            model_predicted_points.append(predicted_points)
            dataset_target_points.append(target_points)
            model_predicted_scaled_points.append(scaled_predicted_points)
            dataset_target_scaled_points.append(scaled_target_points)
            model_eres.append(eres)
            model_modes.append(modes)

            # print intermediate figures
            b = 0
            for visual_name in visuals:
                image_name = meta["file_name"][b]
                figure_save_path = os.path.join(save_path, visual_name,'images',
                                                "{}_{}_{}".format(image_name, model_idx, visual_name))
                if not os.path.exists(os.path.join(save_path, visual_name,'images')):
                    os.makedirs(os.path.join(save_path, visual_name,'images'))
                intermediate_figure(image[0].detach().cpu().numpy(),
                                    output[b].detach().cpu().numpy(),
                                    predicted_points[b].detach().cpu().numpy(),
                                    target_points[b].detach().cpu().numpy(), eres[b].detach().cpu().numpy(),
                                    visual_name, save=True, save_path=figure_save_path)
                
                heatmap_arr = output[b].detach().cpu().numpy()
                # #save each point
                # for p in range(heatmap_arr.shape[0]):
                #     pp = p + 1 #point number
                #     arr_save_path = os.path.join(save_path, visual_name,'Arr',"{}_{}".format(image_name, pp,'heatmap_arr'))
                #     if not os.path.exists(os.path.join(save_path, visual_name,'Arr')):
                #         os.makedirs(os.path.join(save_path, visual_name,'Arr'))
                #     matplotlib.image.imsave(arr_save_path+'.png', heatmap_arr[p])

            if (idx + 1) % 30 == 0:
                logger.info("[{}/{}]".format(idx + 1, len(loader)))

        # move model back to cpu
        model.cpu()

        model_predicted_points = torch.cat(model_predicted_points)
        dataset_target_points = torch.cat(dataset_target_points)
        model_predicted_scaled_points = torch.cat(model_predicted_scaled_points)
        dataset_target_scaled_points = torch.cat(dataset_target_scaled_points)
        model_eres = torch.cat(model_eres)
        model_modes = torch.cat(model_modes)
        # D = Dataset size
        # predicted_points has size [D, N, 2]
        # eres has size [D, N]
        # target_points has size [D, N, 2]

        predicted_points_per_model.append(model_predicted_points)
        scaled_predicted_points_per_model.append(model_predicted_scaled_points)
        eres_per_model.append(model_eres)
        modes_per_model.append(model_modes)

    # predicted_points_per_model is size [M, D, N, 2]
    # eres_per_model is size [M, D, N]
    # target_points is size [D, N, 2]
    predicted_points_per_model = torch.stack(predicted_points_per_model).float()
    scaled_predicted_points_per_model = torch.stack(scaled_predicted_points_per_model).float()
    dataset_target_points = dataset_target_points.float()
    dataset_target_scaled_points = dataset_target_scaled_points.float()
    eres_per_model = torch.stack(eres_per_model).float()
    modes_per_model = torch.stack(modes_per_model).float()

    aggregated_point_dict = use_aggregate_methods(predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_points = aggregated_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]
    aggregated_scaled_point_dict = use_aggregate_methods(scaled_predicted_points_per_model, eres_per_model,
                                                  aggregate_methods=cfg_validation.AGGREGATION_METHODS)
    aggregated_scaled_points = aggregated_scaled_point_dict[cfg_validation.SDR_AGGREGATION_METHOD]

    radial_errors = cal_radial_errors(aggregated_scaled_points, dataset_target_scaled_points)

    measurements_dict = {}
    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = []

    # quick pass through the images at the end
    # This loop is for per image information and saving images
    for idx, (image, _, meta) in enumerate(loader):

        radial_errors_idx = radial_errors[idx]
        target_points_idx = dataset_target_points[idx]
        aggregated_points_idx = aggregated_points[idx]

        b = 0

        name = meta['file_name'][b]
        txt = "[{}/{}] {}:\t".format(idx + 1, len(loader), name)
        for err in radial_errors_idx:
            txt += "{:.3f}\t".format(err.item())
        txt += "Avg: {:.3f}\t".format(torch.mean(radial_errors_idx).item())

        for measurement in cfg_validation.MEASUREMENTS:
            predicted_angle, target_angle, dif = measure(aggregated_points_idx, target_points_idx,
                                cfg_validation.MEASUREMENTS_SUFFIX, measurement)
            measurements_dict[measurement].append([predicted_angle, target_angle])
            txt += "{}: {:.3f} ({:.3f} from target)\t".format(measurement, predicted_angle, dif)

        if not training_mode:
            logger.info(txt)

        # display visuals
        for visual_name in visuals:
            im_name = meta["file_name"][b]
            figure_save_path = save_path+'/'+im_name+".png"
            final_figure(image[b].detach().cpu().numpy(), 
                        aggregated_points_idx.detach().cpu().numpy(),
                        aggregated_point_dict, target_points_idx.detach().cpu().numpy(),
                        cfg_validation.MEASUREMENTS_SUFFIX, visual_name,
                        save=True, save_path=figure_save_path)
        
        #save points to txt file
        if training_mode==False:
            if save_txt == True:
                if not os.path.isdir(save_path+'/'+'txt/'):
                    os.makedirs(save_path+'/'+'txt/')
                
                with open(save_path+'/'+'txt/'+im_name+".txt", 'a') as output:
                    image_scaled_predicted_points_txt = scaled_predicted_points_per_model.cpu().squeeze().numpy()
                    image_scaled_predicted_points_txt = image_scaled_predicted_points_txt[idx]
                    for i in range(len(image_scaled_predicted_points_txt)):
                        row=image_scaled_predicted_points_txt[i]
                        data_str = str(round(row[1],5))+","+str(round(row[0],5))
                        output.write(data_str+"\n")

    ## plot measurements (ddh - alpha and beta)#
    save_me = True
    if save_me==True:
        if list(measurements_dict.keys())[0] == 'alpha_angle':
            #format is predicted to target .apply(lambda x: pd.Series(str(x).split(",")))
            _meausrements_dict = pd.DataFrame.from_dict(measurements_dict)
            #_meausrements_dict['alpha_angle'].sort()
            #_meausrements_dict['beta_angle'].sort()
            aa=_meausrements_dict['alpha_angle'].apply(lambda x: pd.Series(str(x).split(",")))
            aa[1]=aa[1].str.replace(']','')
            aa[0]=aa[0].str.replace('[','')

            bb=_meausrements_dict['beta_angle'].apply(lambda x: pd.Series(str(x).split(",")))
            bb[1]=bb[1].str.replace(']','')
            bb[0]=bb[0].str.replace('[','')

            aa=aa.astype(float).sort_values([1])
            bb=bb.astype(float).sort_values([1])

            aa_pred, aa_true = aa[0].to_numpy(),aa[1].to_numpy()
            ba_pred, ba_true = bb[0].to_numpy(),bb[1].to_numpy()
 
            # sorted_alpha = np.array(_meausrements_dict['alpha_angle'])
            # sorted_beta = np.array(_meausrements_dict['beta_angle'])
            # aa_pred, aa_true = (sorted_alpha)[:,0], (sorted_alpha)[:,1]
            # ba_pred, ba_true = (sorted_beta)[:,0], (sorted_beta)[:,1]
            
            #alpha cutoffline
            x = np.linspace(1,len(aa_pred),len(aa_pred))
            y_1 = np.full(len(aa_pred),60)
            y_l = np.full(len(aa_pred),55)
            y_u = np.full(len(aa_pred),65)

            plt.scatter(x, aa_true, marker='o', c='g', label='True Alpha')
            plt.scatter(x, aa_pred, marker='o', c='r', label='Pred Alpha') 
            plt.plot(x,y_1,'b', label='alpha threshold')
            plt.plot(x,y_l,'b--')
            plt.plot(x,y_u,'b--')

            plt.legend()
            plt.xlabel('Patient')
            plt.ylabel('Angle')
            plt.savefig('angles_truepred_alpha.png')

            plt.close()


            plt.scatter(x, ba_true, marker='*', c='g',label='True Beta')
            plt.scatter(x, ba_pred, marker='*',c='r', label='Pred Beta')
            
            #beta cutoff line
            y = np.full(len(aa_pred),77)
            y_l = np.full(len(aa_pred),72)
            y_u = np.full(len(aa_pred),82)

            x = np.linspace(1,len(ba_true),len(ba_true))
            plt.plot(x,y,'b', label='beta threshold')
            plt.plot(x,y_l,'b--')
            plt.plot(x,y_u,'b--')
            plt.legend()
            plt.xlabel('Patient')
            plt.ylabel('Angle')
            plt.savefig('angles_truepred_beta.png')
            plt.close()

            plt.rcParams["figure.figsize"] = (5,5)
            plt.scatter(aa_pred, aa_true, marker='o', c='g', label='alpha')  
            plt.legend()
            plt.xlabel('Prediction')
            plt.ylabel('Grund Truth')
            #a, b = np.polyfit(aa_pred, aa_true, 1)
            #plt.plot(aa_pred, a*aa_pred+b)
            plt.xlim(0, 120)  
            plt.ylim(0, 120)  
            x = np.linspace(1,120,60)
            y = x

            plt.plot(x,y, c='b')
            plt.plot(x,y+5,'--', c='b')
            plt.plot(x,y-5,'--', c='b')
            #plt.savefig('truevsprediction_alpha.png')
            #plt.close()

            
            plt.rcParams["figure.figsize"] = (5,5)
            plt.scatter(ba_pred, ba_true, marker='*', c='b', label='beta')  
            plt.legend()
            plt.xlabel('Prediction')
            plt.ylabel('Grund Truth')
            plt.xlim(0, 120)  
            plt.ylim(0, 120)  
            #a, b = np.polyfit(ba_pred, ba_true, 1)
            #plt.plot(ba_pred, a*ba_pred+b)    #
            #x y on diagonal
            x = np.linspace(1,120,60)
            y = x
            plt.plot(x,y, c='g')
            plt.plot(x,y+5,'--', c='b')
            plt.plot(x,y-5,'--', c='b')
            plt.savefig('truevsprediction.png')#_beta.png')

            plt.close()


    # Write where images have been saved
    for visual_name in visuals:
        figure_save_path = os.path.join(save_path, visual_name)
        txt = "Saved Images in {} for {}".format(figure_save_path, visual_name)
        logger.info(txt)


    # Overall Statistics
    logger.info("\n-----------Final Statistics-----------")

    # loss
    average_loss = sum(losses) / len(losses)
    txt = "Average loss: {:.3f}".format(average_loss)
    logger.info(txt)

    # Print average landmark localisations
    txt = "Landmark Localisations:\t"
    avg_per_landmark = torch.mean(radial_errors, dim=0)
    std_per_landmark = torch.std(radial_errors, dim=0)
    median_per_landmark = torch.median(radial_errors, dim=0)[0]
    for avg_for_landmark, std_for_landmark, median_for_landmark \
            in zip(avg_per_landmark, std_per_landmark, median_per_landmark):
        txt += "[MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}]\t".format(avg_for_landmark.item(),
                                                                  std_for_landmark.item(),
                                                                  median_for_landmark.item())
    overall_avg = torch.mean(radial_errors).item()
    overall_std = torch.std(radial_errors).item()
    overall_med = torch.median(radial_errors).item()
    txt += "[MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}]\t".format(overall_avg, overall_std, overall_med)

    for measurement in cfg_validation.MEASUREMENTS:
        measurements_dict[measurement] = torch.Tensor(measurements_dict[measurement])
        avg, std, med, icc = get_stats(measurements_dict[measurement][:, 0], measurements_dict[measurement][:, 1])
        txt += "{}: [MEAN: {:.3f}\u00B1{:.3f}, MED: {:.3f}, ICC: {:.3f}]\t".format(measurement, avg, std, med, icc)

    if not temperature_scaling_mode:
        logger.info(txt)

    sdr_rates = get_sdr_statistics(radial_errors, cfg_validation.SDR_THRESHOLDS)
    txt = "Successful Detection Rates are: "
    for sdr_rate in sdr_rates:
        txt += "{:.2f}%\t".format(sdr_rate)

    if not temperature_scaling_mode:
        logger.info(txt)

    # Final graphics
    if not training_mode:
        # Run the diagnosis experiments
        # I need to find the predicted points and the ground truth points
        # aggregated_scaled_points, dataset_target_scaled_points

        for diagnosis in cfg_validation.DIAGNOSES:       
            n, tn, fp, fn, tp, precision, recall, accuracy = diagnose_set(aggregated_scaled_points,
                                                                        dataset_target_scaled_points,
                                                                        cfg_validation.MEASUREMENTS_SUFFIX,
                                                                        diagnosis)

            if type(accuracy) is np.ndarray:
                #multiclass
                for c in range(len(accuracy)):
                    txt = "Results for {} in class {}, are n: {}, tn: {}, fp: {}, fn: {}, tp: {}, " \
                    "precision: {}, recall: {}, accuracy: {}".format(
                    diagnosis, c, n[c], tn[c], fp[c], fn[c], tp[c], precision[c], recall[c], accuracy[c])
                    logger.info(txt)
            else:
                #only one reported accuracy
                txt = "Results for {} are n: {}, tn: {}, fp: {}, fn: {}, tp: {}, " \
                "precision: {:.3f}, recall: {:.3f}, accuracy: {:.3f}".format(
                diagnosis, n, tn, fp, fn, tp, precision, recall, accuracy)
                logger.info(txt)

        radial_errors_np = radial_errors.detach().cpu().numpy()
        eres_np = eres_per_model[0].detach().cpu().numpy()
        confidence_np = modes_per_model[0].detach().cpu().numpy()

        figure_save_path = os.path.join(save_path, "box_plot")
        display_box_plot(radial_errors_np, figure_save_path)
        logger.info("Saving Box Plot to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "correlation_plot")
        radial_error_vs_ere_graph(radial_errors_np.flatten(), eres_np.flatten(), figure_save_path)
        logger.info("Saving Correlation Plot to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "correlation_plot_2")
        radial_error_vs_ere_graph(radial_errors_np.flatten(), confidence_np.flatten(), figure_save_path)
        logger.info("Saving Correlation Plot 2 to {}".format(figure_save_path))

        # Save the heatmap analysis plots
        figure_save_path = os.path.join(save_path, "roc_plot")
        roc_outlier_graph(radial_errors_np.flatten(), eres_np.flatten(), figure_save_path)
        logger.info("Saving ROC Plot to {}".format(figure_save_path))

        # Save the reliability diagram
        correct_thresholds=cfg_validation.CORRECT_THRESHOLDS
        figure_save_path = os.path.join(save_path, "reliability_plot")
        reliability_diagram(radial_errors_np.flatten(), correct_thresholds, confidence_np.flatten(), figure_save_path, meta['pixel_size'].cpu().detach().numpy())
        logger.info("Saving ROC Plot to {}".format(figure_save_path))

        figure_save_path = os.path.join(save_path, "reliability_plot_norm")
        reliability_diagram_norm(radial_errors_np.flatten(), correct_thresholds, confidence_np.flatten(), figure_save_path, meta['pixel_size'].cpu().detach().numpy())
        logger.info("Saving ROC Plot to {}".format(figure_save_path))

    if temperature_scaling_mode:

        radial_errors_np = radial_errors.detach().cpu().numpy()
        confidence_np = modes_per_model[0].detach().cpu().numpy()
        eres_np = eres_per_model[0].detach().cpu().numpy()

        # Save the reliability diagram
        correct_thresholds=cfg_validation.CORRECT_THRESHOLDS
        figure_save_path_ece = os.path.join(save_path, "ece_temp")
        ece = reliability_diagram(radial_errors_np.flatten(), correct_thresholds, confidence_np.flatten(), figure_save_path_ece, meta['pixel_size'].cpu().detach().numpy())
        ece_s = np.round(ece,2)
        logger.info("ECE:"+(str(ece_s)))
        figure_save_path_ece = os.path.join(save_path, "ece_temp_norm")
        ece_norm = reliability_diagram_norm(radial_errors_np.flatten(), correct_thresholds, confidence_np.flatten(), figure_save_path_ece, meta['pixel_size'].cpu().detach().numpy())

        figure_save_path_corr = os.path.join(save_path, "correlation_temp_scaling")
        correlation = radial_error_vs_ere_graph(radial_errors_np.flatten(), eres_np.flatten(), figure_save_path_corr, save=True)
        logger.info("Correlation: {:.3f}".format(correlation))

        temperatures = torch.flatten(ensemble[0].temperatures)
        logger.info("Temperature values: {}".format(temperatures))

    return average_loss, overall_avg
