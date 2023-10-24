import os
import torch

from lib.core.evaluate import cal_radial_errors
from lib.core.evaluate import use_aggregate_methods
from lib.core.evaluate import get_predicted_points
from lib.visualisations import intermediate_figure
from lib.visualisations import final_figure
from lib.measures import measure


def test_over_set(ensemble, loader, final_layer, loss_function, visuals, cfg_validation, save_path, logger=None, training_mode=False):

    all_radial_errors = []
    all_measurement_difs = []
    losses = []

    for idx, (image, channels, meta) in enumerate(loader):

        image_predicted_points = []
        image_scaled_predicted_points = []

        for model in ensemble:

            model.eval()
            output = model(image.float())
            output = final_layer(output)
            loss = loss_function(output, channels)
            losses.append(loss.item())

            predicted_points, scaled_predicted_points, modes \
                = get_predicted_points(output, meta['landmarks_per_annotator'], meta['pixel_size'])

            image_predicted_points.append(predicted_points)
            image_scaled_predicted_points.append(scaled_predicted_points)

            # print figures
            b = 0
            visual_name = 'heatmaps_and_preds'

            if not os.path.isdir(save_path+'/'+visual_name):
                os.mkdir(save_path+'/'+visual_name)

            target_points = []
            eres = []
            intermediate_figure(image[b], output[b].numpy(), predicted_points[b],
                                target_points, eres, visual_name, save=True, save_path=save_path+'/'+visual_name+'/'+meta['file_name'][0])
            
        # put these arrays into a format suitable for the aggregate methods function
        image_predicted_points = torch.unsqueeze(torch.cat(image_predicted_points), 1).float()
        image_scaled_predicted_points = torch.unsqueeze(torch.cat(image_scaled_predicted_points), 1).float()
        
        image_scaled_predicted_points_txt = image_scaled_predicted_points.squeeze().numpy()
        
        if not os.path.isdir(save_path+'/txt/'):
            os.mkdir(save_path+'/txt/')
        
        with open(save_path+'/txt/'+meta['file_name'][0]+".txt", 'a') as output:
            for i in range(len(image_scaled_predicted_points_txt)):
                row=image_scaled_predicted_points_txt[i].tolist()
                data_str = str([round(row[1],5),round(row[0],5)])[1:-1]
                output.write(data_str+"\n")
    

    average_loss = sum(losses) / len(losses)
    txt = "Average loss: {:.3f}".format(average_loss)
    logger.info(txt)
    average_radial_error = 0
    
    
    return average_loss, average_radial_error
