import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math as math
from graf_angle_calculations import graph_angle_calculations

def plt_graf_lines(imgs_path, graf_points_dir, point_radius, plt_points=True, calc_angles=True, save_calculated_angles=True):
    gac = graph_angle_calculations()
    imgs = [f for f in os.listdir(imgs_path) if not f.startswith('.')]
    calc_df = pd.DataFrame([], columns=['Patient','Calculated alpha','Calculated beta','graf class ab', 'graf class a'])

    for img_path in imgs:
        img_name = img_path.split('/')[-1]
        point_path=graf_points_dir+'/'+img_name[:-4]+'.txt'#
    
        if os.path.isfile(point_path):
            points = (open(point_path,'r')).readlines()
            im = plt.imread(imgs_path+'/'+img_name)
            
            fig, ax = plt.subplots(1, 1)

            #ilium
            i1 = [float(i) for i in points[0].strip('\n').split(',')]
            i2 = [float(i) for i in points[1].strip('\n').split(',')]
            #bonyrim
            br = [float(i) for i in points[2].strip('\n').split(',')]
            #lower limb point
            ll = [float(i) for i in points[3].strip('\n').split(',')]
            #labrum
            l = [float(i) for i in points[4].strip('\n').split(',')]

            #add incase divisible by zero 
            if i1[1] == i2[1]:
                i2[1]=+i2[1]+0.000001
            if br[1] == l[1]:
                br[1]=br[1]+0.000001
            if br[1] == ll[1]: 
                br[1]=br[1]+0.000001


            if plt_points==True:
                plt.scatter(i1[1], i1[0],c='r', s=20)
                plt.scatter(i2[1], i2[0], c='r', s=20)
                plt.scatter(l[1], l[0], c='b', s=20)
                plt.scatter(ll[1], ll[0], c='g', s=20)
                plt.scatter(br[1], br[0], c='y', s=20)
                    
            # plt lines 
            # baseline
            x_values = np.linspace(0,im.shape[1])
            m1 = (i2[0]-i1[0])/(i2[1]-i1[1])
            b1 = i2[0]-m1*(i2[1])
            y_values = x_values*m1+b1
            plt.plot(x_values, y_values, 'r', linestyle="-", linewidth=1)
            # bony roof line
            x_values = np.linspace(0,im.shape[0])
            m2 = (ll[0]-br[0])/(ll[1]-br[1])
            b2 = ll[0]-m2*(ll[1])
            y_values = x_values*m2+b2
            plt.plot(x_values, y_values, 'y', linestyle="-.",linewidth=1)
            # cartilage roof line
            x_values = np.linspace(im.shape[1],0)
            m3 = (br[0]-l[0])/(br[1]-l[1])
            b3 = l[0]-m3*(l[1])
            y_values = x_values*m3+b3
            plt.plot(x_values, y_values, 'b', linestyle="--",linewidth=1)

            outpath= imgs_path[:-len(imgs_path.split('/')[-1])]+'graf_angle_plot'+'/pixel_rad_'+str(point_radius)
            if not os.path.isdir(outpath):
                os.makedirs(outpath)

            outpath_ab = outpath+'/alpha_beta_plots'
            if not os.path.isdir(outpath_ab):
                os.makedirs(outpath_ab)

            #caclulate angles
            if calc_angles == True:
                #line vectors
                v_baseline = gac.get_vector(i1,i2)
                v_cartroof = gac.get_vector(br,l)
                v_bonyroof = gac.get_vector(br,ll)
                #angles using arccosbeta
                a_rad = np.arccos(np.dot(v_baseline,v_bonyroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_bonyroof)))
                b_rad = np.arccos(np.dot(v_baseline,v_cartroof)/(np.linalg.norm(v_baseline)*np.linalg.norm(v_cartroof)))
                a = math.degrees(a_rad)
                b = math.degrees(b_rad)

                ##PLOTTING ARC##
                # alpha
                xi = (b1 - b2) / (m2 - m1)
                yi = m1 * xi + b1
                intersection = [xi, yi]
                # inter = np.intersect1d(v_baseline,v_bonyroof)
                x, y, xt, yt = gac.plot_theta(intersection, a, v_baseline)
                plt.plot(x,y,color='w',linewidth=0.5)
                plt.text(xt,yt,'a='+str(round(a))+u"\u00b0",color='w')

                #beta
                xi_b = (b3-b1) / (m1-m3)
                yi_b = m1* xi_b + b1
                intersection_b = [xi_b, yi_b]
                x_b, y_b, xt_b, yt_b = gac.plot_theta(intersection_b, b, v_cartroof)
                plt.plot(x_b,y_b,color='w',linewidth=0.5)
                plt.text(xt_b,yt_b,'b='+str(round(b))+u"\u00b0",color='w')

            # place a text box in upper left 
            graf_class_ab, graph_discription_ab = gac.graf_angle(a,b)
            txt_ab = "graf class ab: " + graf_class_ab
            graf_class_a, graph_discription_a = gac.graf_angle(a)
            txt_a = "graf class a: " + graf_class_a
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.05, 0.95, txt_a, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
            ax.text(0.05, 0.90, txt_ab, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)

            plt.imshow(im)
            plt.savefig(outpath_ab+'/'+img_name)

            #get and plot angle range
            outpath_range = outpath+'/theta_ranges/'
            b_range = gac.get_angle_range(point_radius, i1,i2,'r',br,l,'b',im,img_name[:-4]+'_beta',outpath_range)
            a_range = gac.get_angle_range(point_radius, i1,i2,'r',br,ll,'y',im,img_name[:-4]+'_alpha',outpath_range)

            #if save_calculate_angles 
            if save_calculated_angles==True:
                calc_df = calc_df.append({'Patient':img_name[:-4],
                                        'Calculated alpha': a,
                                        'range alpha min': a_range[0],
                                        'range alpha max': a_range[1],                                        ''
                                        'Calculated beta': b,
                                        'range beta min': b_range[0],
                                        'range beta max': b_range[1],
                                        'graf class ab':graf_class_ab, 
                                        'discription ab':graph_discription_ab,
                                        'graf class a':graf_class_a,
                                        'discription a':graph_discription_a},
                                        ignore_index=True)
        else:
            pass

    #save final csv
    calc_df.to_csv(outpath+'/calculated_graf_angles_clinicans.csv')
    #save inputs to txt
    txt_record_inputs = open(outpath+'/inputs.txt', 'w')
    txt_record_inputs.write('Image Input Dir:'+imgs_dir+'\n')
    txt_record_inputs.write('Graf Point Dir:'+graf_points_dir+'\n')
    txt_record_inputs.write('Point Radius:'+ str(point_radius)+'\n')
    txt_record_inputs.close()

if __name__ == '__main__':
    imgs_dir =  "/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/img" #images
    #
    #imgs_dir =  "/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/img" #images

    graf_points_dir = "/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt" #annotations from reviewer - TRUTH
    #graf_points_dir = "/experiments/medimaging/experimentsallisonclement/CompleteLandmarkPackage/output/ddh_512_352/temp_scale_models/images/txt"

    #point raidus will be used for how much to plot surrounding a specific identified point
    #this about this as almost the ground truth radius around specific identified points.
    all_pixle_rad = [2, 3, 4, 5]
    for point_radius in all_pixle_rad:
        calculated_angles = plt_graf_lines(imgs_dir, graf_points_dir, point_radius)    