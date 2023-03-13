import numpy as np
import math as math
from matplotlib import pyplot as plt
import os
import math
class graph_angle_calculations():
    def __init__(self) -> None:
        self.grf_dic = {
            "1": {'a':'>60', 'b':'NA', 'd': 'Normal: Discharge Patient'},
            "2a/2b": {'a':'50-59', 'b':'NA', 'd': 'Normal/Abnormal: Clinical Review -/+ treat'},
            "2c": {'a':'43-49', 'b':'<77', 'd':'Abnormal: Clinical Review + treat'},
            "D": {'a':'43-49', 'b':'>77', 'd': 'Abnormal: Clinical Review + treat'}, 
            "3/4": {'a':'<43', 'b':'Unable to calculate', 'd': 'Abnormal: Clinical Review + treat'},
            }
        pass
            #"2b": {'a':'50-59', 'b':'NA', 'd': 'Abnormal: Clinical Review -/+ treat'},
            #"4": {'a':'<43', 'b':'Unable to calculate', 'd': 'Abnormal: Clinical Review + treat'},


    def get_alpha_category(self,alpha):
        if alpha >= 60:
            return '>60'
        elif alpha > 50 and alpha < 60:
            return'50-59'
        elif alpha > 43 and alpha < 50:
            return'43-49'
        elif alpha <43:
            return '<43'
        else:
            raise ValueError

    def get_beta_category(self,b):
        if b < 77:
            return '<77'
        elif b > 77:
            return'>77'
        else:
            return 'other'

    def graf_angle(self,alpha,beta=None):
        #classification, angle, beta, discrtiption
        alpha = self.get_alpha_category(alpha)
        if beta!=None:
            beta = self.get_beta_category(beta)

        if beta!=None:
            for key in self.grf_dic.items(): 
                if self.grf_dic[key[0]]['a'] == alpha and self.grf_dic[key[0]]['b'] == beta:
                    graf_class = key[0]
                    graf_discription = self.grf_dic[key[0]]['d']
                elif self.grf_dic[key[0]]['a'] == alpha and self.grf_dic[key[0]]['b'] == 'NA':
                    graf_class = key[0]
                    graf_discription = self.grf_dic[key[0]]['d'] + '(BETA NA)'
                elif self.grf_dic[key[0]]['a'] == alpha and self.grf_dic[key[0]]['b'] == 'Unable to calculate':
                    graf_class = key[0]
                    graf_discription = self.grf_dic[key[0]]['d'] + '(Unable to calculate)'
                else:
                    pass
        else:
            for key in self.grf_dic.items(): 
                if self.grf_dic[key[0]]['a'] == alpha:
                    graf_class = key[0]
                    if graf_class == "D" or graf_class == "2c":
                        graf_class = "D/2c"
                    graf_discription = self.grf_dic[key[0]]['d']

            
        return graf_class, graf_discription

    def plot_theta(self,intersection, a, start_vector, direction='clockwise'):
        r = 50
        x0,y0 = intersection

        theta1 = (np.arctan(start_vector[0]/start_vector[1]))
        theta2 = theta1+math.radians(a)

        if direction == 'clockwise':
            t = np.linspace(theta1, theta2, 11)
        else:
            t = np.linspace(theta2+180, theta1+180, 11)

        x = r*np.cos(t) + x0
        y = r*np.sin(t) + y0

        xt = x[6]+5
        yt = y[6]+5

        return x, y, xt, yt
    
    def get_vector(self,p1,p2):
        x1,y1 = p1
        x2,y2 = p2
        v=[x2-x1,y2-y1]
        return v

    def get_division_range(self, range1, range2):
        '''given two ranges, find the max valuse if they are multiplied'''
        range1_min, range1_max = range1
        range2_min, range2_max = range2

        #possible min max values
        ab_min1min2 = range1_min/range2_min
        ab_max1min2 = range1_max/range2_min
        ab_min1max2 = range1_min/range2_max 
        ab_max1max2 = range1_max/range2_max

        max_r = max(ab_min1min2,ab_max1min2,ab_min1max2,ab_max1max2)
        min_r = min(ab_min1min2,ab_max1min2,ab_min1max2,ab_max1max2)

        return max_r, min_r

    def get_multiply_range(self, range1, range2):
        '''given two ranges, find the max valuse if they are multiplied'''
        range1_min, range1_max = range1
        range2_min, range2_max = range2

        #possible min max values
        ab_min1min2 = range1_min*range2_min
        ab_max1min2 = range1_max*range2_min
        ab_min1max2 = range1_min*range2_max 
        ab_max1max2 = range1_max*range2_max

        max_r = max(ab_min1min2,ab_max1min2,ab_min1max2,ab_max1max2)
        min_r = min(ab_min1min2,ab_max1min2,ab_min1max2,ab_max1max2)

        return max_r, min_r


    def get_angle_range(self, point_radius, p1,p2,c1,p3,p4,c2,im,img_name,outpath,plot_range=True):
        '''
        range of angles you can get for three different points moving with variability x
        note: '_u' is upper and '_l' denotes lower ranges '''
        #pixel radius for variability
        r = point_radius

        #get lower ad upper bounds assuming uniform variability of 'r' radius from center point given
        #break down vectors into points
        ##rr fixes the two baseline points to only have a variability of 2 pixels
        rr = 2
        p1x_u, p1x_l = math.floor(p1[0])+rr,math.ceil(p1[0]-rr)
        p2x_u, p2x_l = math.floor(p2[0])+rr,math.ceil(p2[0]-rr)
        p3x_u, p3x_l = math.floor(p3[0])+r,math.ceil(p3[0]-r)
        p4x_u, p4x_l = math.floor(p4[0])+rr,math.ceil(p4[0]-r)
        
        p1y_u, p1y_l = math.floor(p1[1])+rr,math.ceil(p1[1]-rr)
        p2y_u, p2y_l = math.floor(p2[1])+rr,math.ceil(p2[1]-rr)
        p3y_u, p3y_l = math.floor(p3[1])+r,math.ceil(p3[1]-r)
        p4y_u, p4y_l = math.floor(p4[1])+r,math.ceil(p4[1]-r)
        
        #all possible lines from range, and calculate array of all possible a and b values
        if plot_range == True:
            fig2, ax2 = plt.subplots(1, 1)
            #PLOT STATIC POINTS
            plt.scatter(p1[1], p1[0],c=c1, s=20)
            plt.scatter(p2[1], p2[0],c=c1, s=20)
            plt.scatter(p3[1], p3[0],c=c2, s=20)
            plt.scatter(p4[1], p4[0],c=c2, s=20)

            #can set range for plotting but for computational time take first and last in range defined
            it = int(p1x_u-p1x_l-1)
            #arr_theta = np.array([])
            max_theta = 0
            min_theta = 0
            for x1 in range(p1x_l, p1x_u,it):
                for x2 in range(p2x_l, p2x_u,it):
                    for y1 in range(p1y_l, p1y_u,it):
                        for y2 in range(p2y_l, p2y_u,it):
                            x_values_a = np.linspace(0,im.shape[1])
                            m = (x2-x1)/(y2-y1)
                            b = x2-m*y2
                            y_values_a = x_values_a*m+b
                            ax2.plot(x_values_a, y_values_a,c1, linestyle="-", linewidth=0.5)

                            for x3 in range(p3x_l, p3x_u,it):
                                for x4 in range(p4x_l, p4x_u,it):
                                    for y3 in range(p3y_l, p3y_u,it):
                                        for y4 in range(p4y_l, p4y_u,it):
                                            x_values = np.linspace(0,im.shape[0])
                                            if (y4-y3)==0:
                                                y3=y3+0.0001
                                            m2 = (x4-x3)/(y4-y3)
                                            b2 = x4-m2*y4
                                            y_values = x_values*m2+b2
                                            ax2.plot(x_values, y_values,c2, linestyle="-", linewidth=0.09)
                                            
                                            #calculate theta
                                            v_1 = self.get_vector([x1,y1],[x2,y2])
                                            v_2 = self.get_vector([x3,y3],[x4,y4])

                                            #angles using arccosbeta
                                            theta_rad = np.arccos(np.dot(v_1,v_2)/(np.linalg.norm(v_1)*np.linalg.norm(v_2)))
                                            theta = math.degrees(theta_rad)

                                            if max_theta == 0:
                                                max_theta = theta
                                                min_theta = theta
                                                
                                                m_max_1 = m
                                                b_max_1 = b
                                                m_max_2 = m2
                                                b_max_2 = b2

                                                m_min_1 = m
                                                b_min_1 = b
                                                m_min_2 = m2
                                                b_min_2 = b2


                                            if theta > max_theta:
                                                max_theta = theta
                                                m_max_1 = m
                                                b_max_1 = b
                                                m_max_2 = m2
                                                b_max_2 = b2

                                            if theta < min_theta:
                                                min_theta = theta
                                                m_min_1 = m
                                                b_min_1 = b
                                                m_min_2 = m2
                                                b_min_2 = b2


            print('plot range:', min_theta, 'to', max_theta)
            ax2.imshow(im)
            txt = "theta range: " + str(round(min_theta,1)) + ' - '+str(round(max_theta,1))
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, fontsize=10,verticalalignment='top', bbox=props)
            if not os.path.isdir(outpath):
                os.mkdir(outpath)
                os.mkdir(outpath+'/ranges/')

            plt.savefig(outpath+'/ranges/range_'+img_name)

            #PLOT MAX AND MIN ONLY
            fig3, ax3 = plt.subplots(1,2)
            ax3[1].title.set_text('max')
            x_values = np.linspace(0,im.shape[0])
            y_values = x_values*m_max_1+b_max_1
            ax3[1].plot(x_values, y_values,c1, linestyle="-", linewidth=1)
            x_values = np.linspace(0,im.shape[0])
            y_values = x_values*m_max_2+b_max_2
            ax3[1].plot(x_values, y_values,c2, linestyle="-", linewidth=1)

            ax3[0].title.set_text('min')
            x_values = np.linspace(0,im.shape[0])
            y_values = x_values*m_min_1+b_min_1
            ax3[0].plot(x_values, y_values,c1, linestyle="-", linewidth=1)
            x_values = np.linspace(0,im.shape[0])
            y_values = x_values*m_min_2+b_min_2
            ax3[0].plot(x_values, y_values,c2, linestyle="-", linewidth=1)
        
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax3[1].scatter(p1[1], p1[0],c=c1, s=20)
            ax3[1].scatter(p2[1], p2[0],c=c1, s=20)
            ax3[1].scatter(p3[1], p3[0],c=c2, s=20)
            ax3[1].scatter(p4[1], p4[0],c=c2, s=20)
            ax3[1].imshow(im)
            txt = "theta: " + str(round(max_theta))+u"\u00b0"
            ax3[1].text(0.05, 0.95, txt, transform=ax3[1].transAxes, fontsize=10,verticalalignment='top', bbox=props)
            
            ax3[0].scatter(p1[1], p1[0],c=c1, s=20)
            ax3[0].scatter(p2[1], p2[0],c=c1, s=20)
            ax3[0].scatter(p3[1], p3[0],c=c2, s=20)
            ax3[0].scatter(p4[1], p4[0],c=c2, s=20)
            ax3[0].imshow(im)
            txt = "theta: " + str(round(min_theta))+u"\u00b0"
            ax3[0].text(0.05, 0.95, txt, transform=ax3[0].transAxes, fontsize=10,verticalalignment='top', bbox=props)
            
            #plt.show()
            #Save
            if not os.path.isdir(outpath+'/min-max'):
                os.mkdir(outpath+'/min-max')

            plt.savefig(outpath+'/min-max/min-max_'+img_name)
            plt.close()

            
        # dot product between vector 1 and vector 2 (assume vector 1 is p1, p2 and vector 2 is p1, p3)
        # will be used to calcualte angles
        # theta = np.arccos(((p2[0]-p1[0])*(p3[0]-p4[0])+(p2[1]-p1[1])*(p3[1]-p4[1]))/abs(v1)*abs(v2))
        # so we must internally find the ranges within to find range of theta

        ### arccos numerator
        range_p2p1_x = [p2x_l-p1x_u, p2x_u-p1x_l]
        range_p3p4_x = [p4x_l-p3x_u, p4x_u-p3x_l]
        range_p2p1_y = [p2y_l-p1y_u, p2y_u-p1y_l]
        range_p3p4_y = [p4y_l-p3y_u, p4y_u-p3y_l]

        #find max range since its mutliplying (check 4 options)
        range_xx_l, range_xx_u = self.get_multiply_range(range_p2p1_x,range_p3p4_x)
        range_yy_l, range_yy_u = self.get_multiply_range(range_p2p1_y,range_p3p4_y)

        #add mutltiplied ranges
        range_theta_rad_numerator = [range_xx_l+range_yy_l, range_xx_u+range_yy_u]
        
        ###get absolute ranges for arccos denominator 
        #square differences
        range_p2p1_x_tmp_l, range_p2p1_x_tmp_u = self.get_multiply_range(range_p2p1_x,range_p2p1_x)
        range_p3p4_x_tmp_l, range_p3p4_x_tmp_u = self.get_multiply_range(range_p3p4_x,range_p3p4_x)
        range_p2p1_y_tmp_l, range_p2p1_y_tmp_u = self.get_multiply_range(range_p2p1_y,range_p2p1_y)
        range_p3p4_y_tmp_l, range_p3p4_y_tmp_u = self.get_multiply_range(range_p3p4_y,range_p3p4_y)

        #add sqr_p2p1_range
        range_p2p1_xy_diff_l, range_p2p1_xy_diff_u = [range_p2p1_x_tmp_l+range_p2p1_y_tmp_l,range_p2p1_x_tmp_u+range_p2p1_y_tmp_u]
        range_p3p4_xy_diff_l, range_p3p4_xy_diff_u = [range_p3p4_x_tmp_l+range_p3p4_y_tmp_l,range_p3p4_x_tmp_u+range_p3p4_y_tmp_u]

        #square roots 
        range_p1p2_xy_sqr = [math.sqrt(range_p2p1_xy_diff_l),math.sqrt(range_p2p1_xy_diff_u)]
        range_p3p4_xy_sqr = [math.sqrt(range_p3p4_xy_diff_l), math.sqrt(range_p3p4_xy_diff_u)]

        #multiply square roots
        range_theta_rad_denominator = self.get_multiply_range(range_p1p2_xy_sqr,range_p3p4_xy_sqr)

        range_theta_rad_l, range_theta_rad_u = self.get_division_range(range_theta_rad_numerator,range_theta_rad_denominator)

        theta_range = [np.arccos(range_theta_rad_l),np.arccos(range_theta_rad_u)]
        theta_range = [math.degrees(theta_range[0]),math.degrees(theta_range[1])]
        print('calculated range:',theta_range)

        theta_plot_range = (min_theta+max_theta)/2
        plt.close()
        return min_theta,max_theta
