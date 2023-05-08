import matplotlib.pyplot as plt
import csv
import seaborn as sns 
import pandas as pd
import numpy as np

class plot_from_excel():
    ''' This class takes an input excel file and plots the relationship between x and y of each data point the class from the third column
     is set to different colors. This will help show the relation ship between x, y and class.'''
    def __init__(self, file_path):
        self.file_path = file_path

        #plot limits and figure
        self.x_lim = 120
        self.y_lim = 120
        self.fig_size = 5

        return
    
    def plot_file(self, classes, X, Y, C, title, threshold_line = None):
        _df = pd.read_csv(self.file_path)

        for key in _df.keys():
            if key==X or key==Y or key==C:
                pass
            else:
                _df =_df.drop([key], axis=1)
            
        #plot classes
        cc = 0
        for c in classes:
            cc=cc+1
            _df.loc[_df[C] == c] = _df.loc[_df[C] == c].replace(c, cc)
        
        ax = _df.plot.scatter(X,Y,c=C,colormap='viridis')
        
        ax.set_xlim(0,self.x_lim)
        ax.set_ylim(0,self.y_lim)

        if threshold_line != None:     
            x=np.linspace(0,self.x_lim,self.x_lim)
            a_limit = np.full(self.x_lim,threshold_line)
            plt.plot(a_limit,x,'r--',label='Threshold')

        ax.set_title(title)
        plt.rcParams["figure.figsize"] = (self.fig_size,self.fig_size)
        plt.show()

        return
    
    def get_class_stdmean(self, classes, col_name):
        '''gets table of classes mean and std'''
        _df = pd.read_csv(self.file_path)
        
        try:
            #do for graf data 
            _df=_df.drop(['Unnamed: 0'],axis=1)
            _df=_df.drop(['Patient'],axis=1)
            _df=_df.drop(['discription a'],axis=1)
            _df=_df.drop(['discription ab'],axis=1)
            _df=_df.drop(['graf class a'],axis=1)
        except: 
            pass

        _df_dif_a = _df['range alpha max']-_df['range alpha min']
        _df_dif_b = _df['range beta max']-_df['range beta min']

        _df['dif_a'] = _df_dif_a 
        _df['dif_b'] = _df_dif_b
        
        C = col_name
        cc = 0

        df_mean_std = pd.DataFrame()
        for c in classes:
            cc=cc+1
            mean = (_df.loc[_df[C] == c]).mean()
            std = (_df.loc[_df[C] == c]).std()
            mean[C]=c
            mean['calc'] = 'avg'
            std[C]=c
            std['calc'] = 'std'
            
            df_mean_std = df_mean_std.append(mean, ignore_index = True)
            df_mean_std = df_mean_std.append(std, ignore_index = True)
        
        print(df_mean_std)
        return

def main():
    #this is an example plotting x,y, class for 3 pixel varation
    file_path = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/graf_angle_plot_uniform/pixel_rad_3/calculated_graf_angles_clinicans.csv'
    
    classes = ['1', '2a/2b','2c','D','3/4']

    title = 'Clinician Classification'
    X,Y,C = 'Calculated alpha','Calculated beta','graf class ab'
    plot_from_excel(file_path).plot_file(classes, X, Y, C, title, threshold_line=60)

    title = 'Alpha: 3 Pixel Variation'
    X,Y,C = 'range alpha min','range alpha max','graf class ab'
    plot_from_excel(file_path).plot_file(classes, X, Y, C, title)

    title = 'Beta: 3 Pixel Variation'
    X,Y,C = 'range beta min','range beta max','graf class ab'
    plot_from_excel(file_path).plot_file(classes, X, Y, C, title)    

    #get mean and standard deivation
    col_name = 'graf class ab'
    plot_from_excel(file_path).get_class_stdmean(classes, col_name)


if __name__=='__main__':
    main()
