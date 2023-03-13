import matplotlib.pyplot as plt
import csv
import seaborn as sns 
import pandas as pd
import numpy as np

if __name__=='__main__':
    file_path = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/graf_angle_plot_uniform/pixel_rad_1/calculated_graf_angles_clinicans.csv'
    _df = pd.read_csv(file_path)

    X = 'Calculated alpha'
    Y = 'Calculated beta'
    C = 'graf class ab'

    for key in _df.keys():
        if key==X or key==Y or key==C:
            pass
        else:
            _df =_df.drop([key], axis=1)
        
    #plot classes
    classes = ['1', '2a/2b','2c','D','3/4']
    cc = 0
    for c in classes:
        cc=cc+1
        _df.loc[_df[C] == c] = _df.loc[_df[C] == c].replace(c, cc)
    
    ax = _df.plot.scatter(X,Y,c=C,colormap='viridis')
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    x=np.linspace(0,120,120)
    a_limit = np.full(120,60)
    plt.plot(a_limit,x,'r--', label='Threshold')
    ax.set_title('Clinician Classification')
    plt.rcParams["figure.figsize"] = (5,5)
    plt.show()

#### ALPHA
    file_path = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/graf_angle_plot_uniform/pixel_rad_3/calculated_graf_angles_clinicans.csv'
    _df = pd.read_csv(file_path)

    X = 'range alpha min'
    Y = 'range alpha max'
    C = 'graf class ab'

    for key in _df.keys():
        if key==X or key==Y or key==C:
            pass
        else:
            _df =_df.drop([key], axis=1)
        
    #plot classes
    classes = ['1', '2a/2b','2c','D','3/4']
    cc = 0
    for c in classes:
        cc=cc+1
        _df.loc[_df[C] == c] = _df.loc[_df[C] == c].replace(c, cc)
    
    ax = _df.plot.scatter(X,Y,c=C,colormap='viridis')
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    ax.set_title('Alpha: 3 Pixel Variation')
    plt.rcParams["figure.figsize"] = (5,5)
    plt.show()
    diff=(abs(_df[X]-_df[Y])).mean()
    diff_std=(abs(_df[X]-_df[Y])).std()
    print('avg', diff,'std' ,diff_std)

#### beta
    file_path = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/graf_angle_plot_uniform/pixel_rad_1/calculated_graf_angles_clinicans.csv'
    _df = pd.read_csv(file_path)

    X = 'range beta min'
    Y = 'range beta max'
    C = 'graf class ab'

    for key in _df.keys():
        if key==X or key==Y or key==C:
            pass
        else:
            _df =_df.drop([key], axis=1)
        
    #plot classes
    classes = ['1', '2a/2b','2c','D','3/4']
    cc = 0
    for c in classes:
        cc=cc+1
        _df.loc[_df[C] == c] = _df.loc[_df[C] == c].replace(c, cc)
    
    ax = _df.plot.scatter(X,Y,c=C,colormap='viridis')
    ax.set_xlim(0,120)
    ax.set_ylim(0,120)
    ax.set_title('Beta - 1 Pixel Variation')
    plt.rcParams["figure.figsize"] = (10,10)
    plt.show()
    diff=(abs(_df[X]-_df[Y])).mean()
    diff_std=(abs(_df[X]-_df[Y])).std()
    print('avg', diff,'std' ,diff_std)

    
    
    
#### tables for pixels
    file_path = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/graf_angle_plot_uniform/pixel_rad_2/calculated_graf_angles_clinicans.csv'
    _df = pd.read_csv(file_path)
    _df=_df.drop(['Unnamed: 0'],axis=1)
    _df=_df.drop(['Patient'],axis=1)
    _df=_df.drop(['discription a'],axis=1)
    _df=_df.drop(['discription ab'],axis=1)
    _df=_df.drop(['graf class a'],axis=1)

    _df_dif_a = _df['range alpha max']-_df['range alpha min']
    _df_dif_b = _df['range beta max']-_df['range beta min']

    _df['dif_a'] = _df_dif_a 
    _df['dif_b'] = _df_dif_b
    classes = ['1', '2a/2b','2c','D','3/4']
    cc = 0

    df_mean_std = pd.DataFrame()
    for c in classes:
        cc=cc+1
        mean = (_df.loc[_df[C] == c]).mean()
        std = (_df.loc[_df[C] == c]).std()
        mean['graf class ab']=c
        mean['calc'] = 'avg'
        std['graf class ab']=c
        std['calc'] = 'std'
        
        df_mean_std = df_mean_std.append(mean, ignore_index = True)
        df_mean_std = df_mean_std.append(std, ignore_index = True)
    
    print(df_mean_std)




    