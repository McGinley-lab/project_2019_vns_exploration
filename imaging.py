import sys, os, glob
import numpy as np
import scipy as sp
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from joblib import Memory
from joblib import Parallel, delayed

from IPython import embed as shell

import tools_mcginley 
from tools_mcginley import utils

import imp
import vns_analyses

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()


def lavaan(df, X, Y, M, C=False, zscore=True,):
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter


    df['X'] = df[X].copy()
    df['Y'] = df[Y].copy()
    df['M'] = df[M].copy()

    # convert datafame:
    with localconverter(default_converter + pandas2ri.converter) as cv:
        df_r = pandas2ri.py2ri(df)
    
    # load lavaan and data:
    robjects.r('library(lavaan)')
    robjects.globalenv["data_s"] = df_r

    print(robjects.r("typeof(data_s$X)"))
    print(robjects.r("typeof(data_s$Y)"))
    print(robjects.r("typeof(data_s$M)"))

    # zscore:
    if zscore:
        robjects.r("data_s['X'] = (data_s[['X']] - mean(data_s[['X']])) / sd(data_s[['X']])")
        robjects.r("data_s['Y'] = (data_s[['Y']] - mean(data_s[['Y']])) / sd(data_s[['Y']])")
        robjects.r("data_s['M'] = (data_s[['M']] - mean(data_s[['M']])) / sd(data_s[['M']])")

    if C == True:
        robjects.r("model = 'Y ~ c*X + C\nM ~ a*X + C\nY ~ b*M\nab := a*b\ntotal := c + (a*b)'")
        robjects.r('fit = sem(model, data=data_s, se="bootstrap")')
        c = np.array(pandas2ri.ri2py(robjects.r('coef(fit)')))[np.array([0,2,4])]
    else:
        robjects.r("model = 'Y ~ c*X\nM ~ a*X\nY ~ b*M\nab := a*b\ntotal := c + (a*b)'")
        robjects.r('fit = sem(model, data=data_s, se="bootstrap")')
        c = pandas2ri.ri2py(robjects.r('coef(fit)')[0:3])
    
    c = robjects.r("parameterEstimates(fit)")
    # c = robjects.r("summary(fit)")

    return c

def check_cell(df, conditions, min_timepoints=5, max_pvalue=0.01, start=0, end=5):
    times = np.array(df.columns, dtype=float)
    times = times[(times>=start)&(times>=start)]
    sig_timepoints = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        
        # check single conditions against 0:
        for cond in conditions:
            values = df.loc[(df.index.get_level_values('outcome')==cond), times==t]
            p = sp.stats.ttest_1samp(values, 0)[1]
            if (p <= max_pvalue):
                sig_timepoints[i] = True
        
        # check contrast:
        values_0 = df.loc[(df.index.get_level_values('outcome')==conditions[0]), times==t]
        values_1 = df.loc[(df.index.get_level_values('outcome')==conditions[1]), times==t]
        p = sp.stats.ttest_ind(values_0, values_1)[1]
        if (p <= max_pvalue):
            sig_timepoints[i] = True

    if sum(sig_timepoints) >= min_timepoints:
        return True
    else:
        return False

def cell_selection(df, conditions, min_timepoints=5, max_pvalue=0.01, start=0, end=5, n_jobs=4):
    cells = np.unique(df.index.get_level_values('cell'))
    selection = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(check_cell)(d, conditions, min_timepoints, max_pvalue, start, end) for cell, d in df.groupby('cell')))
    return cells[selection]

def create_calcium_df(fluorescence, time, df, locking='trial_start_time', start=-1, end=6):

    dfs = []
    for i in range(fluorescence.shape[0]):
        responses = []
        for t in range(df.shape[0]):
            # frame = utils.find_nearest(time, df['trial_start_time'].iloc[t])
            
            # frame = utils.find_nearest(time, df[locking].iloc[t])
            # r = (fluorescence.iloc[i, frame+int(start*fs):frame+int(end*fs)] - b) / b
            frame = utils.find_nearest(time, df[locking].iloc[t])
            b = fluorescence.iloc[i, frame-int(0.5*fs):frame].mean()
            r = fluorescence.iloc[i, frame+int(start*fs):frame+int(end*fs)] - b
            rr = np.repeat(np.NaN, (frame+int(end*fs)) - (frame+int(start*fs)))
            rr[0:len(r)] = r
            responses.append(rr)
        responses = pd.DataFrame(np.vstack(responses))
        responses.columns = start + np.cumsum([1/fs for _ in range(responses.shape[1])]) - (1/fs)
        responses['cell'] = i
        # responses['trial'] = np.array(df['trial'])
        # responses['outcome'] = np.array(df['outcome'])
        # responses['block_type'] = np.array(df['block_type'])
        # responses['trial_in_block'] = np.array(df['trial_in_block'])
        dfs.append(responses)

    df_calcium = pd.concat(dfs)
    df_calcium = df_calcium.set_index(['cell'])

    return df_calcium

def cluster_df(df):

    import scipy
    import scipy.cluster.hierarchy as sch

    X = df.corr().values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    return df

def plot_correlation_matrix(df, mask_triangle=True):

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    if mask_triangle:
        mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-.9, vmax=.9, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    return fig

def plot_corr(corr, size=3):
    
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)

    mask = np.array(np.tri(corr.shape[0], k=-1), dtype=bool).T
    corr.values[mask] = np.NaN
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))

    cax = ax.matshow(corr, vmin=-1, vmax=1, cmap=cmap)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    columns = corr.columns
    locs = [np.where(columns==c)[0] for c in np.unique(columns)]

    for l in locs:
        # if len(l) > 1:
        print(l)
        rect = Rectangle((l[0]-0.5,l[0]-0.5),l[-1]-l[0]+1,l[-1]-l[0]+1, linewidth=1, edgecolor='blue', facecolor='None')
        ax.add_patch(rect)

    return fig

def hierarchical_cluster(df, min_r=0.8, include=None):

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    corr = df.loc[include,:].corr()

    dissimilarity = 1 - np.abs(corr)
    hierarchy = linkage(squareform(dissimilarity), method='average')
    labels = fcluster(hierarchy, 1-min_r, criterion='distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(labels)))]
    df = df.reindex(columns, axis=1)

    df.columns = labels[np.argsort(labels)]

    return df

def cluster_corr(df):

    import scipy.cluster.hierarchy as sch

    X = df.corr().values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 0.5*d.max(), 'distance')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    
    return df

def plot_motion(df, motion_cutoff=2):
    motion_ind = (abs(df['motion_x'])<motion_cutoff) & (abs(df['motion_y'])<motion_cutoff)
    plt_nr = 1
    fig = plt.figure(figsize=(6,6))
    for m, ylim, xlim in zip(['velocity', 'motion_x', 'motion_y'], [(-0.4,0.8), (-0.4,0.8), (-0.4,0.8)], [(-0.2,0.2), (-0.5,2.5), (-0.5,2.5)]):
        ax = fig.add_subplot(3,3,plt_nr)
        sns.regplot(df.loc[motion_ind, m], df.loc[motion_ind, 'calcium'])
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        plt_nr += 1
        ax = fig.add_subplot(3,3,plt_nr)
        sns.regplot(df.loc[(df['velocity']<velocity_cutoff)&motion_ind, m], df.loc[(df['velocity']<velocity_cutoff)&motion_ind, 'calcium'])
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        plt_nr += 1
        ax = fig.add_subplot(3,3,plt_nr)
        sns.regplot(df.loc[(df['velocity']>velocity_cutoff)&motion_ind, m], df.loc[(df['velocity']>velocity_cutoff)&motion_ind, 'calcium'])
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        plt_nr += 1
    plt.tight_layout()
    return fig

def cross_validate(df, start=1, end=11, p_cutoff=0.05, baseline=True):

    if baseline:
        x = df.columns
        df = df - np.atleast_2d(df.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)).T
       
    epochs = []
    for trial in df.index.get_level_values('trial').unique():
        
        # good cells:
        df_ = df.loc[df.index.get_level_values('trial')!=trial,:]
        t = np.zeros(len(df_.index.get_level_values('cell').unique()))
        p = np.zeros(len(df_.index.get_level_values('cell').unique()))
        for i, c in enumerate(df_.index.get_level_values('cell').unique()):
            scalars = df_.loc[df_.index.get_level_values('cell')==c,(df_.columns>start)&(df_.columns<end)].mean(axis=1)
            t[i], p[i] = sp.stats.ttest_1samp(scalars, 0)
        good_cells = df_.index.get_level_values('cell').unique()[(p<p_cutoff)&(t>0)]
        
        # compute response:
        epochs.append(df.loc[(df.index.get_level_values('trial')==trial) & df.index.get_level_values('cell').isin(good_cells),:].groupby(['subj_idx', 'session', 'trial', 'amplitude', 'rate', 'width']).mean())
    
    # # get good cell indices:
    # t = np.zeros(len(df.index.get_level_values('cell').unique()))
    # p = np.zeros(len(df.index.get_level_values('cell').unique()))
    # t_even = np.zeros(len(df.index.get_level_values('cell').unique()))
    # p_even = np.zeros(len(df.index.get_level_values('cell').unique()))
    # t_odd = np.zeros(len(df.index.get_level_values('cell').unique()))
    # p_odd = np.zeros(len(df.index.get_level_values('cell').unique()))
    # for i, c in enumerate(df.index.get_level_values('cell').unique()):
    #     scalars = df.loc[df.index.get_level_values('cell')==c,(df.columns>start)&(df.columns<end)].mean(axis=1)
    #     t[i], p[i] = sp.stats.ttest_1samp(scalars, 0)
    #     t_even[i], p_even[i] = sp.stats.ttest_1samp(scalars.loc[(scalars.index.get_level_values('trial')%2)==0,:], 0)
    #     t_odd[i], p_odd[i] = sp.stats.ttest_1samp(scalars.loc[(scalars.index.get_level_values('trial')%2)==1,:], 0)
    # good_cells = df.index.get_level_values('cell').unique()[(p<p_cutoff)&(t>0)]
    # good_cells_even = df.index.get_level_values('cell').unique()[(p_even<p_cutoff)&(t_even>0)]
    # good_cells_odd = df.index.get_level_values('cell').unique()[(p_odd<p_cutoff)&(t_odd>0)]
    
    # # collapse:
    # # df_even = df.loc[((df.index.get_level_values('trial')%2)==0) & df.index.get_level_values('cell').isin(good_cells_odd)].groupby(['subj_idx', 'session', 'trial', 'amplitude', 'rate', 'width']).mean()
    # # df_odd = df.loc[((df.index.get_level_values('trial')%2)==1) & df.index.get_level_values('cell').isin(good_cells_even)].groupby(['subj_idx', 'session', 'trial', 'amplitude', 'rate', 'width']).mean()
    # df_even = df.loc[((df.index.get_level_values('trial')%2)==0) & df.index.get_level_values('cell').isin(good_cells)].groupby(['subj_idx', 'session', 'trial', 'amplitude', 'rate', 'width']).mean()
    # df_odd = df.loc[((df.index.get_level_values('trial')%2)==1) & df.index.get_level_values('cell').isin(good_cells)].groupby(['subj_idx', 'session', 'trial', 'amplitude', 'rate', 'width']).mean()
    # df = pd.concat((df_even, df_odd)).sort_values('trial')
    
    return pd.concat(epochs).sort_values('trial')
    
# raw data dir:
preprocess = False
parallel = True
n_jobs = 12
backend = 'loky'

# parameters:
fs = 50

# signal_to_use = 'spikes'
signal_to_use = 'fluorescence'

raw_dir = '/media/external4/2p_imaging/vns/'
imaging_dir = '/media/internal1/vns/'
# imaging_dir = '/home/jwdegee/temp/'
base_dir = '/home/jwdegee/'
project_name = 'vns_exploration'
project_dir = os.path.join(base_dir, project_name)
fig_dir = os.path.join(base_dir, project_name, 'figures', 'imaging')
data_dir = os.path.join(project_dir, 'data', 'imaging')

subjects = {
    'C7A2': ['1',],
    'C7A6': ['1', '7', '8',],  
    'C1772': ['6', '7', '8',],
    'C1773': ['5', '6', '7', '8', '10'],
}

if preprocess:
    
    # imaging:
    tasks = []
    for subj in subjects:
        print(subj)
        sessions = subjects[subj]
        for ses in sessions:
            tasks.append((raw_dir, imaging_dir, fig_dir, subj, ses))
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.analyse_imaging_session)(*task) for task in tasks)
    epochs_c = pd.concat([res[i][0] for i in range(len(res))], axis=0)
    epochs_x = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_y = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][6] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_c.to_hdf(os.path.join(data_dir, 'epochs_c.hdf'), key='calcium')
    epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x_motion')
    epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y_motion')
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')

epochs_c = pd.read_hdf(os.path.join(data_dir, 'epochs_c.hdf'), key='calcium')
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x_motion')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y_motion')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')


# settings
motion_cutoff = 2
velocity_cutoff = (-0.005, 0.005)
blink_cutoff = 0.1

# add charge:
epochs_c['charge'] = epochs_c.index.get_level_values('amplitude') * epochs_c.index.get_level_values('width')
epochs_c['charge_ps'] = epochs_c.index.get_level_values('amplitude') * epochs_c.index.get_level_values('width') * epochs_c.index.get_level_values('rate')
epochs_c = epochs_c.set_index(['charge', 'charge_ps'], append=True)

# meta data:
df_meta = epochs_c.index.to_frame(index=False)

# make scalars:
# -------------

timewindows = {

            'velocity_-3' : [(-30.1,-30), (-40.1,-40)],
            'velocity_-2' : [(-20.1,-20), (-30.1,-30)],
            'velocity_-1' : [(-10.1,-10), (-20.1,-20)],
            'velocity_0' : [(-0.1,0), (-10.1,-10)],
            'velocity_1' : [(9.9,10), (-0.1,0)],

            'pupil_-3' : [(-27.5, -22.5), (None, None)],  #
            'pupil_-2' : [(-20, -15), (None, None)],      #
            'pupil_-1' : [(-12.5, -7.5), (None, None)],   #
            'pupil_0' : [(-5, 0), (None, None)],          #
            'pupil_1' : [(0, 50), (None, None)],

            'blink_-3' : [(-27.5, -22.5), (None, None)],  #
            'blink_-2' : [(-20, -15), (None, None)],      #
            'blink_-1' : [(-12.5, -7.5), (None, None)],   #
            'blink_0' : [(-5, 0), (None, None)],          #
            'blink_1' : [(2.5, 7.5), (None, None)],

            'eyelid_-3' : [(-27.5, -22.5), (None, None)],  #
            'eyelid_-2' : [(-20, -15), (None, None)],      #
            'eyelid_-1' : [(-12.5, -7.5), (None, None)],   #
            'eyelid_0' : [(-5, 0), (None, None)],          #
            'eyelid_1' : [(0, 25), (None, None)],
            
            'calcium_-3' : [(-27.5, -22.5), (None, None)],  #
            'calcium_-2' : [(-20, -15), (None, None)],      #
            'calcium_-1' : [(-12.5, -7.5), (None, None)],   #
            'calcium_0' : [(-5, 0), (None, None)],          #
            'calcium_1' : [(2.5, 7.5), (None, None)],

            }

epochs = {
            'velocity' : epochs_v,
            'pupil' : epochs_p,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            'calcium' : epochs_c,
            }

ylims = {
            'velocity' : (-0.1, 0.5),
            'walk' : (0,1),
            'pupil' : (-0.05, 0.5),
            'eyelid' : (-0.01, 0.1),
            'blink' : (0, 0.3),
            # 'eyemovement' : (0, 0.2),
            'calcium' : (0, 0.8),
            }

for key in timewindows.keys():
    x = epochs[key.split('_')[0]].columns
    window1, window2 = timewindows[key]
    resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1).values
    if window2[0] == None: 
        df_meta[key] = resp
    else:
        baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1).values
        df_meta[key] = resp-baseline    
    # if key == 'blink':
    #     df_meta['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)

df_meta['walk_-3'] = ((df_meta['velocity_-3'] < velocity_cutoff[0])|(df_meta['velocity_-3'] > velocity_cutoff[1])).astype(int)
df_meta['walk_-2'] = ((df_meta['velocity_-2'] < velocity_cutoff[0])|(df_meta['velocity_-2'] > velocity_cutoff[1])).astype(int)
df_meta['walk_-1'] = ((df_meta['velocity_-1'] < velocity_cutoff[0])|(df_meta['velocity_-1'] > velocity_cutoff[1])).astype(int)
df_meta['walk_0'] = ((df_meta['velocity_0'] < velocity_cutoff[0])|(df_meta['velocity_0'] > velocity_cutoff[1])).astype(int)
df_meta['walk_1'] = ((df_meta['velocity_1'] < velocity_cutoff[0])|(df_meta['velocity_1'] > velocity_cutoff[1])).astype(int)

# throw away weird trial:
remove = np.array(
        (df_meta['subj_idx'] == 'C1773') & (df_meta['session']=='8') & (df_meta['trial']==17) | # weird trial with crazy spike
        (df_meta['width']==0.8) | # should not be...
        (np.isnan(np.array(epochs_c.loc[:,(epochs_c.columns>0)&(epochs_c.columns<1)].mean(axis=1)))) | # missing calcium data...
        np.isnan(df_meta['pupil_1'])
        )
df_meta = df_meta.loc[~remove,:].reset_index(drop=True)
epochs_x = epochs_x.loc[~remove,:].reset_index(drop=True)
epochs_y = epochs_y.loc[~remove,:].reset_index(drop=True)
epochs_v = epochs_v.loc[~remove,:].reset_index(drop=True)
epochs_p = epochs_p.loc[~remove,:].reset_index(drop=True)
epochs_l = epochs_l.loc[~remove,:].reset_index(drop=True)
epochs_c = epochs_c.loc[~remove,:]

# indices:
ind_clean_w = ~(np.isnan(df_meta['velocity_1'])|np.isnan(df_meta['velocity_0']))

group3 = (
        ((epochs_c.index.get_level_values('amplitude')==0.9)&(epochs_c.index.get_level_values('width')==0.4)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.9)&(epochs_c.index.get_level_values('width')==0.2)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.7)&(epochs_c.index.get_level_values('width')==0.4))
)

group2 = (
        ((epochs_c.index.get_level_values('amplitude')==0.9)&(epochs_c.index.get_level_values('width')==0.1)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.7)&(epochs_c.index.get_level_values('width')==0.2)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.7)&(epochs_c.index.get_level_values('width')==0.1)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.5)&(epochs_c.index.get_level_values('width')==0.4)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.5)&(epochs_c.index.get_level_values('width')==0.2)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.3)&(epochs_c.index.get_level_values('width')==0.4))
)

group1 = (
        ((epochs_c.index.get_level_values('amplitude')==0.5)&(epochs_c.index.get_level_values('width')==0.1)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.3)&(epochs_c.index.get_level_values('width')==0.2)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.3)&(epochs_c.index.get_level_values('width')==0.1)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.1)&(epochs_c.index.get_level_values('width')==0.4)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.1)&(epochs_c.index.get_level_values('width')==0.2)) | 
        ((epochs_c.index.get_level_values('amplitude')==0.1)&(epochs_c.index.get_level_values('width')==0.1))
)

epochs_c['group'] = 0
epochs_c.loc[group1, 'group'] = 1
epochs_c.loc[group2, 'group'] = 2
epochs_c.loc[group3, 'group'] = 3
epochs_c = epochs_c.set_index('group', append=True)

# correct scalars:
df_meta, figs = vns_analyses.correct_scalars(df_meta, group=np.ones(df_meta.shape[0], dtype=bool), velocity_cutoff=velocity_cutoff, ind_clean_w=ind_clean_w)
figs[0].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean1.pdf'))
figs[1].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean2.pdf'))
figs[2].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean1.pdf'))
figs[3].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean2.pdf'))
figs[4].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean1.pdf'))
figs[5].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean2.pdf'))
figs[6].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean3.pdf'))
figs[7].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean4.pdf'))
figs[8].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean1.pdf'))
figs[9].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean2.pdf'))
figs[10].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean3.pdf'))
figs[11].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean4.pdf'))



#####################################################################################################

# # cross validate:
# epochs = []
# removed = []
# for (subj, ses), epoch in epochs_c.groupby(['subj_idx', 'session']):
#     print(subj, ses)
#     epoch = cross_validate(epoch, start=1, end=11, baseline=True)
#     if epoch.shape[0] == 0:
#         removed.append((subj, ses))
#         print('removed {}!'.format(removed[-1]))
#     else:
#         epochs.append(epoch)
# epochs_c = pd.concat(epochs)

# # update rest:
# for remove in removed:
#     ind = np.array((df_meta['subj_idx'] == remove[0]) & (df_meta['session'] == remove[1]))
#     df_meta = df_meta.loc[~ind,:].reset_index(drop=True)
#     epochs_v = epochs_v.loc[~ind,:].reset_index(drop=True)
#     epochs_x = epochs_x.loc[~ind,:].reset_index(drop=True)
#     epochs_y = epochs_y.loc[~ind,:].reset_index(drop=True)

####################################################################################################

# image position scalars:
x = epochs_x.columns
image_pos_x = epochs_x.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)
x = epochs_y.columns
image_pos_y = epochs_y.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)

# subtract baselines:
x = epochs_c.columns
epochs_c = epochs_c - np.atleast_2d(epochs_c.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_x.columns
epochs_x = epochs_x - np.atleast_2d(epochs_x.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_y.columns
epochs_y = epochs_y - np.atleast_2d(epochs_y.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_p.columns
epochs_p = epochs_p - np.atleast_2d(epochs_p.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_l.columns
epochs_l = epochs_l - np.atleast_2d(epochs_l.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_v.columns
epochs_v = epochs_v - np.atleast_2d(epochs_v.loc[:,(x>=-0.1)&(x<=-0.0)].mean(axis=1)).T

epochs = {
            'velocity' : epochs_v,
            'pupil' : epochs_p,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            'calcium' : epochs_c,
            }

# x = epochs_v.columns
# epochs_v = epochs_v - np.atleast_2d(epochs_v.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T

# image motion scalars:
image_motion_x = epochs_x.quantile(0.95, axis=1)
image_motion_y = epochs_y.quantile(0.95, axis=1)

df_meta['calcium'] = df_meta['calcium_1']-df_meta['calcium_0']
df_meta['motion_x'] = np.array(image_motion_x)
df_meta['motion_y'] = np.array(image_motion_y)
df_meta['amplitude_bin'] = df_meta['amplitude'].copy()
df_meta['width_bin'] = df_meta['width'].copy()
df_meta['rate_bin'] = df_meta['rate'].copy()
df_meta['charge_bin'] = pd.cut(df_meta['charge'], [0,0.035,0.065,0.105,0.19,1], labels=False)
df_meta['charge_ps_bin'] = df_meta['charge_ps'].copy()

shell()

imp.reload(vns_analyses)
# for measure in ['pupil', 'velocity', 'walk', 'eyelid', 'calcium']:
for measure in ['pupil', 'calcium']:

    if not os.path.exists(os.path.join(fig_dir, measure)):
        os.makedirs(os.path.join(fig_dir, measure))

    for walk in [0,1,2]:
        if walk == 0:
            ind = np.array((abs(df_meta['motion_x'])<motion_cutoff) & (abs(df_meta['motion_y'])<motion_cutoff) & (abs(df_meta['velocity'])<velocity_cutoff[1]))
        elif walk == 1:
            ind = np.array((abs(df_meta['motion_x'])<motion_cutoff) & (abs(df_meta['motion_y'])<motion_cutoff) & (abs(df_meta['velocity'])>velocity_cutoff[1]))
        elif walk == 2:
            ind = np.array((abs(df_meta['motion_x'])<motion_cutoff) & (abs(df_meta['motion_y'])<motion_cutoff))
        
        if ('velocity' in measure) or ('walk' in measure):
            ind = ind & ind_clean_w

        ylim = ylims[measure]
        if (measure == 'blink'):
            ylim = (ylim[0], ylim[1]/3)

        if not measure == 'walk':
            if measure == 'velocity':
                fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[ind, ::10], timewindows=timewindows, ylabel=measure+'_1', ylim=(-ylim[1],ylim[1]))
            else:
                fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[ind, ::10], timewindows=timewindows, ylabel=measure+'_1', ylim=ylim)
            fig.savefig(os.path.join(fig_dir, measure, 'timecourses_{}_{}.pdf'.format(measure, walk)))

        for measure_ext in ['', '_c', '_c2']:
            
            if measure == 'calcium':
                p0 = True
            else:
                p0 = False

            try:
                fig = vns_analyses.plot_scalars(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim, p0=p0)
                fig.savefig(os.path.join(fig_dir, measure, 'scalars_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass

            try:            
                fig = vns_analyses.plot_scalars2(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, 'scalars2_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass
            
            try:
                fig = vns_analyses.plot_scalars3(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, 'scalars3_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass

            try:
                fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind, :], measure=measure+measure_ext, vmin=-ylim[1], vmax=ylim[1])
                fig.savefig(os.path.join(fig_dir, measure, 'matrix_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass
        
        if measure == 'calcium':

            # across good stims:
            x = np.array(epochs_c.columns, dtype=float)
            fig = plt.figure(figsize=(3,2))
            ax = fig.add_subplot(1,1,1)
            m = np.array(epochs_c.loc[ind,:].groupby('group').mean())
            s = np.array(epochs_c.loc[ind,:].groupby('group').sem())
            for i, c in enumerate(['black', 'green', 'red']):
                plt.fill_between(x, m[i]-s[i], m[i]+s[i], color=c, alpha=0.2)
                plt.plot(x, m[i], color=c)


            pvals = sp.stats.ttest_ind(np.array(epochs_c.loc[ind & (epochs_c.index.get_level_values('group')==1),:]), 
                                        np.array(epochs_c.loc[ind & (epochs_c.index.get_level_values('group')==2),:]), axis=0)[1]
            sig_indices = np.array(pvals<0.05, dtype=int)
            sig_indices[0] = 0
            sig_indices[-1] = 0
            s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
            for sig in s_bar:
                ax.hlines(-0.1, x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color='green', alpha=1, linewidth=2.5)
            
            
            plt.axvline(0, lw=0.5, color='k')
            plt.axhline(0, lw=0.5, color='k')
            plt.xlabel('Time from pulse (s)')
            sns.despine(trim=False, offset=3)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, measure, 'vns_pulse_{}.pdf'.format(walk)))

            fig = plt.figure(figsize=(3,2))
            ax = fig.add_subplot(1,1,1)
            m = np.array(epochs_x.loc[ind,:].groupby(epochs_c.loc[ind,:].index.get_level_values('group')).mean())
            s = np.array(epochs_x.loc[ind,:].groupby(epochs_c.loc[ind,:].index.get_level_values('group')).sem())
            for i, c in enumerate(['black', 'green', 'red']):
                plt.fill_between(x, m[i]-s[i], m[i]+s[i], color=c, alpha=0.2)
                plt.plot(x, m[i], color=c)
            plt.axvline(0, lw=0.5, color='k')
            plt.axhline(0, lw=0.5, color='k')
            plt.xlabel('Time from pulse (s)')
            plt.ylim(0,2)
            sns.despine(trim=False, offset=3)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, measure, 'vns_pulse_{}_x.pdf'.format(walk)))

            fig = plt.figure(figsize=(3,2))
            ax = fig.add_subplot(1,1,1)
            m = np.array(epochs_y.loc[ind,:].groupby(epochs_c.loc[ind,:].index.get_level_values('group')).mean())
            s = np.array(epochs_y.loc[ind,:].groupby(epochs_c.loc[ind,:].index.get_level_values('group')).sem())
            for i, c in enumerate(['black', 'green', 'red']):
                plt.fill_between(x, m[i]-s[i], m[i]+s[i], color=c, alpha=0.2)
                plt.plot(x, m[i], color=c)
            plt.axvline(0, lw=0.5, color='k')
            plt.axhline(0, lw=0.5, color='k')
            plt.xlabel('Time from pulse (s)')
            plt.ylim(0,2)
            sns.despine(trim=False, offset=3)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, measure, 'vns_pulse_{}_y.pdf'.format(walk)))


# 3d plot:
pupil_measure = 'pupil_c'

fig = vns_analyses.hypersurface(df_meta, z_measure=pupil_measure)
fig.savefig(os.path.join(fig_dir, '3d_surface_pupil.pdf'))

from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
from rpy2.robjects import pandas2ri

func = vns_analyses.log_logistic_3d
# popt = np.load(os.path.join('/home/jwdegee/vns_exploration', 'params.npy'))

df_meta['ne_p'] = np.NaN
df_meta['ne_c'] = np.NaN
kf = KFold(n_splits=20, shuffle=True)
fold_nr = 1
for train_index, test_index in kf.split(df_meta):
    print('fold {}'.format(fold_nr))
    # print("TRAIN:", train_index, "TEST:", test_index)
    popt, pcov = curve_fit(func, np.array(df_meta[['charge', 'rate']].iloc[train_index]), np.array(df_meta[pupil_measure].iloc[train_index]),)
    df_meta.loc[test_index, 'ne_p'] = func(np.array(df_meta.loc[test_index,['charge', 'rate']]) ,*popt)
    popt, pcov = curve_fit(func, np.array(df_meta[['charge', 'rate']].iloc[train_index]), np.array(df_meta['calcium'].iloc[train_index]),)
    df_meta.loc[test_index, 'ne_c'] = func(np.array(df_meta.loc[test_index,['charge', 'rate']]) ,*popt)
    fold_nr+=1

for X in ['ne_p', 'ne_c']:

    res = lavaan(df=df_meta.loc[~np.isnan(df_meta[pupil_measure]),:], X=X, Y=pupil_measure, M='calcium', C=False, zscore=True,)
    res = pandas2ri.ri2py(res)
    print(res)

    y = np.array([float(res.loc[res['label']=='total','est']), float(res.loc[res['label']=='ab','est']), float(res.loc[res['label']=='c','est'])])
    y1 = np.array([float(res.loc[res['label']=='total','ci.lower']), float(res.loc[res['label']=='ab','ci.lower']), float(res.loc[res['label']=='c','ci.lower'])])
    y2 = np.array([float(res.loc[res['label']=='total','ci.upper']), float(res.loc[res['label']=='ab','ci.upper']), float(res.loc[res['label']=='c','ci.upper'])])
    ci = y2-y1

    fig = plt.figure(figsize=(8,2))
    ax = fig.add_subplot(141)
    plt.bar([0,1,2], y, yerr=ci)
    plt.xticks([0,1,2], ['c','a*b',"c'"])
    ax = fig.add_subplot(142)
    sns.regplot(df_meta[X], df_meta[pupil_measure], line_kws={'color': 'red'})
    ax = fig.add_subplot(143)
    sns.regplot(df_meta[X], df_meta['calcium'], line_kws={'color': 'red'})
    ax = fig.add_subplot(144)
    sns.regplot(df_meta[pupil_measure], df_meta['calcium'], line_kws={'color': 'red'})
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'mediation_{}.pdf'.format(X)))






res = lavaan(df=df_meta.loc[~np.isnan(df_meta['pupil']),:], X='charge_ps', Y='calcium', M='pupil', C=False, zscore=True,)
print(res)

res = lavaan(df=df_meta, X='charge_ps', Y='eyelid', M='calcium', C=False, zscore=True,)
print(res)

import statsmodels.api as sm
outcome_model = sm.OLS.from_formula("pupil ~ calcium + charge_ps", df_meta).fit()
print(outcome_model.summary())

import statsmodels.api as sm
outcome_model = sm.OLS.from_formula("calcium ~ pupil + charge_ps", df_meta).fit()
print(outcome_model.summary())


import statsmodels.api as sm
outcome_model = sm.OLS.from_formula("velocity ~ calcium + charge_ps", df_meta).fit()
print(outcome_model.summary())

import statsmodels.api as sm
outcome_model = sm.OLS.from_formula("calcium ~ velocity + pupil + eyelid + charge_ps", df_meta).fit()
print(outcome_model.summary())


shell()







# regress out walking, and image motion:


import statsmodels.formula.api as sm
# result = sm.ols(formula="calcium ~ velocity + motion_x + motion_y", data=df).fit()
# df['calcium_c'] = np.array(result.resid) + df['calcium'].mean()
# for (subj, ses), d in df.groupby(['subj_idx', 'session']):
#     plot_motion(d)
for (subj, ses), d in df.groupby(['subj_idx', 'session']):

    ind = (df['subj_idx'] == subj) & (df['session'] == ses) 

    result = sm.ols(formula="calcium ~ velocity + motion_x + motion_y", data=df.loc[ind,:]).fit()
    df.loc[ind,'calcium_c'] = np.array(result.resid) + df.loc[ind,'calcium'].mean()

shell()

# parameter dependence:
for walk in [0,1,2]:
    if walk == 0:
        ind = np.array((abs(df['motion_x'])<motion_cutoff) & (abs(df['motion_y'])<motion_cutoff) & (abs(df['velocity'])<velocity_cutoff))
    elif walk == 1:
        ind = np.array((abs(df['motion_x'])<motion_cutoff) & (abs(df['motion_y'])<motion_cutoff) & (abs(df['velocity'])>velocity_cutoff))
    elif walk == 2:
        ind = np.array((abs(df['motion_x'])<motion_cutoff) & (abs(df['motion_y'])<motion_cutoff))
    



# ind = np.array((abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)<velocity_cutoff))

# epochs_xy = np.sqrt(((epochs_x**2)+(epochs_y**2)))

# x = epochs_c.columns
# calcium = utils._butter_lowpass_filter(np.array(epochs_c.loc[ind,(x>0)&(x<20)]).ravel(), highcut=1, fs=fs, order=3)
# xx = utils._butter_lowpass_filter(np.array(epochs_x.loc[ind,(x>0)&(x<20)]).ravel(), highcut=1, fs=fs, order=3)
# yy = utils._butter_lowpass_filter(np.array(epochs_y.loc[ind,(x>0)&(x<20)]).ravel(), highcut=1, fs=fs, order=3)
# resid = lin_regress_resid(calcium, [xx,yy]) + calcium.mean()

# epochs_c.loc[ind,(x>0)&(x<20)] = resid.reshape(epochs_c.loc[ind,(x>0)&(x<20)].shape)


# fig = plt.figure(figsize=(5, 2*epochs_c.loc[ind,:].shape[0],))
# for i in range(epochs_c.loc[ind,:].shape[0]):
#     calcium = epochs_c.loc[ind,:].iloc[i]
#     x = epochs_x.loc[ind,:].iloc[i]
#     y = epochs_x.loc[ind,:].iloc[i]
#     xy = np.sqrt((x**2)+(y**2))
#     ax = fig.add_subplot(epochs_c.loc[ind,:].shape[0],1,i+1)
#     ax.plot(calcium, alpha=1, color='r')
#     ax.set_ylabel('Calcium response')
#     ax = ax.twinx()
#     ax.plot(xy, alpha=0.5, color='k', zorder=1)
#     ax.set_ylabel('Image motion')
#     plt.axvline(0, color='r', lw=0.5)
# ax.set_xlabel('Time (s)')
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'vns_single_pulses.pdf'))














# shell()

# # powerspectrum
# fs = 15
# y = np.array(fluorescence.iloc[0])
# plt.figure()
# n = len(y) # length of the signal
# k = np.arange(n)
# T = n/fs
# frq = k/T # two sides frequency range
# frq = frq[range(int(n/2))] # one side frequency range
# Y = sp.fft(y)/n # fft computing and normalization
# Y = Y[range(int(n/2))]
# plt.plot(frq,abs(Y),'r') # plotting the spectrum
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
# plt.ylim(0,0.05)
# # scipy Welch
# f, Pxx_spec = sp.signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
# plt.semilogy(f, np.sqrt(Pxx_spec))
# # # numpy fft:
# # ps = np.abs(np.fft.fft(x))**2
# # freqs = np.fft.fftfreq(x.size, 1/fs)
# # idx = np.argsort(freqs)
# # plt.plot(freqs[idx], ps[idx])
# plt.xlim(left=0)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Power')
# plt.show()

# # pulse locked figure:
# means = epochs.groupby('cell').mean()
# sems = epochs.groupby('cell').sem()
# scalars = means.loc[:,(means.columns>1)&(means.columns<9)].mean(axis=1)
# sorting = np.argsort(scalars)
# means = means.iloc[sorting,:]
# sems = sems.iloc[sorting,:]
# x = np.array(means.columns, dtype=float)
# offset = 0
# fig = plt.figure(figsize=(12,3+(fluorescence.shape[0]/2)))
# for i in range(fluorescence.shape[0]):
#     plt.fill_between(x, means.iloc[i]+offset-sems.iloc[i], means.iloc[i]+offset+sems.iloc[i], alpha=0.2)
#     plt.plot(x, means.iloc[i]+offset)
#     plt.axhline(offset, lw=0.5, color='k', alpha=0.5)
#     offset += np.percentile(scalars, 75)
# plt.axvline(0, lw=0.5, color='r')
# plt.ylabel('Fluorescence')
# plt.xlabel('Time (s)')
# sns.despine(trim=False, offset=3)
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'vns_{}_{}_pulse.pdf'.format(subj, ses)))


# # good trials:
# motion_cutoff = 3
# velocity_cutoff = 0.005
# ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)<velocity_cutoff)
# good_trials = np.arange(image_pos_x.shape[0])[ind]

# # figure:
# epochs_m = epochs.loc[epochs.index.get_level_values('cell').isin(good_cells),:]
# fig = plt.figure(figsize=(6,3))

# ax = fig.add_subplot(231)
# means = epochs_m.groupby(['cell']).mean()
# for i in range(means.shape[0]):
#     plt.plot(x, means.iloc[i], color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.title('All trials\n(N={})'.format(image_pos_x.shape[0]))
# plt.ylabel('Fluorescence')
# plt.xlabel('Time (s)')

# ax = fig.add_subplot(232)
# ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)>velocity_cutoff)
# good_trials = np.arange(image_pos_x.shape[0])[ind]
# means = epochs_m.loc[epochs_m.index.get_level_values('trial').isin(good_trials),:].groupby(['cell']).mean()
# for i in range(means.shape[0]):
#     plt.plot(x, means.iloc[i], color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.title('Motion < {} μm\nWalk > {}mm\n(N={})'.format(motion_cutoff, velocity_cutoff, sum(ind)))
# plt.ylabel('Fluorescence')
# plt.xlabel('Time (s)')

# ax = fig.add_subplot(233)
# ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)<velocity_cutoff)
# good_trials = np.arange(image_pos_x.shape[0])[ind]
# means = epochs_m.loc[epochs_m.index.get_level_values('trial').isin(good_trials),:].groupby(['cell']).mean()
# for i in range(means.shape[0]):
#     plt.plot(x, means.iloc[i], color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.title('Motion < {} μm\nWalk < {}mm\n(N={})'.format(motion_cutoff, velocity_cutoff, sum(ind)))
# plt.ylabel('Fluorescence')
# plt.xlabel('Time (s)')

# ax = fig.add_subplot(234)
# # plt.fill_between(x, epochs_v.mean(axis=0)-epochs_v.sem(axis=0), epochs_v.mean(axis=0)+epochs_v.sem(axis=0), color='black', alpha=0.2)
# plt.plot(x, epochs_v.mean(axis=0), color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.ylabel('Distance (m)')
# plt.xlabel('Time (s)')

# ax = fig.add_subplot(235)
# ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)>velocity_cutoff)
# good_trials = np.arange(image_pos_x.shape[0])[ind]
# # plt.fill_between(x, epochs_v.loc[ind,:].mean(axis=0)-epochs_v.loc[ind,:].sem(axis=0), epochs_v.loc[ind,:].mean(axis=0)+epochs_v.loc[ind,:].sem(axis=0), color='black', alpha=0.2)
# plt.plot(x, epochs_v.loc[ind,:].mean(axis=0), color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.ylabel('Distance (m)')
# plt.xlabel('Time (s)')

# ax = fig.add_subplot(236)
# ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)<velocity_cutoff)
# good_trials = np.arange(image_pos_x.shape[0])[ind]
# # plt.fill_between(x, epochs_v.loc[ind,:].mean(axis=0)-epochs_v.loc[ind,:].sem(axis=0), epochs_v.loc[ind,:].mean(axis=0)+epochs_v.loc[ind,:].sem(axis=0), color='black', alpha=0.2)
# plt.plot(x, epochs_v.loc[ind,:].mean(axis=0), color='black')
# plt.axhline(0, lw=0.5, color='k', alpha=0.5)
# plt.axvline(0, lw=0.5, color='k', alpha=0.5)
# plt.ylabel('Distance (m)')
# plt.xlabel('Time (s)')

# sns.despine(trim=False, offset=3)
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'vns_{}_{}_pulse_mean.pdf'.format(subj, ses)))


# # parameter dependence:
# for walk in [0,1]:
#     if walk == 0:
#         ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)<velocity_cutoff)
#     elif walk == 1:
#         ind = (abs(image_pos_x)<10) & (abs(image_pos_y)<10) & (abs(image_motion_x)<motion_cutoff) & (abs(image_motion_y)<motion_cutoff) & (abs(distance)>velocity_cutoff)
#     good_trials = np.arange(image_pos_x.shape[0])[ind]
    
#     epochs_mm = epochs.loc[(epochs.index.get_level_values('cell').isin(good_cells))&(epochs.index.get_level_values('trial').isin(good_trials)),:]
    
#     plot_nr = 1
#     fig = plt.figure(figsize=(6,len(good_cells)*2)) 
#     for cell, epochs_m in epochs_mm.groupby(['cell']):

#         epochs_m = epochs_m.groupby(['amplitude', 'rate', 'width']).mean()
        
#         ax = fig.add_subplot(len(good_cells),3,plot_nr)
#         means = epochs_m.groupby('amplitude').mean()
#         sems = epochs_m.groupby('amplitude').sem()
#         colors = sns.dark_palette("red", means.shape[0])
#         for i in range(means.shape[0]):
#             plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color=colors[i], alpha=0.2)
#             plt.plot(x, means.iloc[i], color=colors[i])
#         plt.axhline(0, lw=0.5, color='k', alpha=0.5)
#         plt.axvline(0, lw=0.5, color='k', alpha=0.5)
#         plt.ylabel('Fluorescence')
#         plt.xlabel('Time (s)')
#         if plot_nr == 1:
#             plt.title('Amplitude')
#         plot_nr += 1

#         ax = fig.add_subplot(len(good_cells),3,plot_nr)
#         means = epochs_m.groupby('width').mean()
#         sems = epochs_m.groupby('width').sem()
#         colors = sns.dark_palette("red", means.shape[0])
#         for i in range(means.shape[0]):
#             plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color=colors[i], alpha=0.2)
#             plt.plot(x, means.iloc[i], color=colors[i])
#         plt.axhline(0, lw=0.5, color='k', alpha=0.5)
#         plt.axvline(0, lw=0.5, color='k', alpha=0.5)
#         plt.ylabel('Fluorescence')
#         plt.xlabel('Time (s)')
#         if plot_nr == 2:
#             plt.title('Width')
#         plot_nr += 1
        
#         ax = fig.add_subplot(len(good_cells),3,plot_nr)
#         means = epochs_m.groupby('rate').mean()
#         sems = epochs_m.groupby('rate').sem()
#         colors = sns.dark_palette("red", means.shape[0])
#         for i in range(means.shape[0]):
#             plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color=colors[i], alpha=0.2)
#             plt.plot(x, means.iloc[i], color=colors[i])
#         plt.axhline(0, lw=0.5, color='k', alpha=0.5)
#         plt.axvline(0, lw=0.5, color='k', alpha=0.5)
#         plt.ylabel('Fluorescence')
#         plt.xlabel('Time (s)')
#         if plot_nr == 3:
#             plt.title('Rate')
#         plot_nr += 1
#     sns.despine(trim=False, offset=3)
#     plt.tight_layout()
#     fig.savefig(os.path.join(fig_dir, 'vns_{}_{}_pulse_parametric_{}.pdf'.format(subj, ses, walk)))

# image_pos_x = np.concatenate(image_pos_x_)
# image_pos_y = np.concatenate(image_pos_y_)
# image_motion_x = np.concatenate(image_motion_x_)
# image_motion_y = np.concatenate(image_motion_y_)
# distance = np.concatenate(distance_)

# for cutoff_motion in [10, 5, 4, 3, 2]:

#     cutoff_pos = 10
#     # cutoff_motion = 10
#     cutoff_distance = 0.005

#     fig = plt.figure(figsize=(6,4))
#     ax = fig.add_subplot(231)
#     ind = (abs(image_pos_x)<cutoff_pos)&(abs(image_pos_y)<cutoff_pos)
#     ax.plot(image_pos_x, image_pos_y, 'o', ms=5, markeredgecolor='w', markeredgewidth=0.5)
#     plt.axhline(cutoff_pos, lw=0.5, color='r')
#     plt.axvline(cutoff_pos, lw=0.5, color='r')
#     plt.axhline(-cutoff_pos, lw=0.5, color='r')
#     plt.axvline(-cutoff_pos, lw=0.5, color='r')
#     ax.set_xlabel('Image position x')
#     ax.set_ylabel('Image position y')
#     ax.set_title('{} / {} trials'.format(sum(ind), len(ind)))
    
#     ax = fig.add_subplot(232)
#     ind = (abs(image_pos_x)<cutoff_pos)&(abs(image_pos_y)<cutoff_pos)&(image_motion_x<cutoff_motion)&(image_motion_y<cutoff_motion)
#     ax.plot(image_motion_x, image_motion_y, 'o', ms=5, markeredgecolor='w', markeredgewidth=0.5)
#     plt.axhline(cutoff_motion, lw=0.5, color='r')
#     plt.axvline(cutoff_motion, lw=0.5, color='r')
#     ax.set_xlabel('Image motion x') 
#     ax.set_ylabel('Image motion y')
#     ax.set_title('{} / {} trials'.format(sum(ind), len(ind)))

#     ax = fig.add_subplot(233)
#     ind = (abs(image_pos_x)<cutoff_pos)&(abs(image_pos_y)<cutoff_pos)&(image_motion_x<cutoff_motion)&(image_motion_y<cutoff_motion)&(abs(distance)<cutoff_distance)
#     ax.plot((image_motion_x+image_motion_y)/2, abs(distance), 'o', ms=5, markeredgecolor='w', markeredgewidth=0.5)
#     plt.axhline(cutoff_distance, lw=0.5, color='r')
#     plt.axvline(cutoff_motion, lw=0.5, color='r')
#     ax.set_xlabel('Image motion xy') 
#     ax.set_ylabel('Walking distance')
#     ax.set_title('{} / {} trials'.format(sum(ind), len(ind)))

#     ax = fig.add_subplot(234)
#     sns.kdeplot(image_pos_x, image_pos_y, shade=True, n_levels=15, ax=ax)
#     plt.axhline(cutoff_pos, lw=0.5, color='r')
#     plt.axvline(cutoff_pos, lw=0.5, color='r')
#     plt.axhline(-cutoff_pos, lw=0.5, color='r')
#     plt.axvline(-cutoff_pos, lw=0.5, color='r')
#     ax.set_xlabel('Image position x')
#     ax.set_ylabel('Image position y')
#     ax.set_title('{} / {} trials'.format(sum(ind), len(ind)))
#     ax = fig.add_subplot(235)
#     sns.kdeplot(image_motion_x, image_motion_y, shade=True, n_levels=15, ax=ax)
#     plt.axhline(cutoff_motion, lw=0.5, color='r')
#     plt.axvline(cutoff_motion, lw=0.5, color='r')
#     ax.set_xlabel('Image motion x')
#     ax.set_ylabel('Image motion y')

#     ax = fig.add_subplot(236)
#     sns.kdeplot((image_motion_x+image_motion_y)/2, abs(distance), shade=True, n_levels=15, ax=ax)
#     plt.axhline(cutoff_distance, lw=0.5, color='r')
#     plt.axvline(cutoff_motion, lw=0.5, color='r')
#     ax.set_xlabel('Image motion xy')
#     ax.set_ylabel('Walking distance')

#     plt.tight_layout()
#     fig.savefig(os.path.join(fig_dir, 'preprocess_motion_params_{}.pdf'.format(cutoff_motion)),)