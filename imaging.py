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

def mediation_analysis(df, X, Y, M, bootstrap=True, n_boot=1000, n_jobs=12):

    df = df.loc[~np.isnan(df[Y]),:].reset_index(drop=True)
    if bootstrap:

        res = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(lavaan)(df, X, Y, M, False, True, True) for _ in range(n_boot))
        res = pd.concat(res)
        p_values = [(res.loc[res['label']=='total','est']<0).mean(),
                    (res.loc[res['label']=='ab','est']<0).mean(),
                    (res.loc[res['label']=='c','est']<0).mean(),]

        # res_sem = ((res.loc[:,['label', 'est']].groupby('label').quantile(.84)-res.loc[:,['label', 'est']].groupby('label').quantile(.16))/2).reset_index()
        res_sem = ((res.loc[:,['label', 'est']].groupby('label').quantile(.975)-res.loc[:,['label', 'est']].groupby('label').quantile(.025))/2).reset_index()
        res_mean = res.groupby('label').mean().reset_index()
        print(res)
    else:
        res_mean = lavaan(df=df, X=X, Y=Y, M=M, C=False, zscore=True,).groupby('label').mean()
        print(res)

    y = np.array([float(res_mean.loc[res_mean['label']=='total','est']), float(res_mean.loc[res_mean['label']=='ab','est']), float(res_mean.loc[res_mean['label']=='c','est'])])
    if bootstrap:
        ci = np.array([float(res_sem.loc[res_sem['label']=='total','est']), float(res_sem.loc[res_sem['label']=='ab','est']), float(res_sem.loc[res_sem['label']=='c','est'])])
    else:
        y1 = np.array([float(res.loc[res['label']=='total','ci.lower']), float(res.loc[res['label']=='ab','ci.lower']), float(res.loc[res['label']=='c','ci.lower'])])
        y2 = np.array([float(res.loc[res['label']=='total','ci.upper']), float(res.loc[res['label']=='ab','ci.upper']), float(res.loc[res['label']=='c','ci.upper'])])
        ci = (y2-y1)/2

    # fig = plt.figure(figsize=(2,2))
    # ax = fig.add_subplot(111)
    # plt.bar([0,1,2], y, yerr=ci)
    # for x,p in zip([0,1,2],p_values):
    #     plt.text(x=x, y=max(y), s=round(p,3), size=6)
    # plt.xticks([0,1,2], ['c','a*b',"c'"])
    # sns.despine(trim=False, offset=3)
    # plt.tight_layout()
    
    to_plot = pd.DataFrame({'total':np.array(res.loc[res['label']=='total','est']),
                            'ab':np.array(res.loc[res['label']=='ab','est']),
                            'c':np.array(res.loc[res['label']=='c','est'])})
    to_plot = to_plot.stack().reset_index()

    fig = plt.figure(figsize=(3,2))
    sns.violinplot(x='level_1', y=0, data=to_plot)
    for x,p in zip([0,1,2],p_values):
        plt.text(x=x, y=max(y), s=round(p,3), size=6)
    plt.ylabel('Parameter estimate (a.u.)')
    plt.xlabel('')
    sns.despine(trim=False, offset=3)
    plt.tight_layout()

    # plt.bar([0,1,2], y, yerr=ci)

    # plt.xticks([0,1,2], ['c','a*b',"c'"])
    # sns.despine(trim=False, offset=3)
    # plt.tight_layout()

    return fig

def lavaan(df, X, Y, M, C=False, zscore=True, resample=False):
    
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter

    if resample:
        df = df.sample(frac=1, replace=True)

    # convert datafame:
    data = pd.DataFrame({'X':df[X], 'Y':df[Y], 'M':df[M]})
    with localconverter(default_converter + pandas2ri.converter) as cv:
        data = pandas2ri.py2rpy(data)
    
    # load lavaan and data:
    robjects.r('library(lavaan)')
    robjects.globalenv["data"] = data

    # zscore:
    if zscore:
        robjects.r("data['X'] = (data[['X']] - mean(data[['X']])) / sd(data[['X']])")
        robjects.r("data['Y'] = (data[['Y']] - mean(data[['Y']])) / sd(data[['Y']])")
        robjects.r("data['M'] = (data[['M']] - mean(data[['M']])) / sd(data[['M']])")
    
    # fit:
    if C == True:
        robjects.r("model = 'Y ~ c*X + C\nM ~ a*X + C\nY ~ b*M\nab := a*b\ntotal := c + (a*b)'")
        robjects.r('fit = sem(model, data=data)')
    else:
        robjects.r("model = 'Y ~ c*X\nM ~ a*X\nY ~ b*M\nab := a*b\ntotal := c + (a*b)'")
        robjects.r('fit = sem(model, data=data)')

    robjects.r('summary(fit, fit.measures=TRUE)')

    res = pandas2ri.rpy2py(robjects.r("parameterEstimates(fit)"))

    return res

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

def make_scalars(df):

    timewindows = {

                'velocity_-3' : [(-30.1,-30), (-40.1,-40)],
                'velocity_-2' : [(-20.1,-20), (-30.1,-30)],
                'velocity_-1' : [(-10.1,-10), (-20.1,-20)],
                'velocity_0' : [(-0.1,0), (-10.1,-10)],
                'velocity_1' : [(9.9,10), (-0.1,0)],

                'pupil_-1' : [(-50, -30), (None, None)],
                'pupil_0' : [(-20, 0), (None, None)],
                'pupil_1' : [(10, 30), (None, None)],

                'slope_-1' : [(-9, -6), (None, None)],      #
                'slope_0' : [(-6, 3), (None, None)],          #
                'slope_1' : [(0, 3), (None, None)],

                'blink_-3' : [(-27.5, -22.5), (None, None)],
                'blink_-2' : [(-20, -15), (None, None)],
                'blink_-1' : [(-12.5, -7.5), (None, None)],
                'blink_0' : [(-5, 0), (None, None)],
                'blink_1' : [(2.5, 7.5), (None, None)],

                'eyelid_-3' : [(-27.5, -22.5), (None, None)],
                'eyelid_-2' : [(-20, -15), (None, None)],
                'eyelid_-1' : [(-12.5, -7.5), (None, None)],
                'eyelid_0' : [(-5, 0), (None, None)],
                'eyelid_1' : [(0, 25), (None, None)],
                
                'calcium_-3' : [(-27.5, -22.5), (None, None)],
                'calcium_-2' : [(-20, -15), (None, None)],
                'calcium_-1' : [(-12.5, -7.5), (None, None)],
                'calcium_0' : [(-5, 0), (None, None)],
                'calcium_1' : [(2.5, 7.5), (None, None)],

                'imagex_0' : [(-5, 0), (None, None)],
                'imagex_1' : [(2.5, 7.5), (None, None)],

                'imagey_0' : [(-5, 0), (None, None)],
                'imagey_1' : [(2.5, 7.5), (None, None)],

                'corrXY_0' : [(-5, 0), (None, None)],
                'corrXY_1' : [(0, 5), (None, None)],

                }

    epochs = {
                'velocity' : epochs_v,
                'pupil' : epochs_p,
                'slope' : epochs_s,
                'eyelid' : epochs_l,
                'blink' : epochs_b,
                # 'eyemovement' : epochs_xy,
                'calcium' : epochs_c,
                'imagex' : epochs_x,
                'imagey' : epochs_y,
                'corrXY' : epochs_corrXY,
                }

    for key in timewindows.keys():
        x = epochs[key.split('_')[0]].columns
        window1, window2 = timewindows[key]
        if 'slope' in key:
            # resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1)
            resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].max(axis=1)
            # resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].quantile(0.95, axis=1)
        else:
            resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1)
        if window2[0] == None: 
            df[key] = np.array(resp)
        else:
            baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1).values
            df[key] = np.array(resp-baseline)
        # if key == 'blink':
        #     df['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)

    # add walk probability scalars
    df['walk_-3'] = ((df['velocity_-3'] < velocity_cutoff[0])|(df['velocity_-3'] > velocity_cutoff[1])).astype(int)
    df['walk_-2'] = ((df['velocity_-2'] < velocity_cutoff[0])|(df['velocity_-2'] > velocity_cutoff[1])).astype(int)
    df['walk_-1'] = ((df['velocity_-1'] < velocity_cutoff[0])|(df['velocity_-1'] > velocity_cutoff[1])).astype(int)
    df['walk_0'] = ((df['velocity_0'] < velocity_cutoff[0])|(df['velocity_0'] > velocity_cutoff[1])).astype(int)
    df['walk_1'] = ((df['velocity_1'] < velocity_cutoff[0])|(df['velocity_1'] > velocity_cutoff[1])).astype(int)

    # add image motion scalars:
    df['imagex'] = abs(df['imagex_1']-df['imagex_0'])
    df['imagey'] = abs(df['imagey_1']-df['imagey_0'])
    df['corrXY'] = df['corrXY_1']-df['corrXY_0']

    return df, timewindows

def plot_velocity_histogram(df, bins=100):

    fig = plt.figure(figsize=(6,2))

    ax = fig.add_subplot(131)
    ax.hist(df.loc[:, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)

    ax = fig.add_subplot(132)
    ax.hist(df.loc[ind_s, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_s[ind_clean_w])/np.sum(ind_clean_w)*100,1)))

    ax = fig.add_subplot(133)
    ax.hist(df.loc[ind_w, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_w[ind_clean_w])/np.sum(ind_clean_w)*100,1)))

    plt.tight_layout()
    sns.despine(trim=False, offset=3)

    return fig

def half_max_analyses(df, fig_dir):

    for walk in [2,0]:
        if walk == 0:
            ind = np.array(abs(df['velocity'])<velocity_cutoff[1])
        elif walk == 2:
            ind = np.ones(df.shape[0], dtype=bool)

        imp.reload(vns_analyses)
        nboot = 5000
        n_jobs = 48
        res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df.loc[ind,:], 'pupil_c', resample=True) for _ in range(nboot)))
        res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df.loc[ind,:], 'calcium_c', resample=True) for _ in range(nboot)))

        for m, i in zip(['charge', 'rate'], [0,3]):
            
            m1 = round(res1[:,i].mean(),3)
            m2 = round(res2[:,i].mean(),3)
            s1 = round(res1[:,i].std(),3)
            s2 = round(res2[:,i].std(),3)

            p_value = round((res1[:,i] < res2[:,i]).mean(),3)
            p_value = min((p_value, 1-p_value))

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            sns.distplot(np.array(res1)[:,i], hist=False, color='r') 
            sns.distplot(np.array(res2)[:,i], hist=False, color='g')
            plt.title('m1 = {}, s1 = {}\nm2 = {}, s2 = {}\np = {}'.format(m1, s1, m2, s2, p_value))
            plt.xlabel(m)
            plt.ylabel('KDE')
            sns.despine(trim=False, offset=3)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, 'halfmax_{}_{}.pdf'.format(m, walk)))

def add_nerve_engagement(df_meta, pupil_measure='pupil_c', calcium_measure='calcium_c'):
    from sklearn.model_selection import KFold
    from scipy.optimize import curve_fit
    func = vns_analyses.log_logistic_3d
    df_meta['ne_p'] = np.NaN
    df_meta['ne_c'] = np.NaN
    kf = KFold(n_splits=20, shuffle=True)
    fold_nr = 1
    for train_index, test_index in kf.split(df_meta):
        print('fold {}'.format(fold_nr))
        # print("TRAIN:", train_index, "TEST:", test_index)
        popt, pcov = curve_fit(func, np.array(df_meta[['charge', 'rate']].iloc[train_index]), np.array(df_meta[pupil_measure].iloc[train_index]),
                                method='trf', bounds=([0, 0, 0, 0, 0,], [np.inf, np.inf, np.inf, np.inf, np.inf,]), max_nfev=10000)
        df_meta.loc[test_index, 'ne_p'] = func(np.array(df_meta.loc[test_index,['charge', 'rate']]) ,*popt)
        popt, pcov = curve_fit(func, np.array(df_meta[['charge', 'rate']].iloc[train_index]), np.array(df_meta[calcium_measure].iloc[train_index]),
                                method='trf', bounds=([0, 0, 0, 0, 0,], [np.inf, np.inf, np.inf, np.inf, np.inf,]), max_nfev=10000)
        df_meta.loc[test_index, 'ne_c'] = func(np.array(df_meta.loc[test_index,['charge', 'rate']]) ,*popt)
        fold_nr+=1
    
    return df_meta

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
imaging_dir1 = '/media/internal1/vns/'
imaging_dir2 = '/media/internal2/vns/'
base_dir = '/home/jwdegee/'
project_name = 'vns_exploration'
project_dir = os.path.join(base_dir, project_name)
fig_dir = os.path.join(base_dir, project_name, 'figures', 'imaging')
data_dir = os.path.join(project_dir, 'data', 'imaging')

subjects = {
    'C7A6': (['1', '7', '8', '9'], imaging_dir1),  
    'C1772': (['6', '7', '8'], imaging_dir2),
    'C1773': (['5', '6', '7', '8', '10'], imaging_dir2),
    # 'C1773': ['10'],
}

if preprocess:
    
    # imaging:
    tasks = []
    for subj in subjects:
        print(subj)
        sessions = subjects[subj][0]
        for ses in sessions:
            tasks.append((raw_dir, subjects[subj][1], fig_dir, subj, ses))
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.analyse_imaging_session)(*task) for task in tasks)
    epochs_c = pd.concat([res[i][0] for i in range(len(res))], axis=0)
    epochs_x = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_y = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_corrXY = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][6] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][7] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_c.to_hdf(os.path.join(data_dir, 'epochs_c.hdf'), key='calcium')
    epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x_motion')
    epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y_motion')
    epochs_corrXY.to_hdf(os.path.join(data_dir, 'epochs_corrXY.hdf'), key='corrXY')
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')

epochs_c = pd.read_hdf(os.path.join(data_dir, 'epochs_c.hdf'), key='calcium') * 100
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x_motion')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y_motion')
epochs_corrXY = pd.read_hdf(os.path.join(data_dir, 'epochs_corrXY.hdf'), key='corrXY')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil') * 100
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid') * 100
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')

# preprocess slope:
epochs_s = epochs_p.diff(axis=1) * 50
epochs_s = epochs_s.rolling(window=10, min_periods=1, axis=1).median()
for i in range(epochs_s.shape[0]):
    ind = ~np.isnan(epochs_s.iloc[i,:])
    epochs_s.loc[i,ind] = utils._butter_lowpass_filter(epochs_s.loc[i,ind], highcut=0.5, fs=50, order=3)

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
df_meta['amplitude_bin'] = df_meta['amplitude'].copy()

df_meta.to_csv(os.path.join(data_dir, 'df_meta.csv'))

# make scalars:
df_meta, timewindows = make_scalars(df_meta)

# ylims:
ylims = {
            'velocity' : (-0.1, 0.5),
            'walk' : (0,1),
            'pupil' : (-5, 40),
            'slope' : (-5, 20),
            'eyelid' : (-1, 10),
            'blink' : (0, 0.3),
            'calcium' : (-10, 50),
            'imagex' : (0, 5),
            'imagey' : (0, 5),
            'corrXY' : (-0.02, 0.07),
            }

# motion figure
nr_trials = np.sum((df_meta['imagex']<=motion_cutoff) & (df_meta['imagey']<=motion_cutoff))
g = sns.jointplot("imagex", "imagey", data=df_meta, marginal_kws=dict(bins=50), color="m", height=2.5)
g.fig.axes[0].axvline(motion_cutoff, ls='--', color='r')
g.fig.axes[0].axhline(motion_cutoff, ls='--', color='r')
plt.title('{} / {} pulses'.format(nr_trials, df_meta.shape[0]))
g.savefig(os.path.join(fig_dir, 'image_motion.pdf'))

# throw away too much motion trials and other weird things:
remove = np.array(
        (df_meta['subj_idx'] == 'C1773') & (df_meta['session']=='8') & (df_meta['trial']==17) | # weird trial with crazy spike
        (df_meta['width']==0.8) | # should not be...
        (np.isnan(np.array(epochs_c.loc[:,(epochs_c.columns>0)&(epochs_c.columns<1)].mean(axis=1)))) | # missing calcium data...
        np.isnan(df_meta['pupil_1']) | # missing pupil data...
        (df_meta['imagex']>motion_cutoff) | 
        (df_meta['imagey']>motion_cutoff))
df_meta = df_meta.loc[~remove,:].reset_index(drop=True)
epochs_x = epochs_x.loc[~remove,:].reset_index(drop=True)
epochs_y = epochs_y.loc[~remove,:].reset_index(drop=True)
epochs_corrXY = epochs_corrXY.loc[~remove,:].reset_index(drop=True)
epochs_v = epochs_v.loc[~remove,:].reset_index(drop=True)
epochs_p = epochs_p.loc[~remove,:].reset_index(drop=True)
epochs_s = epochs_s.loc[~remove,:].reset_index(drop=True)
epochs_l = epochs_l.loc[~remove,:].reset_index(drop=True)
epochs_c = epochs_c.loc[~remove,:]

# indices:
ind_clean_w = ~(np.isnan(df_meta['velocity_1'])|np.isnan(df_meta['velocity_0']))
ind_s = ((df_meta['velocity_1'] >= velocity_cutoff[0]) & (df_meta['velocity_1'] <= velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])
ind_w = ((df_meta['velocity_1'] < velocity_cutoff[0]) | (df_meta['velocity_1'] > velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])

# add variables
charge_edges = [0, 0.045, 0.085, 0.16, 0.32, 1]
# df_meta['calcium'] = df_meta['calcium_1']-df_meta['calcium_0']
df_meta['amplitude_bin'] = df_meta['amplitude'].copy()
df_meta['width_bin'] = df_meta['width'].copy()
df_meta['rate_bin'] = df_meta['rate'].copy()
df_meta['charge'] = np.round(df_meta['charge'], 4)
df_meta['charge_bin'] = pd.cut(df_meta['charge'], charge_edges, labels=False)
df_meta['charge_ps_bin'] = df_meta['charge_ps'].copy()
epochs_c['group'] = 0
epochs_c.loc[np.array(df_meta['charge_bin']==0), 'group'] = 1
epochs_c.loc[np.array((df_meta['charge_bin']>0)&(df_meta['charge_bin']<5)), 'group'] = 2
epochs_c.loc[np.array(df_meta['charge_bin']==4), 'group'] = 3
epochs_c = epochs_c.set_index('group', append=True)

# correct scalars:
df_meta, figs = vns_analyses.correct_scalars(df_meta, group=np.ones(df_meta.shape[0], dtype=bool), velocity_cutoff=velocity_cutoff, ind_clean_w=ind_clean_w)
figs[0].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean1.pdf'))
figs[1].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean2.pdf'))
figs[2].savefig(os.path.join(fig_dir, 'slope_reversion_to_mean1.pdf'))
figs[3].savefig(os.path.join(fig_dir, 'slope_reversion_to_mean2.pdf'))
figs[4].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean1.pdf'))
figs[5].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean2.pdf'))
figs[6].savefig(os.path.join(fig_dir, 'calcium_reversion_to_mean1.pdf'))
figs[7].savefig(os.path.join(fig_dir, 'calcium_reversion_to_mean2.pdf'))
figs[8].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean1.pdf'))
figs[9].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean2.pdf'))
figs[10].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean3.pdf'))
figs[11].savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean4.pdf'))
figs[12].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean1.pdf'))
figs[13].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean2.pdf'))
figs[14].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean3.pdf'))
figs[15].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean4.pdf'))

# nerve engagement:
pupil_measure = 'pupil_c'
calcium_measure = 'calcium_c'
df_meta = add_nerve_engagement(df_meta, pupil_measure=pupil_measure, calcium_measure=calcium_measure)

# regress out VNS parameter dependence:
popt = vns_analyses.fit_log_logistic(df_meta, 'calcium_c', resample=False)
df_meta['calcium_predicted'] = vns_analyses.log_logistic_3d(np.array(df_meta[['charge', 'rate']]), *popt)
df_meta['calcium_residuals'] = (df_meta['calcium_c']-df_meta['calcium_predicted']) + df_meta['calcium_c'].mean()
fig = vns_analyses.plot_scalars2(df_meta, measure='calcium_residuals', ylabel='calcium', ylim=ylims['calcium'], p0=False)
fig.savefig(os.path.join(fig_dir, 'calcium', 'scalars2_{}_{}.pdf'.format('calcium_residuals', 2)))
popt = vns_analyses.fit_log_logistic(df_meta, 'pupil_c', resample=False)
df_meta['pupil_predicted'] = vns_analyses.log_logistic_3d(np.array(df_meta[['charge', 'rate']]), *popt)
df_meta['pupil_residuals'] = (df_meta['pupil_c']-df_meta['pupil_predicted']) + df_meta['pupil_c'].mean()
fig = vns_analyses.plot_scalars2(df_meta, measure='pupil_residuals', ylabel='pupil', ylim=ylims['pupil'], p0=False)
fig.savefig(os.path.join(fig_dir, 'pupil', 'scalars2_{}_{}.pdf'.format('pupil_residuals', 2)))

# save source data:
for walk in ['all', 'still']:
    if walk == 'still':
        ind = np.array(abs(df_meta['velocity'])<velocity_cutoff[1])
    elif walk == 'all':
        ind = np.ones(df_meta.shape[0], dtype=bool)
    df = df_meta.loc[ind,['subj_idx', 'session', 'amplitude', 'width', 'rate', 'charge', 'charge_bin', 'calcium_c', 'pupil_c', 'ne_p']].reset_index(drop=True)
    df = df.rename({'calcium_c':'calcium', 'pupil_c':'pupil', 'ne_p':'nerve_engagement'}, axis=1)
    df.to_csv(os.path.join(data_dir, 'imaging_source_data_{}.csv'.format(walk)))

# motion figure
fig = plt.figure(figsize=(4,10))
plt_nr = 1
for X in ['imagex', 'imagey', 'corrXY_0', 'corrXY_1', 'corrXY']:
    ax = fig.add_subplot(5,2,plt_nr)
    plt.hist(df_meta[X], alpha=0.5, bins=12, histtype='stepfilled')
    if 'image' in X:
        plt.axvline(motion_cutoff, ls='--', color='r')
    else:
        plt.axvline(df_meta[X].mean(), ls='-', color='k')
    plt_nr += 1
    ax = fig.add_subplot(5,2,plt_nr)
    sns.regplot(df_meta[X], df_meta['calcium_c'], line_kws={'color': 'red'})
    plt_nr += 1
sns.despine(trim=False, offset=3)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'image_motion2.pdf'))

# velocity historgram:
fig = plot_velocity_histogram(df_meta)
fig.savefig(os.path.join(fig_dir, 'velocity_hist.pdf'))

# subtract baselines:
x = epochs_c.columns
epochs_c = epochs_c - np.atleast_2d(epochs_c.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_p.columns
epochs_p = epochs_p - np.atleast_2d(epochs_p.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_l.columns
epochs_l = epochs_l - np.atleast_2d(epochs_l.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_v.columns
epochs_v = epochs_v - np.atleast_2d(epochs_v.loc[:,(x>=-0.1)&(x<=-0.0)].mean(axis=1)).T
x = epochs_x.columns
epochs_x = abs(epochs_x - np.atleast_2d(epochs_x.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T)
x = epochs_y.columns
epochs_y = abs(epochs_y - np.atleast_2d(epochs_y.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T)
# x = epochs_corrXY.columns
# epochs_corrXY = epochs_corrXY - np.atleast_2d(epochs_corrXY.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T

# epochs:
epochs = {
            'velocity' : epochs_v,
            'pupil' : epochs_p,
            'slope' : epochs_s,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            'calcium' : epochs_c,
            'imagex' : epochs_x,
            'imagey' : epochs_y,
            'corrXY' : epochs_corrXY,
            }

# print r-squares:
for measure in ['calcium_c', 'pupil_c', 'eyelid_c',]:
    for walk in [2,0]:
        if walk == 0:
            ind = np.array(abs(df_meta['velocity'])<velocity_cutoff[1])
        elif walk == 2:
            ind = np.ones(df_meta.shape[0], dtype=bool)
        
        print()
        print(measure)
        print(walk)
        popt = vns_analyses.fit_log_logistic(df_meta.loc[ind, :], measure, resample=False)

# half max analyses:
half_max_analyses(df_meta, fig_dir)

imp.reload(vns_analyses)
for measure in ['calcium', 'corrXY', 'pupil', 'slope', 'velocity', 'walk', 'eyelid',]:
    if not os.path.exists(os.path.join(fig_dir, measure)):
        os.makedirs(os.path.join(fig_dir, measure))
    for walk in [2,0]:
        if walk == 0:
            ind = np.array(abs(df_meta['velocity'])<velocity_cutoff[1])
        elif walk == 1:
            ind = np.array(abs(df_meta['velocity'])>velocity_cutoff[1])
        elif walk == 2:
            ind = np.ones(df_meta.shape[0], dtype=bool)
        if ('velocity' in measure) or ('walk' in measure):
            ind = ind & ind_clean_w
        ylim = ylims[measure]
        if (measure == 'blink'):
            ylim = (ylim[0], ylim[1]/3)
        if not measure == 'walk':
            fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[:,(x>=-20)&(x<=40)].loc[ind, ::10], timewindows=timewindows, ylabel=measure+'_1', ylim=ylim)
            fig.savefig(os.path.join(fig_dir, measure, 'timecourses_{}_{}.pdf'.format(measure, walk)))
        if measure == 'corrXY':
            measure_exts = ['', '_0', '_1']
        else:
            measure_exts = ['', '_c', '_c2']
        for measure_ext in measure_exts:
            if (measure == 'calcium'):
                p0 = True
            else:
                p0 = False
            fig = vns_analyses.plot_scalars(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim, p0=p0)
            fig.savefig(os.path.join(fig_dir, measure, 'scalars_{}_{}.pdf'.format(measure+measure_ext, walk)))
            fig = vns_analyses.plot_scalars2(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim, p0=p0)
            fig.savefig(os.path.join(fig_dir, measure, 'scalars2_{}_{}.pdf'.format(measure+measure_ext, walk)))
            fig = vns_analyses.plot_scalars3(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
            fig.savefig(os.path.join(fig_dir, measure, 'scalars3_{}_{}.pdf'.format(measure+measure_ext, walk)))

            fig1, fig2, fig3 = vns_analyses.plot_swarms(df_meta.loc[ind &  ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext)
            fig1.savefig(os.path.join(fig_dir, measure, 'swarms_amp_{}_{}.pdf'.format(measure+measure_ext, walk)))
            fig2.savefig(os.path.join(fig_dir, measure, 'swarms_width_{}_{}.pdf'.format(measure+measure_ext, walk)))
            fig3.savefig(os.path.join(fig_dir, measure, 'swarms_zone_{}_{}.pdf'.format(measure+measure_ext, walk)))

            try:
                fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, vmin=-ylim[1], vmax=ylim[1])
                fig.savefig(os.path.join(fig_dir, measure, 'matrix_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass
            try:
                fig = vns_analyses.hypersurface2(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], z_measure=measure+measure_ext, ylim=(0,ylim[1]))
                fig.savefig(os.path.join(fig_dir, measure, '3dplot_{}_{}.pdf'.format(measure+measure_ext, walk)))
            except:
                pass
        
        if measure == 'calcium':

            # across good stims:
            x = np.array(epochs_c.columns, dtype=float)
            fig = plt.figure(figsize=(2,2))
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
            plt.ylim(-0.1, 0.6)
            plt.xlabel('Time from pulse (s)')
            sns.despine(trim=False, offset=3)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, measure, 'vns_pulse_{}.pdf'.format(walk)))

            fig = plt.figure(figsize=(2,2))
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

            fig = plt.figure(figsize=(2,2))
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

# mediation analysis:
pupil_measure = 'pupil_c'
calcium_measure = 'calcium_c'
for walk in [2,0]:
# for walk in [2]:
    if walk == 0:
        ind = np.array(abs(df_meta['velocity'])<velocity_cutoff[1])
    elif walk == 1:
        ind = np.array(abs(df_meta['velocity'])>velocity_cutoff[1])
    elif walk == 2:
        ind = np.ones(df_meta.shape[0], dtype=bool)

    # mediation analysis:
    fig = mediation_analysis(df_meta.loc[ind,:].reset_index(drop=True), X='ne_p', Y=pupil_measure, M=calcium_measure, 
                                bootstrap=True, n_boot=10000, n_jobs=48)
    fig.savefig(os.path.join(fig_dir, 'correlations', 'mediation_{}_{}.pdf'.format('ne_p', walk)))

    # # # reverse:
    # # fig = mediation_analysis(df_meta.loc[ind,:].reset_index(drop=True), X='ne_p', Y=calcium_measure, M=pupil_measure, 
    # #                             bootstrap=True, n_boot=10000, n_jobs=48)
    # # fig.savefig(os.path.join(fig_dir, 'correlations', 'mediation_reverse_{}_{}.pdf'.format('ne_p', walk)))

    # straight correlations:
    for X,Y in zip(['ne_p', 'ne_p', 'calcium_c'], ['pupil_c', 'calcium_c', 'pupil_c'],):
        fig = vns_analyses.plot_correlation(df_meta.loc[ind,:].reset_index(drop=True), X=X, Y=Y)
        fig.savefig(os.path.join(fig_dir, 'correlations', 'correlations_{}_{}_{}.pdf'.format(X, Y, walk)))
    
    # partial correlations:
    for X,Y,M in zip(['ne_p', 'ne_p', 'calcium_c'], ['pupil_c', 'calcium_c', 'pupil_c'], ['calcium_c', 'pupil_c', 'ne_p']):
        fig = vns_analyses.plot_partial_correlations(df_meta.loc[ind,:].reset_index(drop=True), X=X, Y=Y, M=M)
        fig.savefig(os.path.join(fig_dir, 'correlations', 'partial_correlations_{}_{}_{}.pdf'.format(X, Y, walk)))

# state dependence analysis:
df_meta['velocity_0_abs'] = abs(df_meta['velocity_0'])
df_meta['bins_velocity'] = pd.cut(df_meta['velocity_0_abs'], 8, labels=False)
# for walk in [2]:
for walk in [2,0]:
    for X, bin_measure in zip(['pupil_0', 'calcium_0', 'velocity_0_abs'], ['bins_pupil', None, None]):
        for Y in ['pupil_0', 'calcium_0', 'calcium_residuals', 'pupil_residuals']:
            if walk == 0:
                ind = np.array(abs(df_meta['velocity'])<velocity_cutoff[1])
            elif walk == 2:
                ind = np.ones(df_meta.shape[0], dtype=bool)
            if 'velocity' in X:
                ind = ind & (df_meta['velocity_0']<0.03)
            if not X == Y:
                fig = vns_analyses.plot_correlation(df_meta.loc[ind,:], X=X, Y=Y, bin_measure=bin_measure)
                fig.savefig(os.path.join(fig_dir, 'correlations', 'state_dependence_{}_{}_{}.pdf'.format(X,Y,walk)))
    
    fig = plt.figure(figsize=(2,2))
    dd = df_meta.loc[ind,:].groupby(['subj_idx', 'session', 'bins_pupil']).mean()
    for subj, d in dd.groupby(['subj_idx']):
        sns.regplot(d['pupil_0'], d['calcium_residuals'], fit_reg=False)
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'correlations', 'state_dependence2_{}.pdf'.format(walk)))