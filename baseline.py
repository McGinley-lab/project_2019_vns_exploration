import os, sys
import glob
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import h5py
from joblib import Parallel, delayed

from IPython import embed as shell

import vns_analyses
from vns_analyses import analyse_baseline_session

# matplotlib.use("Qt4Agg")
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

raw_dir = '/media/external1/raw/vns_baseline/data/'
process_dir = '/media/external1/projects/vns_baseline/preprocess/'
project_dir = '/home/jwdegee/vns_exploration'
fig_dir = os.path.join(project_dir, 'figures', 'baseline')
data_dir = os.path.join(project_dir, 'data', 'baseline')

df_stim = pd.DataFrame([])

tasks = []
for cuff_type in ['intact']:
    subjects = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, '*'))]
    for subj in subjects:
        sessions = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, subj, '*'))]
        for ses in sessions:
            file_tdms = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*.tdms'))
            file_meta = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*preprocessed.mat'))
            # file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*.hdf'))
            file_pupil = os.path.join(process_dir, subj, ses, '{}_{}_df_pupil_preprocessed.hdf'.format(subj, ses))
            # if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1):
            if (len(file_meta)==1)&(len(file_tdms)==1)&os.path.exists(file_pupil):
                tasks.append((file_meta[0], file_pupil, file_tdms[0], fig_dir, subj, cuff_type,))

preprocess = False
if preprocess:
    n_jobs = 12
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(analyse_baseline_session)(*task) for task in tasks)

    # sort:
    df_meta = pd.concat([res[i][0] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_x = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_y = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)

    # save:
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    epochs_b.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
    df_meta.to_csv(os.path.join(data_dir, 'meta_data.csv'))
    # epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
    # epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')

# load:
print('loading data')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil') * 100
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid') * 100
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
epochs_s = epochs_p.diff(axis=1) * 50
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
# epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
# epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
print('finished loading data')

# sample size:
print('subjects: {}'.format(df_meta.loc[:,:].groupby(['subj_idx']).count().shape[0]))
print('sessions: {}'.format(df_meta.loc[:,:].groupby(['subj_idx', 'session']).count().shape[0]))

# cutoffs:
velocity_cutoff = (-0.005, 0.005)
blink_cutoff = 0.1

# # compute eye movement
# # --------------------
# epochs_xy = ((epochs_x.diff(axis=1)**2) + (epochs_y.diff(axis=1)**2))
# epochs_xy.replace(0, np.nan, inplace=True)
# epochs_xy = epochs_xy.fillna(method='pad', axis=1)
# epochs_xy = epochs_xy.rolling(window=100, center=True, min_periods=10, axis=1).mean()
# epochs_xy = epochs_xy * 500
# epochs_xy[epochs_b==1] = np.NaN

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
            'pupil_1' : [(2.5, 7.5), (None, None)],

            'slope_-3' : [(-25, -20), (None, None)],      #
            'slope_-2' : [(-20, -15), (None, None)],      #
            'slope_-1' : [(-15, -10), (None, None)],      #
            'slope_0' : [(-5, 0), (None, None)],          #
            'slope_1' : [(0, 5), (None, None)],

            'blink_-3' : [(-27.5, -22.5), (None, None)],  #
            'blink_-2' : [(-20, -15), (None, None)],      #
            'blink_-1' : [(-12.5, -7.5), (None, None)],   #
            'blink_0' : [(-5, 0), (None, None)],          #
            'blink_1' : [(2.5, 7.5), (None, None)],

            'eyelid_-3' : [(-27.5, -22.5), (None, None)],  #
            'eyelid_-2' : [(-20, -15), (None, None)],      #
            'eyelid_-1' : [(-12.5, -7.5), (None, None)],   #
            'eyelid_0' : [(-5, 0), (None, None)],          #
            'eyelid_1' : [(2.5, 7.5), (None, None)],
            
            }

epochs = {
            'velocity' : epochs_v,
            'pupil' : epochs_p,
            'slope' : epochs_s,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            }

ylims = {
            'velocity' : (-0.1, 0.5),
            'walk' : (0,1),
            'pupil' : (-5, 30),
            'slope' : (-5, 30),
            'eyelid' : (-1, 3),
            'blink' : (0, 0.3),
            # 'eyemovement' : (0, 0.2),
            }

for key in timewindows.keys():
    x = epochs[key.split('_')[0]].columns
    window1, window2 = timewindows[key]
    if 'slope' in key:
        # resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1)
        # resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].max(axis=1)
        resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].quantile(0.95, axis=1)
    else:
        resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1)
    if window2[0] == None: 
        baseline = 0
    else:
        baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1)
    df_meta[key] = resp-baseline
    # if key == 'blink':
    #     df_meta['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)

# remove blink trials:
# df_meta['blink'] = ((df_meta[['blink_-3', 'blink_-2', 'blink_-1']].mean(axis=1)>blink_cutoff) | (df_meta['blink_0']>blink_cutoff) | (df_meta['blink_1']>blink_cutoff) | np.isnan(df_meta['pupil_0']) | np.isnan(df_meta['pupil_1']) )
df_meta['blink'] = ( (df_meta['blink_0']>blink_cutoff) | (df_meta['blink_1']>blink_cutoff) | np.isnan(df_meta['pupil_0']) | np.isnan(df_meta['pupil_1']) )
epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_s = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

# indices:
# --------
ind_clean_w = ~(np.isnan(df_meta['velocity_1'])|np.isnan(df_meta['velocity_0']))
all_trials = np.ones(df_meta.shape[0], dtype=bool)

# correct scalars:
df_meta, figs = vns_analyses.correct_scalars(df_meta, group=all_trials, velocity_cutoff=velocity_cutoff, ind_clean_w=ind_clean_w)
figs[0].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean1.pdf'))
figs[1].savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean2.pdf'))
figs[2].savefig(os.path.join(fig_dir, 'slope_reversion_to_mean1.pdf'))
figs[3].savefig(os.path.join(fig_dir, 'slope_reversion_to_mean2.pdf'))
figs[4].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean1.pdf'))
figs[5].savefig(os.path.join(fig_dir, 'eyelid_reversion_to_mean2.pdf'))

# plot 3 -- baseline dependence:
# import statsmodels.formula.api as smf
# model = smf.ols("pupil ~ 1 + pupil_0", data=df_meta).fit()
# print(model.summary())
# model = smf.ols("pupil ~ 1 + pupil_0 + np.power(pupil_0, 2)", data=df_meta).fit()
# print(model.summary())
# model = smf.ols("pupil_c ~ 1 + pupil_0", data=df_meta).fit()
# print(model.summary())
# model = smf.ols("pupil_c ~ 1 + pupil_0 + np.power(pupil_0, 2)", data=df_meta).fit()
# print(model.summary())

# save source data:
df = df_meta.loc[:,['subj_idx', 'session', 'pupil_0', 'pupil_c']].reset_index(drop=True)
df = df.rename({'pupil_c':'pupil', 'pupil_0':'pupil_baseline'}, axis=1)
df.to_csv(os.path.join(data_dir, 'baseline_dependence_source_data_{}.csv'.format(measure.split('_')[0])))

# plot 1 -- time courses:
bins = np.array([-10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,10])
fig = vns_analyses.plot_pupil_responses(df_meta, epochs_p.loc[:, ::10], bins=bins, ylabel='Pupil response\n(% max)', ylim=(0, 1.2))
fig.savefig(os.path.join(fig_dir, 'pupil_timecourses.pdf'))

# df_meta['bins_pupil'] = df_meta.groupby(['subj_idx', 'session'])['pupil_0'].apply(make_bins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) 

# plot 1 -- baseline dependence:
df = df_meta.copy()
df['pupil_cc'] = np.NaN
df['include'] = 1
for (subj, ses), d in df.groupby(['subj_idx', 'session']):
    df.loc[d.index, 'pupil_cc'] = df.loc[d.index, 'pupil_c'] / df.loc[d.index, 'pupil_c'].mean()
    
    if df.loc[d.index, 'pupil_c'].mean() < 1:
        df.loc[d.index, 'include'] = 0

for X, bin_measure in zip(['pupil_0',], ['bins_pupil']):
    for Y in ['pupil', 'pupil_c', 'pupil_cc']:
        if not X == Y:
            # plot:
            fig = vns_analyses.plot_correlation(df, X=X, Y=Y, bin_measure=bin_measure, scatter=True)
            fig.savefig(os.path.join(fig_dir, 'state_dependence_{}_{}.pdf'.format(X,Y)))

for X, bin_measure in zip(['pupil_0',], ['bins_pupil']):
    for Y in ['pupil', 'pupil_c', 'pupil_cc']:
        if not X == Y:
            # plot:
            fig = vns_analyses.plot_correlation(df.loc[df['include']==1,:], X=X, Y=Y, bin_measure=bin_measure, scatter=True)
            fig.savefig(os.path.join(fig_dir, 'state_dependence_{}_{}_subselected.pdf'.format(X,Y)))

# figure:
fig = plt.figure(figsize=(12,8))
plt_nr = 1
for (subj, ses), df in df_meta.groupby(['subj_idx', 'session']):
    ax = fig.add_subplot(4,6,plt_nr)
    plt.hist(df['pupil_0'], bins=15)
    plt.title('{} ses {}'.format(subj, ses))
    plt_nr += 1
plt.hist(df_meta['pupil_0'], color='r', bins=15)
plt.title('group')
sns.despine(trim=False, offset=3)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'baseline_histograms.pdf'))