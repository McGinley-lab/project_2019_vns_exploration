import os, sys
import glob
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import h5py
from joblib import Parallel, delayed
import imp

from IPython import embed as shell

import vns_analyses
from vns_analyses import analyse_exploration_session

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

# FIXME: clean-up except statements
# FIXME: motion is 0 is some cases --> set to NaN!

raw_dir = '/media/external1/raw/vns_exploration/data/'
project_dir = '/home/jwdegee/vns_exploration'
fig_dir = os.path.join(project_dir, 'figures', 'exploration')
# fig_dir = '/media/external2/projects/vns_exploration/figures/exploration'
data_dir = os.path.join(project_dir, 'data', 'exploration')

df_stim = pd.read_excel(os.path.join(raw_dir, 'stim_summary.xlsx'), sheet_name='python')

tasks = []
# for cuff_type in ['intact',]:
for cuff_type in ['intact', 'single', 'double']:
    subjects = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, '*'))]
    for subj in subjects:
        sessions = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, subj, '*'))]
        for ses in sessions:
            file_tdms = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*.tdms'))
            file_meta = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*preprocessed.mat'))
            file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*.hdf'))
            if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1):
                tasks.append((file_meta[0], file_pupil[0], file_tdms[0], fig_dir, subj, cuff_type, df_stim))

            # if (subj == 'V5N1')&(ses=='4a'):
            #     file_tdms = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*.tdms'))
            #     file_meta = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*preprocessed.mat'))
            #     file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*.hdf'))
            #     if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1):
            #         tasks.append((file_meta[0], file_pupil[0], file_tdms[0], fig_dir, subj, cuff_type, df_stim))

preprocess = False
if preprocess:
    n_jobs = 64
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(analyse_exploration_session)(*task) for task in tasks)

    # sort:
    df_meta = pd.concat([res[i][0] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_x = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_y = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][6] for i in range(len(res))], axis=0).reset_index(drop=True)

    # save:
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
    epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
    epochs_b.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
    df_meta.to_csv(os.path.join(data_dir, 'meta_data.csv'))

# load:
print('loading data')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil') * 100
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid') * 100
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
print('finished loading data')

# round:
df_meta['date'] = pd.to_datetime(df_meta.date, format='%Y/%m/%d', errors = 'coerce')
df_meta['amplitude'] = np.round(df_meta['amplitude'], 3)
df_meta['width'] = np.round(df_meta['width'], 3)
df_meta['rate'] = np.round(df_meta['rate'], 3)
df_meta['charge'] = np.round(df_meta['charge'], 3)
df_meta['charge_ps'] = np.round(df_meta['charge_ps'], 3)

# plot charges:
fig = vns_analyses.plot_pupil_responses_matrix_()
fig.savefig(os.path.join(fig_dir, 'charges.pdf'))

# # set responses incl. blinks to NaN:
# x = epochs_p.columns
# bad_epochs = (np.sum(pd.isna(epochs_p.loc[:,(x>0)&(x<10)]), axis=1) > 1) | (np.mean(epochs_b.loc[:,(x>0)&(x<10)], axis=1) > 0.1) 
# epochs_v = epochs_v.loc[~bad_epochs,:].reset_index(drop=True)
# epochs_p = epochs_p.loc[~bad_epochs,:].reset_index(drop=True)
# epochs_l = epochs_l.loc[~bad_epochs,:].reset_index(drop=True)
# epochs_x = epochs_x.loc[~bad_epochs,:].reset_index(drop=True)
# epochs_y = epochs_y.loc[~bad_epochs,:].reset_index(drop=True)
# epochs_b = epochs_b.loc[~bad_epochs,:].reset_index(drop=True)
# df_meta = df_meta.loc[~bad_epochs,:].reset_index(drop=True)

# cutoffs:
leak_cut_off = 0.25
velocity_cutoff = (-0.005, 0.005)
blink_cutoff = 0.1

# session leaks:
# --------------
# fig = plt.figure(figsize=(9,3))
# for i, cuff_type in enumerate(['intact', 'single', 'double']):
#     ax = fig.add_subplot(1,3,i+1)
#     ax.hist(df_meta.loc[df_meta['cuff_type']==cuff_type, :].groupby(['subj_idx', 'session']).mean()['leak'], bins=50)
#     plt.axvline(leak_cut_off, ls='--', color='r')
#     plt.title(cuff_type)
#     plt.xlabel('Leak fraction')
#     plt.ylabel('Sessions (#)')
# sns.despine(trim=False, offset=3)
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'leak_fractions.pdf'))

fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(1,1,1)
ax.hist(df_meta.groupby(['subj_idx', 'session']).mean()['leak'], bins=25)
plt.axvline(leak_cut_off, ls='--', color='r')
plt.xlabel('Leak fraction')
plt.ylabel('Sessions (#)')
sns.despine(trim=False, offset=3)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'leak_fractions.pdf'))

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
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            }

ylims = {
            'velocity' : (-0.1, 0.8),
            'walk' : (0,0.75),
            'pupil' : (-5, 20),
            'eyelid' : (-1, 3),
            'blink' : (0, 0.3),
            # 'eyemovement' : (0, 0.2),
            }

for key in timewindows.keys():
    x = epochs[key.split('_')[0]].columns
    window1, window2 = timewindows[key]
    resp = epochs[key.split('_')[0]].loc[:,(x>=window1[0])&(x<=window1[1])].mean(axis=1)
    if window2[0] == None: 
        df_meta[key] = resp
    else:
        baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1)
        df_meta[key] = resp-baseline    
    # if key == 'blink':
    #     df_meta['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)

df_meta['walk_-3'] = ((df_meta['velocity_-3'] < velocity_cutoff[0])|(df_meta['velocity_-3'] > velocity_cutoff[1])).astype(int)
df_meta['walk_-2'] = ((df_meta['velocity_-2'] < velocity_cutoff[0])|(df_meta['velocity_-2'] > velocity_cutoff[1])).astype(int)
df_meta['walk_-1'] = ((df_meta['velocity_-1'] < velocity_cutoff[0])|(df_meta['velocity_-1'] > velocity_cutoff[1])).astype(int)
df_meta['walk_0'] = ((df_meta['velocity_0'] < velocity_cutoff[0])|(df_meta['velocity_0'] > velocity_cutoff[1])).astype(int)
df_meta['walk_1'] = ((df_meta['velocity_1'] < velocity_cutoff[0])|(df_meta['velocity_1'] > velocity_cutoff[1])).astype(int)

# remove blink trials:
# df_meta['blink'] = ((df_meta[['blink_-3', 'blink_-2', 'blink_-1']].mean(axis=1)>blink_cutoff) | (df_meta['blink_0']>blink_cutoff) | (df_meta['blink_1']>blink_cutoff) | np.isnan(df_meta['pupil_0']) | np.isnan(df_meta['pupil_1']) )
df_meta['blink'] = ( (df_meta['blink_0']>blink_cutoff) | (df_meta['blink_1']>blink_cutoff) | np.isnan(df_meta['pupil_0']) | np.isnan(df_meta['pupil_1']) )

epochs_v = epochs_v.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_l = epochs_l.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_x = epochs_x.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_y = epochs_y.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_b = epochs_b.loc[df_meta['blink']==0,:].reset_index(drop=True)
df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

# indices:
# --------
ind_clean_w = ~(np.isnan(df_meta['velocity_1'])|np.isnan(df_meta['velocity_0']))

ind_g = (df_meta['date']<'2018-10-14')
ind_u = (df_meta['date']>'2018-10-14')
ind_g.loc[(df_meta['leak']<leak_cut_off)] = False

# ind_g = df_meta['date']<'2018-10-14'
ind_s = ((df_meta['velocity_1'] >= velocity_cutoff[0]) & (df_meta['velocity_1'] <= velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])
ind_w = ((df_meta['velocity_1'] < velocity_cutoff[0]) | (df_meta['velocity_1'] > velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])

# # remove blink trials:
# df_meta['blink'] = ((df_meta[['blink_-3', 'blink_-2', 'blink_-1', 'blink_0']].mean(axis=1)>blink_cutoff) | (df_meta['blink_1']>blink_cutoff))
# epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
# df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

shell()

mins = []
medians = []
maxs = []
for (subj, ses), df in df_meta.groupby(['subj_idx', 'session']):
    mins.append(df['time'].diff().max())
    medians.append(df['time'].diff().median())
    maxs.append(df['time'].diff().min())
print(min(mins))
print(np.median(medians))
print(max(maxs))


# correct scalars:
for group in [ind_u, ind_g]:
    df_meta, figs = vns_analyses.correct_scalars(df_meta, group=group, velocity_cutoff=velocity_cutoff, ind_clean_w=ind_clean_w)
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

# # plot 1 -- baseline dependence:
# fig = plt.figure(figsize=(6,2))
# for i, measure in enumerate(['pupil', 'pupil_c',]):
#     means = df_meta.loc[ind_u,:].groupby('bins_pupil')[['pupil_c', 'pupil', 'pupil_0']].mean()
#     sems = df_meta.loc[ind_u,:].groupby('bins_pupil')[['pupil_c', 'pupil', 'pupil_0']].sem()
#     ax = fig.add_subplot(1,3,i+1)
#     plt.errorbar(means['pupil_0'], means[measure], yerr=sems[measure], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
#     x = np.linspace(min(means['pupil_0']),max(means['pupil_0']),100)
#     popt,pcov = curve_fit(vns_analyses.gaus, means['pupil_0'],means[measure],)
#     plt.plot(x, vns_analyses.gaus(x,*popt), '--', color='r')
#     plt.xlabel('Baseline pupil')
#     plt.ylabel('Pupil response')
#     plt.title(measure)
# plt.tight_layout()
# sns.despine(trim=False, offset=3)
# fig.savefig(os.path.join(fig_dir, 'pupil_state_dependence.pdf'))


# subtract baselines:
# -------------------

x = epochs_p.columns
epochs_p = epochs_p - np.atleast_2d(epochs_p.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_l.columns
epochs_l = epochs_l - np.atleast_2d(epochs_l.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T
x = epochs_v.columns
epochs_v = epochs_v - np.atleast_2d(epochs_v.loc[:,(x>=-0.1)&(x<=-0.0)].mean(axis=1)).T
# x = epochs_x.columns
# epochs_x = epochs_x - np.atleast_2d(epochs_x.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)).T
# x = epochs_y.columns
# epochs_y = epochs_y - np.atleast_2d(epochs_y.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)).T
# x = epochs_b.columns
# epochs_b = epochs_b - np.atleast_2d(epochs_b.loc[:,(x>=-5.5)&(x<=-0.5)].mean(axis=1)).T

epochs = {
            'velocity' : epochs_v,
            'pupil' : epochs_p,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            }

fig = plt.figure(figsize=(20,3))
ax = fig.add_subplot(111)
sns.barplot(x='subj_idx', y='pupil', hue='session', data=df_meta.loc[ind_u&(df_meta['cuff_type']=='intact'),:], ax=ax)
ax.legend().remove()
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'pupil_responses_u.pdf'))

fig = plt.figure(figsize=(20,3))
ax = fig.add_subplot(111)
sns.barplot(x='subj_idx', y='pupil', hue='session', data=df_meta.loc[ind_g&(df_meta['cuff_type']=='intact'),:], ax=ax)
ax.legend().remove()
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'pupil_responses_g.pdf'))

# # pupil histogram:
# # ----------------
# fig = plt.figure(figsize=(4,4))
# plt.hist(df_meta.loc[ind_w_0,'pupil_c'], histtype='stepfilled', alpha=0.2, bins=100, color='r', label='walking')
# plt.hist(df_meta.loc[ind_s_0,'pupil_c'], histtype='stepfilled', alpha=0.2, bins=100, color='g', label='still')
# plt.axvline(df_meta.loc[ind_w_0,'pupil_c'].mean(), color='r')
# plt.axvline(df_meta.loc[ind_s_0,'pupil_c'].mean(), color='g')
# plt.xlabel('Pupil response')
# plt.ylabel('Trials')
# plt.legend()
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'histograms.pdf'))


# velocity historgram:
# --------------------

bins = 30

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(331)
ax.hist(df_meta.loc[ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)

ax = fig.add_subplot(332)
ax.hist(df_meta.loc[ind_s&ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_s[ind_u&ind_clean_w])/np.sum(ind_u&ind_clean_w)*100,1)))

ax = fig.add_subplot(333)
ax.hist(df_meta.loc[ind_w&ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_w[ind_u&ind_clean_w])/np.sum(ind_u&ind_clean_w)*100,1)))

ax = fig.add_subplot(334)
ax.hist(df_meta.loc[ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)

ax = fig.add_subplot(335)
ax.hist(df_meta.loc[ind_s&ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_s[ind_g&ind_clean_w])/np.sum(ind_g&ind_clean_w)*100,1)))

ax = fig.add_subplot(336)
ax.hist(df_meta.loc[ind_w&ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
# plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_w[ind_g&ind_clean_w])/np.sum(ind_g&ind_clean_w)*100,1)))
plt.tight_layout()
sns.despine(trim=False, offset=3)
fig.savefig(os.path.join(fig_dir, 'velocity_hist.pdf'))

# # for Matt:
# df = df_meta.loc[(df_meta['cuff_type']=='intact')&ind_u, ['pupil_c2', 'charge', 'amplitude', 'width', 'rate']]
# df = df.rename(columns={'pupil_c2': 'pupil',})
# df['charge'] = np.round(df['charge'],3)
# df['amplitude'] = np.round(df['amplitude'],3)
# df.to_csv(os.path.join(fig_dir, 'data_exploration.csv'))

# # variance explained charge:
# ind = (df_meta['cuff_type'] == 'intact') & ind_u
# x = np.array(df_meta.loc[ind,['charge', 'rate']])
# y = np.array(df_meta.loc[ind,['pupil_c']]).ravel()
# func = vns_analyses.log_logistic_3d
# popt, pcov = curve_fit(func, x, y, 
#                 method='dogbox', bounds=([0, 0, 0, 0, 0,], [np.inf, np.inf, np.inf, np.inf, np.inf,]))
# predictions = func(x, *popt) 
# r2 = (sp.stats.pearsonr(y, predictions)[0]**2) * 100
# print(r2)

fig = plt.figure(figsize=(8,8))
d = df_meta.loc[(df_meta['width']>0.2),:].groupby(['subj_idx', 'date', 'amplitude_bin']).mean().reset_index()
d['leak_fraction'] = 1 - (d['amplitude_m']/d['amplitude'])
plt_nr = 1
for measure in ['leak_fraction', 'impedance']:
    ax = fig.add_subplot(2,1,plt_nr)
    for amp, dd in d.groupby(['amplitude_bin']):
        dd = dd.sort_values('date', ascending=True)
        d_mean = dd.groupby('date').mean().reset_index()
        d_sem = dd.groupby('date').sem().reset_index()
        plt.errorbar(d_mean['date'], d_mean[measure], yerr=d_sem[measure], fmt='-o', label='bin {}'.format(amp), alpha=0.5)
    plt.ylabel(measure)
    plt.legend()
    plt_nr += 1
plt.tight_layout()
sns.despine(trim=False, offset=3)  
fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_time.pdf'))

def leak_fractions(df):
    d = df.loc[(df['width']>0.2),:].groupby(['subj_idx', 'session', 'amplitude_bin']).mean().reset_index()
    d['leak_fraction'] = 1 - (d['amplitude_m']/d['amplitude'])
    
    fig = plt.figure(figsize=(6,4))
    plt_nr = 1
    ax = fig.add_subplot(2,3,plt_nr)
    for (subj, ses), dd in d.groupby(['subj_idx', 'session']):
        plt.plot(dd['amplitude'], dd['leak_fraction'])
    plt.xlabel('Amplitude intended')
    plt.ylabel('Leak fraction')
    
    ylim = ax.get_ylim()

    plt_nr += 1
    
    for b in [0,1,2,3,4]:
        ax = fig.add_subplot(2,3,plt_nr)
        dd = d.loc[d['amplitude_bin']==b,:].groupby(['subj_idx', 'session']).mean().reset_index()
        ind = (~np.isnan(dd['impedance']))# & (d['amplitude_bin']==4)
        sns.regplot(dd.loc[ind,'impedance'], dd.loc[ind,'leak_fraction'], line_kws={'color': 'red'})
        
        r,p = sp.stats.pearsonr(dd.loc[ind,'impedance'], dd.loc[ind,'leak_fraction'])
        ax.set_title('r = {}, p = {}'.format(round(r,3), round(p,3)))

        ax.set_ylim(ylim)
        plt.xlabel('Impedance')
        plt.ylabel('Leak fraction')

        plt_nr += 1

    plt.tight_layout()
    sns.despine(trim=False, offset=3)    
    return fig

for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
    fig = vns_analyses.plot_param_preprocessing(df_meta.loc[ind,:])
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'average_{}.pdf'.format(title)))

    fig = leak_fractions(df_meta.loc[ind,:])
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_{}.pdf'.format(title)))

for cuff_type in ['intact', 'single', 'double']:
    for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
        cuff_ind = np.array(df_meta['cuff_type'] == cuff_type)
        ind = ind & cuff_ind
        fig = vns_analyses.plot_param_preprocessing(df_meta.loc[ind,:])
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'average_{}_{}.pdf'.format(title, cuff_type,)))

        fig = leak_fractions(df_meta.loc[ind,:])
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_{}_{}.pdf'.format(title, cuff_type,)))

        fig = plt.figure()
        sns.regplot(df_meta.loc[ind,'amplitude'], df_meta.loc[ind,'amplitude_m'])
        plt.title('{}_{}'.format(title, cuff_type,))
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'amplitude_scatters_{}_{}.pdf'.format(title, cuff_type,)))

# normalize per session:
means = df_meta.loc[(df_meta['cuff_type']=='intact')&ind_u].groupby(['subj_idx', 'session'])['pupil_c'].mean().reset_index()
stds = df_meta.loc[(df_meta['cuff_type']=='intact')&ind_u].groupby(['subj_idx', 'session'])['pupil_c'].std().reset_index()
cv = stds.copy()
cv['pupil_c'] = cv['pupil_c'] / means['pupil_c']

fig = plt.figure(figsize=(2,6))
ax = fig.add_subplot(311)
plt.hist(means['pupil_c'], bins=15)
plt.xlabel('mean pupil')
ax = fig.add_subplot(312)
plt.hist(stds['pupil_c'], bins=15)
plt.xlabel('std pupil')
ax = fig.add_subplot(313)
plt.hist(cv['pupil_c'], bins=5000)
plt.xlim(-30,30)
plt.xlabel('Cov. variation')
plt.tight_layout()
sns.despine(trim=False, offset=3)
fig.savefig(os.path.join(fig_dir, 'varation_across_session.pdf'))

# time courses:
# -------------
imp.reload(vns_analyses)
for measure in ['pupil', 'velocity', 'walk', 'eyelid',]:
# for measure in ['pupil',]:
    for cuff_type in ['intact', 'single', 'double']:
    # for cuff_type in ['intact']:
        cuff_ind = np.array(df_meta['cuff_type'] == cuff_type)
        for ind, title in zip([ind_u, ind_g], ['u', 'g']):
        # for ind, title in zip([ind_u], ['u']):
            if not os.path.exists(os.path.join(fig_dir, measure, title)):
                os.makedirs(os.path.join(fig_dir, measure, title))
            
            if ('velocity' in measure):
                ind = ind & ind_w & ind_clean_w
            if ('walk' in measure):
                ind = ind & ind_clean_w
            ind = ind & cuff_ind
            ylim = ylims[measure]
            if (measure == 'blink'):
                ylim = (ylim[0], ylim[1]/3)
            if (measure == 'pupil') & (title=='g'):
                ylim = (ylim[0], ylim[1]*2)
            
            if not measure == 'walk':
                x = np.array(epochs[measure].columns, dtype=float)
                fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[:,(x>=-20)&(x<=40)].loc[ind,::10], timewindows=timewindows, ylabel=measure+'_1', ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'timecourses_{}_{}_{}.pdf'.format(title, cuff_type, measure)))

            for measure_ext in ['', '_c', '_c2',]:
            # for measure_ext in ['_c3']:
                
                if measure_ext == '_c3':
                    ylim = (0,3)

                fig = vns_analyses.plot_scalars(df_meta.loc[ind &  ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                
                fig = vns_analyses.plot_scalars2(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars2_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))

                fig = vns_analyses.plot_scalars3(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars3_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))

                try:
                    fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, vmin=-ylim[1], vmax=ylim[1])
                    fig.savefig(os.path.join(fig_dir, measure, title, 'matrix_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                except:
                    pass

                try:
                    fig = vns_analyses.hypersurface2(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], z_measure=measure+measure_ext, ylim=(0,ylim[1]))
                    fig.savefig(os.path.join(fig_dir, measure, title, '3dplot_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                except:
                    pass
