import os, sys
import glob
import numpy as np
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
    n_jobs = 24
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
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
print('finished loading data')

# add charge:
df_meta['amplitude_bin'] = df_meta['amplitude_m_bin'].copy()
df_meta['amplitude'] = ((df_meta['amplitude_bin']+1)*0.2)-0.1
df_meta['width'] = (2**(df_meta['width_bin']+1))/20 
df_meta['charge'] = df_meta['amplitude']*df_meta['width']
df_meta['charge_ps'] = df_meta['amplitude']*df_meta['width']*df_meta['rate']
df_meta['charge_bin'] = df_meta.groupby(['subj_idx', 'session'])['charge'].apply(pd.qcut, q=5, labels=False)
df_meta['charge_ps_bin'] = df_meta.groupby(['subj_idx', 'session'])['charge_ps'].apply(pd.qcut, q=5, labels=False)

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
fig = plt.figure(figsize=(9,3))
for i, cuff_type in enumerate(['intact', 'single', 'double']):
    ax = fig.add_subplot(1,3,i+1)
    ax.hist(df_meta.loc[df_meta['cuff_type']==cuff_type, :].groupby(['subj_idx', 'session']).mean()['leak'], bins=50)
    plt.axvline(leak_cut_off, ls='--', color='r')
    plt.title(cuff_type)
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
            'pupil' : (-0.05, 0.3),
            'eyelid' : (-0.01, 0.03),
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
ind_u = (df_meta['leak']<leak_cut_off)
ind_g  = (df_meta['leak']>=leak_cut_off)
ind_s = ((df_meta['velocity_1'] >= velocity_cutoff[0]) & (df_meta['velocity_1'] <= velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])
ind_w = ((df_meta['velocity_1'] < velocity_cutoff[0]) | (df_meta['velocity_1'] > velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])

# # remove blink trials:
# df_meta['blink'] = ((df_meta[['blink_-3', 'blink_-2', 'blink_-1', 'blink_0']].mean(axis=1)>blink_cutoff) | (df_meta['blink_1']>blink_cutoff))
# epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
# df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

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
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(221)
ax.hist(df_meta.loc[ind_s&ind_u, 'velocity'], bins=50, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_s[ind_u])/np.sum(ind_u)*100,1)))

ax = fig.add_subplot(222)
ax.hist(df_meta.loc[ind_s&ind_g, 'velocity'], bins=50, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_s[ind_g])/np.sum(ind_g)*100,1)))

ax = fig.add_subplot(223)
ax.hist(df_meta.loc[ind_w&ind_u, 'velocity'], bins=50, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_w[ind_u])/np.sum(ind_u)*100,1)))

ax = fig.add_subplot(224)
ax.hist(df_meta.loc[ind_w&ind_g, 'velocity'], bins=50, density=False, histtype='stepfilled')
plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
plt.ylim(0,800)
ax.set_title('{}%'.format(round(np.sum(ind_w[ind_g])/np.sum(ind_g)*100,1)))
plt.tight_layout()
sns.despine(trim=False, offset=3)
fig.savefig(os.path.join(fig_dir, 'velocity_hist.pdf'))

# for Matt:
df = df_meta.loc[(df_meta['cuff_type']=='intact')&ind_u, ['pupil_c2', 'charge', 'amplitude', 'width', 'rate']]
df = df.rename(columns={'pupil_c2': 'pupil',})
df['charge'] = np.round(df['charge'],3)
df['amplitude'] = np.round(df['amplitude'],3)
df.to_csv(os.path.join(fig_dir, 'data_exploration.csv'))

shell()

# 3d plot:
fig = vns_analyses.hypersurface(df_meta.loc[ind_u & (df_meta['cuff_type']=='intact'),:].reset_index(), z_measure='pupil_c')
fig.savefig(os.path.join(fig_dir, '3d_surface_pupil.pdf'))

# time courses:
# -------------
imp.reload(vns_analyses)
for measure in ['pupil', 'velocity', 'walk', 'eyelid',]:
# for measure in ['pupil',]:
    for cuff_type in ['intact', 'single', 'double']:
    # for cuff_type in ['intact']:
        cuff_ind = np.array(df_meta['cuff_type'] == cuff_type)
        # for ind, title in zip([ind_u, ind_g, ind_u&ind_s, ind_u&ind_w], ['u', 'g', 'us', 'uw']):
        for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
            
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
            if (measure == 'pupil') & (title=='us'):
                ylim = (ylim[0], ylim[1]/3)

            if not measure == 'walk':
                if measure == 'velocity':
                    fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[ind,::10], timewindows=timewindows, ylabel=measure+'_1', ylim=(-ylim[1],ylim[1]))
                else:
                    fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[ind,::10], timewindows=timewindows, ylabel=measure+'_1', ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'timecourses_{}_{}_{}.pdf'.format(title, cuff_type, measure)))

            for measure_ext in ['', '_c', '_c2']:
                
                fig = vns_analyses.plot_scalars(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                
                fig = vns_analyses.plot_scalars2(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars2_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                
                fig = vns_analyses.plot_scalars3(df_meta.loc[ind, :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                fig.savefig(os.path.join(fig_dir, measure, title, 'scalars3_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))

                try:
                    fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind, :], measure=measure+measure_ext, vmin=-ylim[1], vmax=ylim[1])
                    fig.savefig(os.path.join(fig_dir, measure, title, 'matrix_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                except:
                    pass

# #############################
# # REGRESSION ################

# import statsmodels.api as sm
# X = df_meta.loc[ind_u&ind_clean, ['train_amp']]
# # X = df_meta.loc[ind_u&ind_clean, ['train_amp', 'train_width', 'train_rate']]
# X = sm.add_constant(X)
# Y = df_meta.loc[ind_u&ind_clean, 'pupil_resp_0_all']

# Ymin = Y.min()
# Ymax = Y.max()
# Y = (Y - Ymin) / (Ymax-Ymin)

# # GLM:
# model = sm.GLM(Y, X, family=sm.families.Binomial(link=sm.genmod.families.links.cauchy()),)
# res = model.fit()
# print(res.summary())

# # model = sm.Logit(Y, X)
# # res = model.fit()
# # print(res.summary())

# Y_test = df_meta.loc[ind_u&ind_clean,:].groupby('train_amp_bin').mean()['pupil_resp_0_all']
# Y_test = (Y_test - Ymin) / (Ymax-Ymin)
# X_test = df_meta.loc[ind_u&ind_clean,:].groupby('train_amp_bin').mean()['train_amp']
# X_test = sm.add_constant(X_test)

# plt.plot(X_test['train_amp'], Y_test, 'o')
# plt.plot(X_test['train_amp'], res.predict(X_test))
# # plt.xscale('log')
# # plt.yscale('log')
# plt.show()







# # plot scalars:
# # -------------

# for key in timewindows.keys():
#     for ind, title in zip([ind_u, ind_g, ind_u&ind_s, ind_u&ind_w], ['u', 'g', 'us', 'uw']):
#         if not (measure == 'velocity') | (measure == 'blink'):
#             ind = ind & ind_clean
#         for i, window in enumerate(timewindows[key]):
#             measure = '{}_resp_{}'.format(key, i)
#             fig = vns_analyses.catplot_scalars(df_meta.loc[ind,:], measure=measure, ylim=ylims[key], )
#             fig.savefig(os.path.join(fig_dir, 'responses_{}_{}_{}_catplot.pdf'.format(title, key, i)))

# for ind, title in zip([ind_u, ind_g,], ['u', 'g']):
#     fig = vns_analyses.catplot_nr_trials(df_meta, ind1=ind, ind2=ind&ind_s, )
#     fig.savefig(os.path.join(fig_dir, 'trials_still_{}_catplot.pdf'.format(title)))
#     fig = vns_analyses.catplot_nr_trials(df_meta, ind1=ind, ind2=ind&ind_clean, )
#     fig.savefig(os.path.join(fig_dir, 'trials_clean_{}_catplot.pdf'.format(title)))
#     fig = vns_analyses.catplot_nr_trials(df_meta, ind1=ind, ind2=ind&ind_s&ind_clean,)
#     fig.savefig(os.path.join(fig_dir, 'trials_still_clean_{}_catplot.pdf'.format(title)))

# df = df_meta.loc[
#             # (df_meta['train_amp_bin'] == 4)&
#             # (df_meta['train_width_bin'] == 3)&
#             (df_meta['train_amp_corr'] > 0.4)&(df_meta['train_amp_corr'] < 0.7)&
#             (df_meta['train_width'] == 0.2)|(df_meta['train_width'] == 0.4)&
#             (df_meta['train_rate'] == 20),:]
# df = df.groupby(['subj_idx', 'session', 'cuff_type']).mean().reset_index()
# df = df.loc[df['session_amp_leak']>0,:]

# df.loc[df.session_amp_leak<0.25,'pupil_resp_2'].mean()

# for measure in ['pupil_resp_1', 'blink_resp_1', 'eye_resp_1', 'pupil_resp_2', 'blink_resp_2', 'eye_resp_2']:
#     fig = plt.figure(figsize=(3,3))
#     plt.plot(df.loc[df['cuff_type']=='double', 'session_amp_leak'], df.loc[df['cuff_type']=='double', measure], 'o', markeredgewidth=0.5, markeredgecolor='w', color='red')
#     plt.plot(df.loc[df['cuff_type']=='single', 'session_amp_leak'], df.loc[df['cuff_type']=='single', measure], 'o', markeredgewidth=0.5, markeredgecolor='w', color='orange')
#     plt.plot(df.loc[df['cuff_type']=='intact', 'session_amp_leak'], df.loc[df['cuff_type']=='intact', measure], 'o', markeredgewidth=0.5, markeredgecolor='w', color='g')
#     plt.xlim(0,1)
#     plt.xlabel('Leak fraction')
#     plt.ylabel(measure)
#     plt.axvline(cut_off, ls='--', color='r')
#     sns.despine(trim=False, offset=3)
#     plt.tight_layout()
#     fig.savefig(os.path.join(fig_dir, 'scatter_{}.pdf'.format(measure)))


# # matrices:
# # ---------
# fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind_u, :], vmin=-0.2, vmax=0.2)
# fig.savefig(os.path.join(fig_dir, 'pupil_responses_matrix_u.pdf'))
# fig = vns_analyses.plot_pupil_responses_matrix(df_meta.loc[ind_g, :], vmin=-0.2, vmax=0.2)
# fig.savefig(os.path.join(fig_dir, 'pupil_responses_matrix_g.pdf'))

# import ternary
# scale = 60

# figure, tax = ternary.figure(scale=scale)
# tax.heatmap(shannon_entropy, boundary=True, style="triangular")
# tax.boundary(linewidth=2.0)
# tax.set_title("Shannon Entropy Heatmap")

# tax.show()


# def generate_random_heatmap_data(scale=5):
#     from ternary.helpers import simplex_iterator
#     import random
#     d = dict()
#     for (i,j,k) in simplex_iterator(scale):
#         d[(i,j)] = random.random()
#     return d



# d = {
#     (0, 0): 0.5671351527830725,
#     (0, 1): 0.6215131076044016,
#     (0, 2): 0.012781753598793633,
#     (0, 3): 0.707684078957192,
#     (0, 4): 0.9650973183040746,
#     (1, 0): 0.3969760579643895,
#     (1, 1): 0.3097635940052216,
#     (1, 2): 0.9531730871255774,
#     (1, 3): 0.9364529229723594,
#     (2, 0): 0.20446303966933876,
#     (2, 1): 0.14090809049974928,
#     (2, 2): 0.47887819021958244,
#     (3, 0): 0.0018001448519757712,
#     (3, 1): 0.8081398969569612,
#     (4, 0): 0.7275761581090349
#     }



# scale = 4
# d = generate_random_heatmap_data(scale)
# figure, tax = ternary.figure(scale=scale)
# tax.heatmap(d)
# tax.boundary()
# tax.set_title("Heatmap Test: Hexagonal")




# ################################################################################################################################################################################################################

# # session-wise walking probabilites:
# averages = df.groupby(['subj_idx', 'session']).mean().reset_index()
# averages2 = df.groupby(['subj_idx', 'session', 'walk_b']).mean().reset_index()
# probs = (df.groupby(['subj_idx', 'session', 'walk_b']).sum()['walk'] / df.groupby(['subj_idx', 'session'])['ones'].sum()).reset_index()
# probs.columns = ['subj_idx', 'session', 'walk_b', 'p_walk']

# # plot reversion to mean:
# fig = vns_analyses.plot_reversion_to_mean_correction(df.loc[df['walk_transition']==3,:], measure='velocity', func=vns_analyses.quadratic)
# fig.savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean.pdf'))

# # plot across sessions:
# titles = ['still', 'walk']
# fig = plt.figure(figsize=(8,2))
# plt_nr = 1
# for i in range(2):
#     ax = fig.add_subplot(1,4,plt_nr)
#     x = np.array(averages['velocity'])
#     y = np.array(probs.loc[probs['walk_b']==i, 'p_walk'])
#     x = np.log10(x)
#     # y = np.log10(y)
#     y = sp.special.logit(y)
#     ax.scatter(x,y)
#     func = vns_analyses.linear
#     popt, pcov = curve_fit(func, x, y,)
#     ax.plot(np.linspace(min(x), max(x), 100), func(np.linspace(min(x), max(x), 100), *popt), color='k', ls='-', zorder=10)
#     ax.set_xlabel('Velocity (dm / s)')
#     ax.set_ylabel('P(walk)')
#     ax.set_title(titles[i])
#     plt_nr += 1
# for i in range(2):
#     ax = fig.add_subplot(1,4,plt_nr)
#     x = np.array(averages['velocity'])
#     y = np.array(averages2.loc[averages2['walk_b']==i, 'velocity'])
#     x = np.log10(x)
#     y = np.log10(y)
#     ax.scatter(x,y)
#     try:
#         func = vns_analyses.linear
#         popt, pcov = curve_fit(func, x, y,)
#         ax.plot(np.linspace(min(x), max(x), 100), func(np.linspace(min(x), max(x), 100), *popt), color='k', ls='-', zorder=10)
#     except:
#         pass
#     ax.set_xlabel('Velocity (dm / s)')
#     ax.set_ylabel('Velocity change')
#     ax.set_title(titles[i])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt_nr += 1
# sns.despine(trim=False, offset=3)
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'velocity_session_wise.pdf'))





# for walk_b in [1]:
    
#     # session-wise walking probabilites:
#     averages = df.loc[df['walk_b']==walk_b,:].groupby(['subj_idx', 'bins_velocity']).mean().reset_index()
#     averages2 = df.loc[df['walk_b']==walk_b,:].groupby(['subj_idx', 'bins_velocity', 'walk_b']).mean().reset_index()
#     probs = (df.loc[df['walk_b']==walk_b,:].groupby(['subj_idx', 'bins_velocity', 'walk_b']).sum()['walk'] / df.loc[df['walk_b']==walk_b,:].groupby(['subj_idx', 'bins_velocity'])['ones'].sum()).reset_index()
#     probs.columns = ['subj_idx', 'bins_velocity', 'walk_b', 'p_walk']


#     fig = plt.figure(figsize=(8,10))
#     for i, s in enumerate(df['subj_idx'].unique()):
#         ax = fig.add_subplot(5,4,i+1)
#         x = np.array(averages.loc[averages['subj_idx']==s,:].groupby('bins_velocity').mean()['velocity'])
#         y = np.array(probs.loc[probs['subj_idx']==s,:].groupby('bins_velocity').mean()['p_walk'])
#         # y = sp.special.logit(y)
#         ax.scatter(x,y)
#         try:
#             func = vns_analyses.linear
#             popt, pcov = curve_fit(func, x, y,)
#             ax.plot(np.linspace(min(x), max(x), 100), func(np.linspace(min(x), max(x), 100), *popt), color='k', ls='-', zorder=10)
#         except:
#             pass   
#         plt.ylim(-0.1,1.1)
#         plt.title('{} - p(walk)={}%'.format(s, int(100*df.loc[(df['subj_idx']==s)&(df['walk_b']==walk_b),'walk'].mean())))
#     sns.despine(trim=False, offset=3)
#     plt.tight_layout()
#     plt.show()



# ################################################################################################################################################################################################################

# # session-wise walking probabilites:
# averages = df.groupby(['subj_idx', 'session']).mean().reset_index()
# averages2 = df.groupby(['subj_idx', 'session', 'walk_transition']).mean().reset_index()
# probs_s = (df.loc[(df['walk_b']==0),:].groupby(['subj_idx', 'session', 'walk_transition']).count()['ones'] / df.loc[(df['walk_b']==0),:].groupby(['subj_idx', 'session'])['ones'].sum()).reset_index()
# probs_w = (df.loc[(df['walk_b']==1),:].groupby(['subj_idx', 'session', 'walk_transition']).count()['ones'] / df.loc[(df['walk_b']==1),:].groupby(['subj_idx', 'session'])['ones'].sum()).reset_index()

# # plot reversion to mean:
# fig = vns_analyses.plot_reversion_to_mean_correction(df.loc[df['walk_transition']==3,:], measure='velocity', func=vns_analyses.quadratic)
# fig.savefig(os.path.join(fig_dir, 'velocity_reversion_to_mean.pdf'))

# # plot across sessions:

# ylabels = ['(still -> still)', '(still -> walk)', '(walk -> still)', '(walk -> walk)']
# fig = plt.figure(figsize=(6,4))
# plt_nr = 1
# for i in range(4):
#     ax = fig.add_subplot(2,4,plt_nr)
#     x = np.array(averages['velocity'])
#     if i < 2:
#         y = np.array(probs_s.loc[probs_s['walk_transition']==i, 'ones'])
#     else:
#         y = np.array(probs_w.loc[probs_w['walk_transition']==i, 'ones'])
#     x = np.log10(x)
#     # y = np.log10(y)
#     y = sp.special.logit(y)
#     ax.scatter(x,y)
#     func = vns_analyses.linear
#     popt, pcov = curve_fit(func, x, y,)
#     ax.plot(np.linspace(min(x), max(x), 100), func(np.linspace(min(x), max(x), 100), *popt), color='k', ls='-', zorder=10)
#     ax.set_xlabel('Velocity (dm / s)')
#     ax.set_ylabel('P'+ylabels[i])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt_nr += 1
# for i in range(4):
#     ax = fig.add_subplot(2,4,plt_nr)
#     x = np.array(averages['velocity'])
#     y = np.array(averages2.loc[averages2['walk_transition']==i, 'velocity'])
#     x = np.log10(x)
#     # y = np.log10(y)
#     ax.scatter(x,y)
#     func = vns_analyses.linear
#     try:
#         popt, pcov = curve_fit(func, x, y,)
#         ax.plot(np.linspace(min(x), max(x), 100), func(np.linspace(min(x), max(x), 100), *popt), color='k', ls='-', zorder=10)
#     except:
#         pass
#     ax.set_xlabel('Velocity (dm / s)')
#     ax.set_ylabel('Velocity change '+ylabels[i])
#     # plt.xscale('log')
#     # plt.yscale('log')
#     plt_nr += 1
# sns.despine(trim=False, offset=3)
# plt.tight_layout()
# fig.savefig(os.path.join(fig_dir, 'velocity_session_wise.pdf'))

# ################################################################################################################################################################################################################






# probs = df.groupby(['walk_transition']).count()['ones']
# probs.iloc[0:2] = probs.iloc[0:2] / probs.iloc[0:2].sum()
# probs.iloc[2:4] = probs.iloc[2:4] / probs.iloc[2:4].sum()




# #  / df['ones'].sum())
# df_meta.loc[df_meta['walk_transition']==0, 'walk_1'] = df_meta.loc[df_meta['walk_transition']==1, 'walk_1'] - probs.iloc[0] 
# df_meta.loc[df_meta['walk_transition']==1, 'walk_1'] = df_meta.loc[df_meta['walk_transition']==1, 'walk_1'] - probs.iloc[1] 
# df_meta.loc[df_meta['walk_transition']==2, 'walk_1'] = df_meta.loc[df_meta['walk_transition']==1, 'walk_1'] - probs.iloc[2] 
# df_meta.loc[df_meta['walk_transition']==3, 'walk_1'] = df_meta.loc[df_meta['walk_transition']==3, 'walk_1'] - probs.iloc[3] 


# # # plot #FIXME:
# # d = (df_meta.loc[ind_clean_w,:].groupby(['subj_idx', 'session', 'walk_transition']).count()['ones'] / df_meta.loc[ind_clean_w,:].groupby(['subj_idx', 'session'])['ones'].sum())
# # probabilities = d.groupby('walk_transition').mean()
# # sems = d.groupby('walk_transition').sem()
# # plt.figure()
# # plt.bar([0,1,2,3],probabilities, yerr=sems)
# # plt.ylim(0,1)

# for subj, df in df_meta.loc[ind_clean_w&ind_u,:].groupby(['subj_idx',]):
#     probabilities = df.groupby('walk_transition').count()['ones'] / df.groupby('walk_transition').count()['ones'].sum()
#     plt.figure()
#     plt.bar(probabilities.index, probabilities)
#     plt.xlim(0,4)
#     plt.ylim(0,1)

# # correct 






# b = df.groupby(['walk_transition']).mean()['velocity']






# d = (df.groupby(['subj_idx', 'session', 'walk_transition']).count()['ones'] / df.groupby(['subj_idx', 'session'])['ones'].sum())
# probabilities = d.groupby('walk_transition').mean()
# sems = d.groupby('walk_transition').sem()
# plt.figure()
# plt.bar([0,1,2,3],probabilities, yerr=sems)
# plt.ylim(0,1)





# df = df.loc[df['walk_transition']==3,:]

# #FIXME: 
# # walking-walking --> model
# # still-walking --> average
# # walking-still --> average
# # still-still --> average

# # bins:
# bins = np.array([-10,-0.005,0.005,0.25,0.5,0.75,1,1.25,10])
# df_meta['bins_velocity'] = pd.cut(df_meta['velocity_0'], bins=bins, labels=False)
# df['bins_velocity'] = pd.cut(df['velocity_b'], bins=bins, labels=False)

# # bin count:
# velocity_counts = df.groupby(['bins_velocity']).count().reset_index()
# df['bins_velocity_count'] = 0
# for b in df['bins_velocity'].unique():
#     df.loc[df['bins_velocity']==b, 'bins_velocity_count'] = int(velocity_counts.loc[velocity_counts['bins_velocity']==b, 'velocity'])

# # correct:
# popt, pcov = curve_fit(vns_analyses.qubic, df['velocity_b'], df['velocity'])
# df_meta['velocity'] = df_meta['velocity_1']
# df_meta['velocity_c'] = df_meta['velocity'] - vns_analyses.qubic(df_meta['velocity_0'], *popt)



# # plots:
# # ------
# imp.reload(vns_analyses)
# # # plot 1 -- time courses:
# # fig = vns_analyses.plot_pupil_responses(df_meta, epochs_p.loc[:, ::50], bins=bins, ylabel='Pupil response\n(% max)', ylim=(0, 1.2))
# # fig.savefig(os.path.join(fig_dir, 'pupil_responses_u_b.pdf'))

