import os, sys
import glob
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import h5py
from joblib import Parallel, delayed

from IPython import embed as shell

from tools_mcginley import preprocess_pupil
from tools_mcginley import utils

import vns_analyses
from vns_analyses import analyse_baseline_session

raw_dir = '/media/external2/raw2/vns_baseline_pupil/data/'
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
            file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*.hdf'))
            if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1):
                tasks.append((file_meta[0], file_pupil[0], file_tdms[0], fig_dir, subj, cuff_type,))

preprocess = False
if preprocess:
    n_jobs = 12
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(analyse_baseline_session)(*task) for task in tasks)

    # sort:
    df_meta = pd.concat([res[i][0] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_x = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_y = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][6] for i in range(len(res))], axis=0).reset_index(drop=True)

    # save:
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='distance')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='l')
    epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x')
    epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y')
    epochs_b.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='b')
    df_meta.to_csv(os.path.join(data_dir, 'meta_data.csv'))

# load:
print('loading data')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='distance')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='l')
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='x')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='y')
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='b')
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
print('finished loading data')

# cutoffs:
distance_cutoff = (-0.005, 0.005)
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

            'distance_-3' : [(-30.1,-30), (-40.1,-40)],
            'distance_-2' : [(-20.1,-20), (-30.1,-30)],
            'distance_-1' : [(-10.1,-10), (-20.1,-20)],
            'distance_0' : [(-0.1,0), (-10.1,-10)],
            'distance_1' : [(9.9,10), (-0.1,0)],

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
            'distance' : epochs_v,
            'pupil' : epochs_p,
            'eyelid' : epochs_l,
            'blink' : epochs_b,
            # 'eyemovement' : epochs_xy,
            }

ylims = {
            'distance' : (-0.1, 0.5),
            'walk' : (0,1),
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
        baseline = 0
    else:
        baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1)
    df_meta[key] = resp-baseline
    # if key == 'blink':
    #     df_meta['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)

# remove blink trials:
df_meta['blink'] = ((df_meta[['blink_-3', 'blink_-2', 'blink_-1', 'blink_0']].mean(axis=1)>blink_cutoff) | (df_meta['blink_1']>blink_cutoff))
epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

# make random df:
dfs = []
for i in [0,-1,-2]:
    df = df_meta.loc[:,['pupil_{}'.format(str(i)), 'pupil_{}'.format(str(i-1)), 'subj_idx', 'session']]
    df = df.rename(columns={'pupil_{}'.format(str(i)): 'pupil', 'pupil_{}'.format(str(i-1)): 'pupil_b'}).reset_index(drop=True)
    dfs.append(df)
df = pd.concat(dfs)
df['pupil_change'] = df['pupil'] - df['pupil_b']

# bins:
bins = np.array([-10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,10])
# bins = np.array([-10,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,10])
df_meta['bins_pupil'] = pd.cut(df_meta['pupil_0'], bins=bins, labels=False)
df['bins_pupil'] = pd.cut(df['pupil_b'], bins=bins, labels=False)

# # bin count:
# pupil_counts = df.groupby(['bins_pupil']).count().reset_index()
# df['bins_pupil_count'] = 0
# for b in df['bins_pupil'].unique():
#     df.loc[df['bins_pupil']==b, 'bins_pupil_count'] = int(pupil_counts.loc[pupil_counts['bins_pupil']==b, 'pupil'])

# function:
func = vns_analyses.qubic

# plot 1 -- reversion to mean:
fig1, fig2 = vns_analyses.plot_reversion_to_mean_correction(df, measure='pupil', bin_measure='pupil', func=func)
fig1.savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean1.pdf'))
fig2.savefig(os.path.join(fig_dir, 'pupil_reversion_to_mean2.pdf'))


df['pupil_b2'] = np.power(df['pupil_b'],2)
df['pupil_b3'] = np.power(df['pupil_b'],3)

from bambi import Model

# Assume we already have our data loaded
model = Model(df)
# results = model.fit('pupil_change ~ pupil_b + pupil_b2 + pupil_b3', random=['1+pupil_b+pupil_b2+pupil_b3|session', 'session|subj_idx'], samples=10, chains=4,)
results = model.fit('pupil_change ~ pupil_b + pupil_b2 + pupil_b3', random=['1+pupil_b+pupil_b2+pupil_b3|subj_idx',], samples=5000, chains=4,)
results[1000:].plot()
results[1000:].summary()

shell()

# correct:
# --------
df_meta['pupil'] = df_meta['pupil_1'] - df_meta['pupil_0']

# across subjects:
popt, pcov = curve_fit(func, df['pupil_b'], df['pupil_change'])
df_meta['pupil_c'] = df_meta['pupil'] - func(df_meta['pupil_0'], *popt)

# per subject:
popts = []
df_meta['pupil_c2'] = np.zeros(df_meta.shape[0])
for subj in df['subj_idx'].unique():
    ind = df['subj_idx'] == subj
    popt, pcov = curve_fit(func, df.loc[ind,'pupil_b'], df.loc[ind,'pupil_change'])
    popts.append(popt)
    ind = df_meta['subj_idx'] == subj
    df_meta.loc[ind,'pupil_c2'] = df_meta.loc[ind,'pupil'] - func(df_meta.loc[ind,'pupil_0'], *popt)

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

# plot 1 -- time courses:
fig = vns_analyses.plot_pupil_responses(df_meta, epochs_p.loc[:, ::50], bins=bins, ylabel='Pupil response\n(% max)', ylim=(0, 1.2))
fig.savefig(os.path.join(fig_dir, 'pupil_timecourses.pdf'))

# plot 1 -- baseline dependence:
fig = plt.figure(figsize=(6,2))
for i, measure in enumerate(['pupil', 'pupil_c', 'pupil_c2']):
    means = df_meta.groupby('bins_pupil')[['pupil_c2', 'pupil_c', 'pupil', 'pupil_0']].mean()
    sems = df_meta.groupby('bins_pupil')[['pupil_c2', 'pupil_c', 'pupil', 'pupil_0']].sem()
    ax = fig.add_subplot(1,3,i+1)
    plt.errorbar(means['pupil_0'], means[measure], yerr=sems[measure], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
    x = np.linspace(min(means['pupil_0']),max(means['pupil_0']),100)
    popt,pcov = curve_fit(vns_analyses.gaus, means['pupil_0'],means[measure],)
    plt.plot(x, vns_analyses.gaus(x,*popt), '-', color='r')
    plt.xlabel('Baseline pupil')
    plt.ylabel('Pupil response')
    plt.title(measure)
plt.tight_layout()
sns.despine(trim=False, offset=3)
fig.savefig(os.path.join(fig_dir, 'pupil_state_dependence.pdf'))


# sf = 500
# win = 15 * sf
# x = epochs_p.columns
# ind = (x>-60)&(x<0)
# freqs = []
# psds = []
# for i in range(epochs_p.shape[0]):
#     print(i)
#     data = np.array(epochs_p.loc[i,ind])
#     data = data - data.mean()
#     from scipy import signal
#     freq, psd = signal.welch(data, sf, nperseg=win)
#     # from scipy import fftpack
#     # X = fftpack.fft(data)
#     # freqs = fftpack.fftfreq(len(data)) * sf
#     freqs.append(freq)
#     psds.append(psd)
# df_psd = pd.DataFrame(psds, columns=freqs[0])
# means = df_psd.groupby([df_meta['subj_idx'], df_meta['session']]).mean()

# # plot:
# plt.figure()
# plt.axvspan(0.9, 1.4, color='grey', alpha=0.1)
# for i in range(means.shape[0]):
#     plt.plot(means.iloc[i], color='k', alpha=0.2)
# plt.plot(df_psd.mean(axis=0), color='red')
# # plt.xscale('log')
# plt.yscale('log')
# plt.xlim(0,50)

# # compute reversion to mean:
# func = vns_analyses.linear
# measure = 'pupil'
# popts = []
# for (subj, ses), d in df.groupby(['subj_idx', 'session']):
#     print((subj, ses))
#     popt, pcov = curve_fit(func, d['{}_b'.format(measure)], d['{}_change'.format(measure)],)
#     popts.append(popt)
# popts = np.vstack(popts)

# plt.figure()
# x = np.linspace(0,1,100)
# for p in popts:
#     plt.plot(x, func(x, *p))


# # compute power:
# from scipy.integrate import simps
# low = 0.1
# high = 0.5
# f = means.columns
# idx_band = np.logical_and(f >= low, f <= high)
# freq_res = np.diff(f)[0]
# bp = simps(means.loc[:,idx_band], dx=freq_res)

# r0 = []
# r1 = []
# for low in np.linspace(0.1, 10, 100):
#     high = low + 0.5
#     f = means.columns
#     idx_band = np.logical_and(f >= low, f <= high)
#     freq_res = np.diff(f)[0]
#     bp = simps(means.loc[:,idx_band], dx=freq_res)
#     r0.append(sp.stats.pearsonr(bp, popts[:,0])[0])
#     r1.append(sp.stats.pearsonr(bp, popts[:,1])[0])

# plt.figure()
# # plt.plot(np.linspace(0.1, 10, 100), r0)
# plt.plot(np.linspace(0.1, 10, 100), r1)



# low = 0.8
# high = low + 0.5
# f = means.columns
# idx_band = np.logical_and(f >= low, f <= high)
# freq_res = np.diff(f)[0]
# bp = simps(means.loc[:,idx_band], dx=freq_res)
# plt.figure()
# plt.scatter(bp, popts[:,1])





# epochs_p['mean'] = epochs_p.mean(axis=1)
# epochs_p['std'] = epochs_p.std(axis=1)
# averages = epochs_p.groupby([df_meta['subj_idx'], df_meta['session']]).mean()

# fig = plt.figure(figsize=(6,3))
# ax = fig.add_subplot(1,3,1)
# x = np.array(averages['mean'])
# y = popts[:,1]
# ax.scatter(x,y)
# # popt_s, pcov = curve_fit(vns_analyses.linear, x, y,)
# # ax.plot(np.linspace(0, max(x), 100), vns_analyses.linear(np.linspace(0, max(x), 100), *popt_s), color='k', ls='-', zorder=10)
# ax.set_xlabel('Pupil (mean)')
# ax.set_ylabel('Regression to mean')
# ax = fig.add_subplot(1,3,2)
# x = np.array(averages['std'])
# y = popts[:,1]
# ax.scatter(x,y)
# # popt_s, pcov = curve_fit(vns_analyses.linear, x, y,)
# # ax.plot(np.linspace(0, max(x), 100), vns_analyses.linear(np.linspace(0, max(x), 100), *popt_s), color='k', ls='-', zorder=10)
# ax.set_xlabel('Pupil (std)')
# ax.set_ylabel('Regression to mean')
# sns.despine(trim=False, offset=3)
# plt.tight_layout()

# shell()





# # Plot the power spectrum
# plt.figure(figsize=(8, 4))
# for f, p in zip(group_freq, group_power):
#     try:
#         plt.plot(f, p, color='grey')
#     except:
#         pass
# plt.plot(group_freq.mean(axis=0), np.nanmean(group_power, axis=0), color='red')
# plt.xscale('log')
# plt.yscale('log')
# # plt.xlim(0,50)





# r0 = []
# r1 = []

# for low in np.linspace(0.1, 10, 100):

#     high = low + 0.5
#     bp = []
#     for f,p in zip(group_freq, group_power):

#         # Frequency resolution
#         freq_res = f[1] - f[0]

#         # Find closest indices of band in frequency vector
#         idx_band = np.logical_and(f >= low, f <= high)

#         # Integral approximation of the spectrum using Simpson's rule.
#         bp.append(simps(p[idx_band], dx=freq_res))



# plt.figure()
# plt.plot(np.linspace(0.1, 10, 100), r0)

# plt.figure()
# plt.plot(np.linspace(0.1, 10, 100), r1)


# plt.figure()
# plt.scatter(bp, popts[:,0])

# plt.figure()
# plt.scatter(bp, popts[:,1])










# plt.plot(freqs, psd, color='k', lw=2)

# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power spectral density (V^2 / Hz)')
# # plt.ylim([0, psd.max() * 1.1])
# plt.title("Welch's periodogram")
# # plt.xlim([0, freqs.max()])
# plt.yscale('log')
# sns.despine()






# plt.plot(np.vstack(group_freq).mean(axis=0), np.vstack(group_power).mean(axis=0))
# plt.yscale('log')
# plt.xlim(0,50)



# popts = np.vstack(popts)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# for popt in popts:
#     x = np.linspace(0,1,100)
#     ax.plot(x, func(x, *popt), color='k', alpha=0.2)

# plt.figure()
# plt.hist(popts[:,0], bins=10)

# plt.figure()
# plt.hist(popts[:,1], bins=10)






# # for Matt:
# means = df.groupby(['subj_idx', 'session', 'bins']).mean().reset_index()
# means.to_csv(os.path.join(fig_dir, 'scalars_for_regression_to_mean.csv'))

# # individual sesisons:
# for subj, dd in df.groupby(['subj_idx',]):
#     means = dd.groupby('bins').mean()
#     sems = dd.groupby('bins').sem()
#     model = ols("pupil_change ~ 1 + pupil_b", data=means).fit()
#     df_pred = pd.DataFrame(np.linspace(0,1,100), columns=['pupil_b'])
#     predictions = model.predict(df_pred)
#     fig = plt.figure(figsize=(2.5,2.5))
#     plt.errorbar(means['pupil_b'], means['pupil']-means['pupil_b'], yerr=sems['pupil'], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
#     plt.axhline(0, ls='--')
#     # plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), '--')
#     plt.plot(np.linspace(0,1,100), predictions, '-', color='r')
#     plt.xlabel('Pupil size\n(t-10)')
#     plt.ylabel('Pupil change\n(t-10 --> t)')
#     plt.tight_layout()
#     sns.despine(trim=False, offset=3)