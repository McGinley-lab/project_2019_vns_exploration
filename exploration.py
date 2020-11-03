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
from tools_mcginley import utils

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

def plot_leak_histogram(df, leak_cut_off):

    # fig = plt.figure(figsize=(9,3))
    # for i, cuff_type in enumerate(['intact', 'single', 'double']):
    #     ax = fig.add_subplot(1,3,i+1)
    #     ax.hist(df.loc[df['cuff_type']==cuff_type, :].groupby(['subj_idx', 'session']).mean()['leak'], bins=50)
    #     plt.axvline(leak_cut_off, ls='--', color='r')
    #     plt.title(cuff_type)
    #     plt.xlabel('Leak fraction')
    #     plt.ylabel('Sessions (#)')
    # sns.despine(trim=False, offset=3)
    # plt.tight_layout()
    # fig.savefig(os.path.join(fig_dir, 'leak_fractions.pdf'))

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(1,1,1)
    ax.hist(df.groupby(['subj_idx', 'session']).mean()['leak'], bins=25)
    plt.axvline(leak_cut_off, ls='--', color='r')
    plt.xlabel('Leak fraction')
    plt.ylabel('Sessions (#)')
    sns.despine(trim=False, offset=3)
    plt.tight_layout()

    return fig

def plot_leak_per_session(df):
    fig = plt.figure(figsize=(8,8))
    d = df.loc[(df_meta['width']>0.2),:].groupby(['subj_idx', 'date', 'amplitude_bin']).mean().reset_index()
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
    return fig

def plot_leak_fractions(df):
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

def plot_velocity_histogram(df, bins=30):

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(331)
    ax.hist(df.loc[ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)

    ax = fig.add_subplot(332)
    ax.hist(df.loc[ind_s&ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_s[ind_u&ind_clean_w])/np.sum(ind_u&ind_clean_w)*100,1)))

    ax = fig.add_subplot(333)
    ax.hist(df.loc[ind_w&ind_u, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_w[ind_u&ind_clean_w])/np.sum(ind_u&ind_clean_w)*100,1)))

    ax = fig.add_subplot(334)
    ax.hist(df.loc[ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)

    ax = fig.add_subplot(335)
    ax.hist(df.loc[ind_s&ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_s[ind_g&ind_clean_w])/np.sum(ind_g&ind_clean_w)*100,1)))

    ax = fig.add_subplot(336)
    ax.hist(df.loc[ind_w&ind_g, 'velocity'], bins=bins, density=False, histtype='stepfilled')
    plt.axvline(velocity_cutoff[0], color='r', ls='--', lw=0.5)
    plt.axvline(velocity_cutoff[1], color='r', ls='--', lw=0.5)
    # plt.ylim(0,800)
    ax.set_title('{}%'.format(round(np.sum(ind_w[ind_g&ind_clean_w])/np.sum(ind_g&ind_clean_w)*100,1)))
    plt.tight_layout()
    sns.despine(trim=False, offset=3)

    return fig

def plot_baseline_dependence(df):

    popt = vns_analyses.fit_log_logistic(df, 'pupil_c', resample=False)
    df['pupil_predicted'] = vns_analyses.log_logistic_3d(np.array(df[['charge', 'rate']]), *popt)
    df['pupil_residuals'] = (df['pupil_c']-df['pupil_predicted']) + df['pupil_c'].mean()

    # plot 1 -- baseline dependence:
    fig = plt.figure(figsize=(6,2))
    for i, measure in enumerate(['pupil', 'pupil_c', 'pupil_residuals']):
        means = df.groupby('bins_pupil')[['pupil_c', 'pupil', 'pupil_residuals', 'pupil_0']].mean()
        sems = df.groupby('bins_pupil')[['pupil_c', 'pupil', 'pupil_residuals', 'pupil_0']].sem()
        ax = fig.add_subplot(1,3,i+1)
        plt.errorbar(means['pupil_0'], means[measure], yerr=sems[measure], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
        x = np.linspace(min(means['pupil_0']),max(means['pupil_0']),100)
        popt,pcov = curve_fit(vns_analyses.gaus, means['pupil_0'],means[measure],)
        plt.plot(x, vns_analyses.gaus(x,*popt), '--', color='r')
        plt.xlabel('Baseline pupil')
        plt.ylabel('Pupil response')
        plt.title(measure)
    plt.tight_layout()
    sns.despine(trim=False, offset=3)
    return fig

def make_scalars(df):

    timewindows = {

                'velocity_-3' : [(-30.1,-30), (-40.1,-40)],
                'velocity_-2' : [(-20.1,-20), (-30.1,-30)],
                'velocity_-1' : [(-10.1,-10), (-20.1,-20)],
                'velocity_0' : [(-0.1,0), (-10.1,-10)],
                'velocity_1' : [(9.9,10), (-0.1,0)],

                'slope_-3' : [(-12, -9), (None, None)],      #
                'slope_-2' : [(-9, -6), (None, None)],      #
                'slope_-1' : [(-6, -3), (None, None)],      #
                'slope_0' : [(-3, 0), (None, None)],          #
                'slope_1' : [(0, 3), (None, None)],

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
                'slope' : epochs_s,
                'eyelid' : epochs_l,
                'blink' : epochs_b,
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
            df[key] = resp
        else:
            baseline = epochs[key.split('_')[0]].loc[:,(x>=window2[0])&(x<=window2[1])].mean(axis=1)
            df[key] = resp-baseline    
        # if key == 'blink':
        #     df['{}_resp_{}'.format(key, i)] = (epochs[key].loc[:,(x>=window[0])&(x<=window[1])].mean(axis=1) > 0).astype(int)
    df['walk_-3'] = ((df['velocity_-3'] < velocity_cutoff[0])|(df['velocity_-3'] > velocity_cutoff[1])).astype(int)
    df['walk_-2'] = ((df['velocity_-2'] < velocity_cutoff[0])|(df['velocity_-2'] > velocity_cutoff[1])).astype(int)
    df['walk_-1'] = ((df['velocity_-1'] < velocity_cutoff[0])|(df['velocity_-1'] > velocity_cutoff[1])).astype(int)
    df['walk_0'] = ((df['velocity_0'] < velocity_cutoff[0])|(df['velocity_0'] > velocity_cutoff[1])).astype(int)
    df['walk_1'] = ((df['velocity_1'] < velocity_cutoff[0])|(df['velocity_1'] > velocity_cutoff[1])).astype(int)

    # add offset responses:
    x = np.array(epochs_p.columns, dtype=float)
    df['offset'] = epochs_p.loc[:,(x>10)&(x<12)].mean(axis=1) - epochs_p.loc[:,(x>8)&(x<10)].mean(axis=1)

    # # add pupil slopes:
    # x = np.array(epochs_p.columns, dtype=float)
    # # plt.plot(x, epochs_p.loc[ind_u,:].mean(axis=0)-epochs_p.loc[ind_u,:].mean(axis=0).mean())
    # # plt.plot(x, epochs_p.loc[ind_u,:].diff(axis=1).mean(axis=0)*50)
    # df_meta['pupil_s'] = np.array(epochs_p.diff(axis=1).loc[:,(x>0)&(x<5)].max(axis=1)) * 50

    return df, timewindows

def compute_eyemovent_trf(epochs, fs=50, window_len=1):
    
    from scipy import signal

    Sxx = []
    for i in range(epochs.shape[0]):
        f, t, S = signal.spectrogram(np.array(epochs.iloc[i]), fs=fs, nperseg=fs*window_len, noverlap=window_len/2)
        S = pd.DataFrame(S, columns=t+epochs.columns[0])
        S['freq'] = f
        S['trial'] = i
        S = S.set_index(['trial', 'freq'])
        Sxx.append(S)
    Sxx = pd.concat(Sxx)
    return Sxx

def plot_eyemovent_power(S, df_meta, ind_u, ind_g):

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['lightgrey', 'yellow', 'red'], N=100)

    # plot:
    fig = plt.figure(figsize=(5,3))
    plt_nr = 1
    for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
        for cuff_type in ['intact', 'single', 'double']:
            cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind

            SS = S.loc[S.index.get_level_values('trial').isin(df_meta.loc[cuff_ind,:].index),:].groupby('freq').mean()
            ind_b = (SS.columns>=-5.5)&(SS.columns<=-0.5)
            SS.loc[:,:] = (SS.loc[:,:]-np.atleast_2d(SS.loc[:,ind_b].mean(axis=1)).T) / np.atleast_2d(SS.loc[:,ind_b].mean(axis=1)).T 
            
            ax = fig.add_subplot(2,3,plt_nr)
            ind_p = (SS.columns>=-2)&(SS.columns<=15)

            pc = plt.pcolormesh(SS.columns[ind_p], SS.index.get_level_values('freq') , SS.loc[:,ind_p], cmap=cmap, vmin=0, vmax=10)
            fig.colorbar(pc)
            plt.axvline(0, color='k', lw=1, ls='--')
            plt.axvline(10, color='k', lw=1, ls='--')
            plt.title(cuff_type)
            plt_nr += 1

    # fig2 = plt.figure(figsize=(5,3))
    # plt_nr = 1
    # colors = sns.dark_palette("red", 5)
    # for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
    #     for cuff_type in ['intact', 'single', 'double']:
    #         cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind
    #         ax = fig2.add_subplot(2,3,plt_nr)
    #         plt.title(cuff_type)
    #         for cb in [0,1,2,3,4]:
    #             SS = np.nanmean(S[cuff_ind & (df_meta['charge_bin']==cb)], axis=0)
    #             SS = (SS-np.atleast_2d(SS[:,(t>=-5.5)&(t<=-0.5)].mean(axis=1)).T) / np.atleast_2d(SS[:,(t>=-5.5)&(t<=-0.5)].mean(axis=1)).T 
    #             plt.plot(SS[:,(t>=0)&(t<=10)].mean(axis=1), color=colors[cb])
    #         plt_nr += 1
    #         plt.ylim(0,200)

    return fig

def print_sample_sizes(df):

    ses_total = 0
    for cuff_type in ['intact', 'single', 'double']:
        a = df.loc[ind_g&(df['cuff_type']==cuff_type),:].groupby(['subj_idx',]).count().shape[0]
        b = df.loc[ind_g&(df['cuff_type']==cuff_type),:].groupby(['subj_idx', 'session']).count().shape[0]
        c = df.loc[ind_u&(df['cuff_type']==cuff_type),:].groupby(['subj_idx']).count().shape[0]
        d = df.loc[ind_u&(df['cuff_type']==cuff_type),:].groupby(['subj_idx', 'session']).count().shape[0]
        ses_total =  ses_total + b + d
        print()
        print(cuff_type)
        print('grounded subjects: {}'.format(a))
        print('grounded sessions: {}'.format(b))
        print('ungrounded subjects: {}'.format(c))
        print('ungrounded sessions: {}'.format(d))
    
    print()
    print()
    print('total subjects: {}'.format(len(df['subj_idx'].unique())))
    print('total sessions: {}'.format(ses_total))
    
def print_isi(df):
    mins = []
    medians = []
    maxs = []
    for (subj, ses), d in df.groupby(['subj_idx', 'session']):
        mins.append(d['time'].diff().max())
        medians.append(d['time'].diff().median())
        maxs.append(d['time'].diff().min())
    print()
    print(min(mins))
    print(np.median(medians))
    print(max(maxs))

def print_log_logistic_params(df):

    for measure in ['pupil_c', 'eyelid_c', 'velocity', 'walk',]:
        for grounded in [ind_u, ind_g]:
            for cuff_type in ['intact', 'single', 'double']:
                
                ind = grounded & (df['cuff_type'] == cuff_type)
                ind = ind & ~np.isnan(df[measure])

                if ('velocity' in measure):
                    ind = ind & ind_w & ind_clean_w
                if ('walk' in measure):
                    ind = ind & ind_clean_w

                print()
                print(measure)
                if (grounded == ind_u).mean()==1:
                    print('ungrounded')
                else:
                    print('grounded')
                print(cuff_type)
                vns_analyses.fit_log_logistic(df.loc[ind, :], measure)

def half_max_analyses(df, fig_dir):

    nboot = 5000
    n_jobs = 48
    for comparison in [0,1,2,3,4]:
        if comparison == 0:
            ind1 = ind_u & (df['cuff_type'] == 'double') & ~np.isnan(df['pupil_c'])
            ind2 = ind_g & (df['cuff_type'] == 'double') & ~np.isnan(df['pupil_c'])
            res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df.loc[ind1,:], 'pupil_c', resample=True) for _ in range(nboot)))
            res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df.loc[ind2,:], 'pupil_c', resample=True) for _ in range(nboot)))
        if comparison == 1:
            ind = ind_u & (df_meta['cuff_type'] == 'intact') & ind_clean_w & ~np.isnan(df_meta['pupil_c']) & ~np.isnan(df_meta['walk_c'])
            res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'pupil_c', resample=True) for _ in range(nboot)))
            res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'walk_c', resample=True) for _ in range(nboot)))
        if comparison == 2:
            ind = ind_u & (df_meta['cuff_type'] == 'intact') & ind_clean_w & ~np.isnan(df_meta['pupil_c']) & ~np.isnan(df_meta['walk_c']) & ind_w
            res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'pupil_c', resample=True) for _ in range(nboot)))
            res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'velocity_c', resample=True) for _ in range(nboot)))
        if comparison == 3:
            ind = ind_u & (df_meta['cuff_type'] == 'intact') & ~np.isnan(df_meta['pupil_c']) & ~np.isnan(df_meta['eyelid_c'])
            res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'pupil_c', resample=True) for _ in range(nboot)))
            res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind,:], 'eyelid_c', resample=True) for _ in range(nboot)))
        if comparison == 4:
            ind1 = ind_u & (df_meta['cuff_type'] == 'intact') & ~np.isnan(df_meta['pupil_c'])
            ind2 = ind_u & (df_meta['cuff_type'] == 'double') & ~np.isnan(df_meta['pupil_c']) 
            res1 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind1,:], 'pupil_c', resample=True) for _ in range(nboot)))
            res2 = np.array(Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(vns_analyses.fit_log_logistic)(df_meta.loc[ind2,:], 'pupil_c', resample=True) for _ in range(nboot)))

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
            fig.savefig(os.path.join(fig_dir, 'halfmax_{}_{}.pdf'.format(m, comparison)))

# variables:
raw_dir = '/media/external1/raw/vns_exploration/data/'
process_dir = '/media/external1/projects/vns_exploration/preprocess/'
project_dir = '/home/jwdegee/vns_exploration'
fig_dir = os.path.join(project_dir, 'figures', 'exploration')
data_dir = os.path.join(project_dir, 'data', 'exploration')
df_stim = pd.read_excel(os.path.join(raw_dir, 'stim_summary.xlsx'), sheet_name='python')

# cutoffs:
leak_cut_off = 0.25
velocity_cutoff = (-0.005, 0.005)
blink_cutoff = 0.25

# tasks:
tasks = []
counter = 0
# for cuff_type in ['intact',]:
for cuff_type in ['intact', 'single', 'double']:
    subjects = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, '*'))]
    for subj in subjects:
        sessions = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, cuff_type, subj, '*'))]
        for ses in sessions:
            file_tdms = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*.tdms'))
            file_meta = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*preprocessed.mat'))
            file_pupil = os.path.join(process_dir, subj, ses, '{}_{}_df_pupil_preprocessed.hdf'.format(subj, ses))
            # file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*df_pupil*.hdf'))
            # file_frames = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*frame*.txt'))
            # if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1)&(len(file_frames)==1):
            if (len(file_meta)==1)&(len(file_tdms)==1)&os.path.exists(file_pupil):
            # if (len(file_meta)==1)&(len(file_tdms)==1)&(len(file_pupil)==1):
                tasks.append((file_meta[0], file_pupil, file_tdms[0], fig_dir, subj, cuff_type, df_stim))
            counter += 1
            # if (subj == 'V5N1')&(ses=='4a'):
            #     file_tdms = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*.tdms'))
            #     file_meta = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, '*preprocessed.mat'))
            #     file_pupil = glob.glob(os.path.join(raw_dir, cuff_type, subj, ses, 'pupil_preprocess', '*.hdf'))
            #     if (len(file_meta)==1)&(len(file_pupil)==1)&(len(file_tdms)==1):
            #         tasks.append((file_meta[0], file_pupil[0], file_tdms[0], fig_dir, subj, cuff_type, df_stim))

preprocess = 0
if preprocess:
    n_jobs = 32
    # n_jobs = 1
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(analyse_exploration_session)(*task) for task in tasks)

    # sort:
    df_meta = pd.concat([res[i][0] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_v = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_l = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_x = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_y = pd.concat([res[i][6] for i in range(len(res))], axis=0).reset_index(drop=True)

    # save:
    epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    epochs_b.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
    epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
    epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
    df_meta.to_csv(os.path.join(data_dir, 'meta_data.csv'))
    
# load:
print('loading data')
epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil') * 100
epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid') * 100
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
print('finished loading data')

# pupil slope:
epochs_s = epochs_p.diff(axis=1) * 50

# round:
df_meta['date'] = pd.to_datetime(df_meta.date, format='%Y/%m/%d', errors='coerce')
df_meta['amplitude'] = np.round(df_meta['amplitude'], 3)
df_meta['width'] = np.round(df_meta['width'], 3)
df_meta['rate'] = np.round(df_meta['rate'], 3)
df_meta['charge'] = np.round(df_meta['charge'], 3)
df_meta['charge_ps'] = np.round(df_meta['charge_ps'], 3)

# make scalars:
df_meta, timewindows = make_scalars(df_meta)

# remove blink trials:
df_meta['blink'] = ( (df_meta['blink_0']>blink_cutoff) | (df_meta['blink_1']>blink_cutoff) | np.isnan(df_meta['pupil_0']) | np.isnan(df_meta['pupil_1']) )
epochs_v = epochs_v.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_p = epochs_p.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_s = epochs_s.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_l = epochs_l.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_b = epochs_b.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_x = epochs_x.loc[df_meta['blink']==0,:].reset_index(drop=True)
epochs_y = epochs_y.loc[df_meta['blink']==0,:].reset_index(drop=True)
df_meta = df_meta.loc[df_meta['blink']==0,:].reset_index(drop=True)

# indices:
ind_clean_w = ~(np.isnan(df_meta['velocity_1'])|np.isnan(df_meta['velocity_0']))
ind_g = (df_meta['date']<'2018-10-14')
ind_u = (df_meta['date']>'2018-10-14')
ind_g.loc[(df_meta['leak']<leak_cut_off)] = False
# ind_g = df_meta['date']<'2018-10-14'
ind_s = ((df_meta['velocity_1'] >= velocity_cutoff[0]) & (df_meta['velocity_1'] <= velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])
ind_w = ((df_meta['velocity_1'] < velocity_cutoff[0]) | (df_meta['velocity_1'] > velocity_cutoff[1])) & ~np.isnan(df_meta['velocity_1'])

# print sample size:
print_sample_sizes(df_meta)

# print inter stimulation interval:
print_isi(df_meta)

cuff_ind = np.array(df_meta['cuff_type'] == 'intact') 
plt.figure()
plt.plot(epochs_p.loc[cuff_ind&ind_u,:].mean(axis=0))

# correct scalars:
for group in [ind_u, ind_g]:
    df_meta, figs = vns_analyses.correct_scalars(df_meta, group=group, velocity_cutoff=velocity_cutoff, ind_clean_w=ind_clean_w)
    figs[0].savefig(os.path.join(fig_dir,  'pupil_reversion_to_mean1.pdf'))
    figs[1].savefig(os.path.join(fig_dir,  'pupil_reversion_to_mean2.pdf'))
    figs[2].savefig(os.path.join(fig_dir,  'slope_reversion_to_mean1.pdf'))
    figs[3].savefig(os.path.join(fig_dir,  'slope_reversion_to_mean2.pdf'))
    figs[4].savefig(os.path.join(fig_dir,  'eyelid_reversion_to_mean1.pdf'))
    figs[5].savefig(os.path.join(fig_dir,  'eyelid_reversion_to_mean2.pdf'))
    figs[6].savefig(os.path.join(fig_dir,  'velocity_reversion_to_mean1.pdf'))
    figs[7].savefig(os.path.join(fig_dir,  'velocity_reversion_to_mean2.pdf'))
    figs[8].savefig(os.path.join(fig_dir,  'velocity_reversion_to_mean3.pdf'))
    figs[9].savefig(os.path.join(fig_dir,  'velocity_reversion_to_mean4.pdf'))
    figs[10].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean1.pdf'))
    figs[11].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean2.pdf'))
    figs[12].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean3.pdf'))
    figs[13].savefig(os.path.join(fig_dir, 'walk_reversion_to_mean4.pdf'))

# subtract baselines:
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

# dictionairies:
epochs = {
    'velocity' : epochs_v,
    'pupil' : epochs_p,
    'slope' : epochs_s,
    'eyelid' : epochs_l,
    'blink' : epochs_b,
    }

ylims = {
            'velocity' : (-0.1, 0.8),
            'walk' : (0,0.75),
            'pupil' : (-5, 20),
            'slope' : (-5, 20),
            'eyelid' : (-1, 3),
            'blink' : (0, 0.3),
            }

# plot charges:
fig = vns_analyses.plot_pupil_responses_matrix_()
fig.savefig(os.path.join(fig_dir, 'charges.pdf'))

# plot eyemovement power:
S_x = compute_eyemovent_trf(epochs_x, fs=50, window_len=1)
S_y = compute_eyemovent_trf(epochs_y, fs=50, window_len=1)
for S, title in zip([S_x, S_y], ['x', 'y']):
    fig = plot_eyemovent_power(S, df_meta, ind_u, ind_g)
    fig.savefig(os.path.join(fig_dir, 'eyemovement', '{}_trf.pdf'.format(title)))
    # fig2.savefig(os.path.join(fig_dir, 'eyemovement', '{}_power.pdf'.format(title)))

for S, m in zip([S_x, S_y], ['x', 'y']):
    ind_b = (S.columns>=-5.5)&(S.columns<=-0.5)
    ind_s = (S_x.columns>=0) & (S_x.columns<=10)
    for f in [5,10,20]:
        df_meta.loc[df_meta['rate']==f, 'eyemovement_{}_1'.format(m)] = np.array(S.loc[S.index.get_level_values('trial').isin(df_meta.loc[df_meta['rate']==f,:].index) & 
                                                                                        (S.index.get_level_values('freq')==f),ind_s].mean(axis=1))
        df_meta.loc[df_meta['rate']==f, 'eyemovement_{}_0'.format(m)] = np.array(S.loc[S.index.get_level_values('trial').isin(df_meta.loc[df_meta['rate']==f,:].index) & 
                                                                                        (S.index.get_level_values('freq')==f),ind_b].mean(axis=1))
    # baseline per condition:
    for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
        for cuff_type in ['intact', 'single', 'double']:
            cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind
            df_meta.loc[cuff_ind, 'eyemovement_{}_1'.format(m)] = (df_meta.loc[cuff_ind, 'eyemovement_{}_1'.format(m)] - df_meta.loc[cuff_ind, 'eyemovement_{}_0'.format(m)].mean()) / df_meta.loc[cuff_ind, 'eyemovement_{}_0'.format(m)].mean()

    # plot:
    for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
        for cuff_type in ['intact', 'single', 'double']:
            cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind
            fig = vns_analyses.plot_scalars2(df_meta.loc[cuff_ind,:], measure='eyemovement_{}_1'.format(m), ylabel=None, ylim=(0,75))
            fig.savefig(os.path.join(fig_dir, 'eyemovement', title, 'scalars2_{}_{}_{}.pdf'.format(title, cuff_type, 'eyemovement_{}_1'.format(m))))

# save source data:
for cuff in ['intact', 'single', 'double']:
    for grounded_ind, grounded in zip([ind_u, ind_g], ['u', 'g']):
        for measure in ['pupil_c', 'eyelid_c', 'walk_c', 'velocity_c', 'walk_1', 'velocity_1', 'eyemovement_x_1', 'eyemovement_y_1']:
            ind = grounded_ind&(df_meta['cuff_type']==cuff)
            if ('walk' in measure) | ('velocity' in measure):
                ind = ind & ind_clean_w
            df = df_meta.loc[ind,['subj_idx', 'session', 'amplitude', 'width', 'rate', 'charge', 'charge_bin', measure]].reset_index(drop=True)
            if 'eyemovement' in measure:
                df = df.rename({measure:measure.split('_')[0]+'_'+measure.split('_')[1]}, axis=1)
                df.to_csv(os.path.join(data_dir, 'exploration_source_data_{}_{}_{}.csv'.format(grounded, cuff, measure.split('_')[0]+'_'+measure.split('_')[1])))
            else:
                df = df.rename({measure:measure.split('_')[0]}, axis=1)
                df.to_csv(os.path.join(data_dir, 'exploration_source_data_{}_{}_{}.csv'.format(grounded, cuff, measure.split('_')[0])))



# plot leaks:
fig = plot_leak_histogram(df_meta, leak_cut_off=leak_cut_off)
fig.savefig(os.path.join(fig_dir, 'leak_fractions.pdf'))
fig = plot_leak_per_session(df_meta)
fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_time.pdf'))
for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
    fig = plot_leak_fractions(df_meta.loc[ind,:])
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_{}.pdf'.format(title)))
    for cuff_type in ['intact', 'single', 'double']:
        cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind
        fig = plot_leak_fractions(df_meta.loc[cuff_ind,:])
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'leak_fractions_{}_{}.pdf'.format(title, cuff_type,)))

# plot param preprocessing:
for ind, title in zip([ind_u, ind_g,], ['u', 'g',]):
    fig = vns_analyses.plot_param_preprocessing(df_meta.loc[ind,:])
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'average_{}.pdf'.format(title)))
    for cuff_type in ['intact', 'single', 'double']:
        cuff_ind = np.array(df_meta['cuff_type'] == cuff_type) & ind
        fig = vns_analyses.plot_param_preprocessing(df_meta.loc[cuff_ind,:])
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'average_{}_{}.pdf'.format(title, cuff_type,)))
        fig = plt.figure()
        sns.regplot(df_meta.loc[cuff_ind,'amplitude'], df_meta.loc[cuff_ind,'amplitude_m'])
        plt.title('{}_{}'.format(title, cuff_type,))
        fig.savefig(os.path.join(fig_dir, 'preprocess', 'amplitude_scatters_{}_{}.pdf'.format(title, cuff_type,)))

# print print log logistic params:
print_log_logistic_params(df_meta)

# velocity histogram:
fig = plot_velocity_histogram(df_meta)
fig.savefig(os.path.join(fig_dir, 'velocity_hist.pdf'))

# half max analyses:
half_max_analyses(df_meta, fig_dir)

# plot pupil responses:
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

# state dependence:
df = df_meta.loc[ind_u & (df_meta['cuff_type'] == 'intact'),:].reset_index()
popt = vns_analyses.fit_log_logistic(df, 'pupil_c', resample=False)
df['pupil_predicted'] = vns_analyses.log_logistic_3d(np.array(df[['charge', 'rate']]), *popt)
df['pupil_residuals'] = (df['pupil_c']-df['pupil_predicted']) + df['pupil_c'].mean()
fig = vns_analyses.plot_scalars2(df, measure='pupil_residuals', ylabel='pupil', ylim=ylims['pupil'], p0=False)
fig.savefig(os.path.join(fig_dir, 'pupil', 'scalars2_{}_{}.pdf'.format('pupil_residuals', 2)))
for X, bin_measure in zip(['pupil_0',], ['bins_pupil']):
    for Y in ['pupil_0', 'pupil_c', 'pupil_residuals']:
        if not X == Y:
            # plot:
            fig = vns_analyses.plot_correlation(df, X=X, Y=Y, bin_measure=bin_measure, scatter=True)
            fig.savefig(os.path.join(fig_dir, 'pupil', 'state_dependence_{}_{}.pdf'.format(X,Y)))

# baseline dependence:
fig = plot_baseline_dependence(df_meta.loc[ind_u])
fig.savefig(os.path.join(fig_dir, 'pupil_state_dependence.pdf'))

# all remaining plots:
run_all = 1
if run_all:
    imp.reload(vns_analyses)

    # for measure in ['velocity']:
    #     for cuff_type in ['intact']:
    
    for measure in ['pupil', 'eyelid', 'velocity', 'walk']:
        for cuff_type in ['intact', 'single', 'double']:

            cuff_ind = np.array(df_meta['cuff_type'] == cuff_type)
            for ind, title in zip([ind_u, ind_g], ['u', 'g']):
                if not os.path.exists(os.path.join(fig_dir, measure, title)):
                    os.makedirs(os.path.join(fig_dir, measure, title))
                if ('velocity' in measure):
                    ind = ind & ind_w & ind_clean_w
                    # ind = ind & ind_clean_w
                if ('walk' in measure):
                    ind = ind & ind_clean_w
                ind = ind & cuff_ind
                try:
                    ylim = ylims[measure]
                except:
                    ylim = ylims['pupil']
                
                if (measure == 'blink'):
                    ylim = (ylim[0], ylim[1]/3)
                if (measure == 'pupil') & (title=='g'):
                    ylim = (ylim[0], ylim[1]*2)

                # ylim = (ylim[0], ylim[1]/2)

                if not ((measure == 'walk') or (measure == 'offset')):
                    x = np.array(epochs[measure].columns, dtype=float)
                    fig = vns_analyses.plot_timecourses(df_meta.loc[ind, :], epochs[measure].loc[:,(x>=-20)&(x<=40)].loc[ind,::10], timewindows=timewindows, ylabel=measure+'_1', ylim=ylim)
                    fig.savefig(os.path.join(fig_dir, measure, title, 'timecourses_{}_{}_{}.pdf'.format(title, cuff_type, measure)))
                if measure == 'offset':
                    measure_exts = ['',]
                else:
                    measure_exts = ['', '_c', '_c2']
                for measure_ext in measure_exts:
                    fig = vns_analyses.plot_scalars(df_meta.loc[ind &  ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                    fig.savefig(os.path.join(fig_dir, measure, title, 'scalars_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                    fig = vns_analyses.plot_scalars2(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                    fig.savefig(os.path.join(fig_dir, measure, title, 'scalars2_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                    fig = vns_analyses.plot_scalars3(df_meta.loc[ind & ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext, ylabel=measure, ylim=ylim)
                    fig.savefig(os.path.join(fig_dir, measure, title, 'scalars3_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))

                    fig1, fig2, fig3 = vns_analyses.plot_swarms(df_meta.loc[ind &  ~np.isnan(df_meta[measure+measure_ext]), :], measure=measure+measure_ext)
                    fig1.savefig(os.path.join(fig_dir, measure, title, 'swarms_amp_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                    fig2.savefig(os.path.join(fig_dir, measure, title, 'swarms_width_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))
                    fig3.savefig(os.path.join(fig_dir, measure, title, 'swarms_zone_{}_{}_{}.pdf'.format(title, cuff_type, measure+measure_ext)))

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