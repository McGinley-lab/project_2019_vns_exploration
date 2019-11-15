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
import statsmodels as sm
from statsmodels.formula.api import ols

from IPython import embed as shell

from tools_mcginley.preprocess_pupil import preprocess_pupil
from tools_mcginley.preprocess_calcium import preprocess_calcium 
from tools_mcginley import utils

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def func(x, a, b, c):
  #return a * np.exp(-b * x) + c
  return a * np.log(b * x) + c

def fsigmoid(x, a, b, s):
    return s / (1.0 + np.exp(-a*(x-b)))
    # x = np.linspace(-1, 1, 100) 
    # plt.plot(x, fsigmoid(x, 12, 0, 0.15)) 

def log_logistic(x, a, b, s):
    return s/(1+((x/a)**-b))

def log_logistic_3d(M, a1, b1, s1, a2, b2, s2):
    x = M[:,0]
    y = M[:,1]
    return (s1/(1+((x/a1)**-b1))) * (s2/(1+((y/a2)**-b2)))

def linear(x,a,b):
  return a + b*x 

def quadratic(x,a,b,c):
  return a + b*x + c*x**2  

def qubic(x,a,b,c,d):
  return a + b*x + c*x**2 + d*x**3  

def gaus(x,a,x0,sigma,b):
    from scipy import asarray as ar,exp
    return a*exp(-(x-x0)**2/(2*sigma**2)) + b*x

def hypersurface(df, z_measure):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.model_selection import KFold

    # data:
    means = df.groupby(['charge', 'rate']).mean().reset_index()
    X = means['charge']
    Y = means['rate']
    Z = means[z_measure]

    # fit:
    func = log_logistic_3d
    popt, pcov = curve_fit(func, np.array(df.loc[:,['charge', 'rate']]), 
                                np.array(df.loc[:,z_measure]),)

    # fitted surface:
    XX, YY = np.meshgrid(np.linspace(0,0.8,25), np.linspace(0,20,25))
    ZZ = np.zeros(XX.shape)
    for i in range(XX.shape[0]):
        for j in range(YY.shape[1]):
            ZZ[i,j] = func(np.vstack((XX[i,j],YY[i,j])).T, *popt)

    # plot:
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
    fig = plt.figure(figsize=(6,3))
    for i in range(2):
        ax = fig.add_subplot(1,2,i+1, projection='3d')
        ax.scatter(X, Y, Z, color='black', marker='o')
        ax.set_xlabel('Charge')
        ax.set_ylabel('Rate')
        ax.set_zlabel('Pupil')
        vmax = 0.30
        surf = ax.plot_surface(XX, YY, ZZ, cmap=cmap, vmin=-vmax, vmax=vmax,
                            linewidth=0, antialiased=False, alpha=0.5)
        ax.view_init(elev=15, azim=-145)
    ax.set_yticks([5, 10, 20])
    
    # variance explained:
    df['ne_p'] = np.NaN
    kf = KFold(n_splits=20, shuffle=True)
    fold_nr = 1
    for train_index, test_index in kf.split(df):
        print('fold {}'.format(fold_nr))
        # print("TRAIN:", train_index, "TEST:", test_index)
        popt, pcov = curve_fit(func, np.array(df[['charge', 'rate']].iloc[train_index]), np.array(df[z_measure].iloc[train_index]),)
        df.loc[test_index, 'ne_p'] = func(np.array(df.loc[test_index,['charge', 'rate']]) ,*popt)
        fold_nr += 1
    df['resid_ne_p'] = (df[z_measure] - df['ne_p'])**2
    df['resid_mean'] = (df[z_measure] - df[z_measure].mean())**2

    s0 = sum(df['resid_mean'])
    s1 = sum(df['resid_ne_p'])
    var_explained = (s1-s0) / s0 * 100
    
    print()
    print('{}% more variance explained!'.format(round(var_explained, 3)))

    return fig

def plot_param_preprocessing(df, popt, popts):

    X = np.array(df.groupby(['width']).mean().reset_index()['width'])
    y = np.array(df.groupby(['width']).mean().reset_index()['amplitude_m'])
    widths = np.unique(df['width'])

    fig = plt.figure(figsize=(9,12))
    gs = gridspec.GridSpec(4,3)

    ax = plt.subplot(gs[0,0])
    ax.scatter(df['width'], df['amplitude_m'],)
    rect = patches.Rectangle((widths[-2]-0.1, df['amplitude_m_mean'].mean()-0.05), widths[-1]-widths[-2]+0.2, 0.1, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.xticks(np.unique(df['width']), np.unique(df['width']))
    plt.xlim(0, 1)
    plt.xlabel('Pulse width')
    plt.ylabel('Pulse amplitude')
    
    ax = plt.subplot(gs[0,1])
    plt.bar([0,1], [df['amplitude_m_mean'].iloc[0], df['amplitude_i_min'].iloc[0],])
    plt.xticks([0,1], ['M', 'I'])
    plt.ylabel('Pulse amplitude\n(lowest bin; largest widths)')
    
    ax = plt.subplot(gs[0,2])
    plt.bar([0,1,2], [0, df['leak'].iloc[0], 0])
    plt.xticks([0,1,2], ['','L',''])
    plt.ylabel('Leak fraction')

    ax = plt.subplot(gs[1,0])
    ax.scatter(df['width'], df['amplitude_m'],)
    plt.xlim(0, 1)
    plt.xlabel('Pulse width')
    plt.ylabel('Pulse amplitude')
    x = np.linspace(X.min(), X.max(), 50)
    for _popt in popts:
        y_fit = fsigmoid(x, popt[0], popt[1], _popt)
        ax.plot(x, y_fit, color='orange')
    plt.xticks(np.unique(df['width']), np.unique(df['width']))

    ax = plt.subplot(gs[1,1])
    x = np.linspace(X.min(), X.max(), 50)
    y_fit = fsigmoid(x, *popt)
    ax.plot(X, y, 'o', label='data')
    ax.plot(x, y_fit, label='fit', color='orange')
    plt.xlabel('Pulse width')
    plt.ylabel('Pulse amplitude (collapsed)')
    plt.legend(loc='best')
    plt.xticks(np.unique(df['width']), np.unique(df['width']))

    ax = plt.subplot(gs[2,0])
    ax.scatter(df['width'], df['amplitude_m'], alpha=0.5)
    ax.scatter(df['width'], df['amplitude_c'], alpha=0.5, color='lightgreen')
    rect = patches.Rectangle((widths[0]-0.1, df.loc[df['amplitude_m_bin']==max(df['amplitude_m_bin']), 'amplitude_c'].mean()-0.05), widths[-1]-widths[0]+0.2, 0.1, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.xlim(0, 1)
    plt.xlabel('Pulse width')
    plt.ylabel('Pulse amplitude')
    plt.xticks(np.unique(df['width']), np.unique(df['width']))

    ax = plt.subplot(gs[2,1])
    plt.bar([0,1,], [df.loc[df['amplitude_m_bin']==max(df['amplitude_m_bin']), 'amplitude_c'].mean(), 
                        df['amplitude_i_max'].iloc[0]])
    plt.xticks([0,1], ['C', 'I'])
    plt.ylabel('Pulse amplitude\n(highest bin)')

    ax = plt.subplot(gs[2,2])
    plt.bar([0,1,2], [0, df['saturation'].iloc[0], 0])
    plt.xticks([0,1,2], ['','S',''])
    plt.ylabel('Saturation')

    ax = plt.subplot(gs[3,:])
    ax.plot(np.arange(60), df['amplitude_c_norm'], '-o')
    plt.xlabel('Stim #')
    plt.ylabel('Pulse amplitude\n(normalized)')
    plt.ylim(0,2)

    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

def catplot_nr_trials(df, ind1, ind2, charge=None):

    fig = plt.figure(figsize=(6,6))
    plot_nr = 1
    for rate in np.sort(np.unique(df['rate_bin'])):
        for cuff_type in ['intact', 'single', 'double']:
            ax = fig.add_subplot(3,3,plot_nr)
            colors = sns.dark_palette("red", 4)
            for width, color in zip(np.sort(df['width_bin'].unique()), colors):
                ind_1 = (df['cuff_type']==cuff_type)&(df['rate_bin']==rate)&(df['width_bin']==width)&ind1
                ind_2 = (df['cuff_type']==cuff_type)&(df['rate_bin']==rate)&(df['width_bin']==width)&ind2
                
                means = df.loc[ind_1,:].groupby(['amplitude_bin', 'width_bin']).mean().reset_index()
                fractions = df.loc[ind_2,:].groupby(['amplitude_bin', 'width_bin']).count().reset_index() / df.loc[ind_1,:].groupby(['amplitude_bin', 'width_bin']).count().reset_index()
                ax.errorbar(x=means['train_amp'], y=fractions['ones'], color=color, elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
            plt.xlabel('Amplitude')
            plt.ylabel('Fraction of trials')
            plt.ylim(0,1)
            plot_nr += 1
    plt.tight_layout()
    sns.despine(trim=False, offset=3)
    return fig

def plot_timecourses(df_stim, epochs, timewindows, ylabel='Pupil response', ylim=(None, None)):

    x = np.array(epochs.columns, dtype=float)
    fig = plt.figure(figsize=(10,2))
    plot_nr = 1
    for bin_by, param in zip(['amplitude_bin', 'width_bin', 'rate_bin', ['charge_bin', 'rate_bin'], ['amplitude_bin', 'width_bin', 'rate_bin']], 
                                ['amplitude', 'width', 'rate', 'charge', 'charge_ps']):
        
        # try:
        if isinstance(bin_by, list):
            for b in bin_by:
                epochs[b] = np.array(df_stim[b])
        else:
            epochs[bin_by] = np.array(df_stim[bin_by])
        epochs = epochs.set_index(bin_by, append=True, inplace=False)
        
        m = np.array(epochs.groupby(bin_by).mean())
        s = np.array(epochs.groupby(bin_by).sem())
        
        # # smooth:
        # fs = round(1/np.diff(epochs.columns)[0],1)
        # for i in range(m.shape[0]):
        #     m[i,:] = utils._butter_lowpass_filter(m[i,:], highcut=0.5, fs=fs, order=3)
        #     s[i,:] = utils._butter_lowpass_filter(s[i,:], highcut=0.5, fs=fs, order=3)
        
        ax = fig.add_subplot(1,5,plot_nr)
        colors = sns.color_palette("YlOrRd", m.shape[0])
        
        for span in timewindows[ylabel]:
            ax.axvspan(span[0], span[1], color='k', alpha=0.1)

        if 'charge' in param:
            sorting = np.argsort(df_stim.groupby(bin_by).mean()['charge_ps'].reset_index()['charge_ps'])
            m = m[sorting,:]
            s = s[sorting,:]
        
        for i in np.arange(m.shape[0]):
            if param == 'charge_ps':
                # plt.fill_between(x, m[i,:]-s[i,:], m[i,:]+s[i,:], color=colors[i], alpha=1)
                plt.plot(x, m[i,:], color=colors[i], label='bin {}'.format(i+1), lw=0.5, alpha=1)
            else:
                plt.fill_between(x, m[i,:]-s[i,:], m[i,:]+s[i,:], color=colors[i], alpha=0.2)
                plt.plot(x, m[i,:], color=colors[i], label='bin {}'.format(i+1))
        plt.ylim(ylim)
        plt.axvline(0, lw=0.5, color='k')
        plt.axhline(0, lw=0.5, color='k')
        plt.title('{}'.format(param))
        plt.xlabel('Time from pulse (s)')
        plt.ylabel(ylabel)
        # plt.legend()
        # ax.get_legend().remove()
        # except:
        #     pass
        plot_nr += 1
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

def catplot_scalars(df, measure, ylim, log_scale=False, charge=None):

    fig = plt.figure(figsize=(6,6))
    plot_nr = 1
    for rate in np.sort(np.unique(df['rate_bin'])):
        for cuff_type in ['intact', 'single', 'double']:
            try:
                ax = fig.add_subplot(3,3,plot_nr)
                colors = sns.dark_palette("red", 4)
                for width, color in zip(np.sort(df['width_bin'].unique()), colors):
                    ind = (df['cuff_type']==cuff_type)&(df['rate_bin']==rate)&(df['width_bin']==width)
                    means = df.loc[ind,:].groupby(['amplitude_bin', 'width_bin']).mean()
                    sems = df.loc[ind,:].groupby(['amplitude_bin', 'width_bin']).sem()
                    if charge == 'charge':
                        ax.errorbar(x=means['train_amp']*means['train_width'], y=means[measure], xerr=sems['train_amp'], yerr=sems[measure], color=color, elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
                        x = np.array(means['train_amp']*means['train_width'])
                    elif charge == 'charge_ps':
                        ax.errorbar(x=means['train_amp']*means['train_width']*means['train_rate'], y=means[measure], xerr=sems['train_amp'], yerr=sems[measure], color=color, elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
                        x = np.array(means['train_amp']*means['train_width']*means['train_rate'])
                    else:
                        ax.errorbar(x=means['train_amp'], y=means[measure], xerr=sems['train_amp'], yerr=sems[measure], color=color, elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
                        x = np.array(means['train_amp'])  
                    y = np.array(means[measure])
                    try:
                        (m,b) = sp.polyfit(x, y, 1)
                        regression_line = sp.polyval([m,b],x)
                        plt.plot(x,regression_line, ls='-', color=color)
                    except:
                        pass
                if log_scale:
                    plt.xscale('log')
                plt.ylim(ylim)
                plt.ylabel(measure)
                plt.xlabel('Amplitude')
                # stats:
                if rate == 0:
                    model = ols("{} ~ train_width + train_amp + train_rate".format(measure), data=df.loc[(df['cuff_type']==cuff_type),:],).fit()
                    ax.text(x=0, y=0.9, s='A: p = {}'.format(min(round(3*model.pvalues['train_amp'],3),1)), transform=ax.transAxes, size=8, horizontalalignment='left')
                    ax.text(x=0, y=0.8, s='W: p = {}'.format(min(round(3*model.pvalues['train_width'],3),1)), transform=ax.transAxes, size=8, horizontalalignment='left')
                    ax.text(x=0, y=0.7, s='R: p = {}'.format(min(round(3*model.pvalues['train_rate'],3),1)), transform=ax.transAxes, size=8, horizontalalignment='left')
            except:
                pass
            plot_nr += 1
    plt.tight_layout()
    sns.despine(trim=False, offset=3)
    return fig

def plot_scalars(df_stim, measure, ylabel='Pupil response', ylim=(None, None), p0=False):

    fig = plt.figure(figsize=(8,2))
    plot_nr = 1
    
    if 'calcium' in measure:
        # bar_split = 'charge_bin'
        bar_split = ['charge_bin', 'rate']
    else:
        # bar_split = 'charge_bin'
        bar_split = ['charge_bin', 'rate']
    
    for bin_by, x_measure, param in zip(['amplitude_bin', 'width_bin', 'rate_bin', bar_split,], 
                                        ['amplitude', 'width', 'rate', 'charge',], 
                                        ['amplitude', 'width', 'rate', 'charge',]):

        means = df_stim.groupby(bin_by)[[measure, x_measure]].mean().reset_index()
        sems = df_stim.groupby(bin_by)[[measure, x_measure]].sem().reset_index()
        sems[measure] = sems[measure]*1.96

        # try:
        #     means = df_stim.groupby(['subj_idx', bin_by]).mean().groupby(bin_by)[[measure, x_measure]].mean().reset_index()
        #     sems = df_stim.groupby(['subj_idx', bin_by]).mean().groupby(bin_by)[[measure, x_measure]].sem().reset_index()
        # except:
        #     continue
    
        colors = sns.color_palette("YlOrRd", means.shape[0])

        ax = fig.add_subplot(1,4,plot_nr)

        if 'charge' in x_measure:
            
            p_values = []
            for df, d in df_stim.groupby(bin_by):
                if p0:

                    if len(bin_by) == 2:
                        bb = bin_by[0]
                    else:
                        bb = bin_by
                    print(bb)
                    baseline = df_stim.loc[df_stim[bb]==min(df_stim[bb]), measure]
                    t,p = sp.stats.ttest_ind(d[measure], baseline)
                else:
                    t,p = sp.stats.ttest_1samp(d[measure], 0)
                p_values.append(p)
            if p0:
                p_values = np.concatenate((np.array([1]), sm.stats.multitest.multipletests(np.array(p_values)[1:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]))
            else:
                p_values = sm.stats.multitest.multipletests(np.array(p_values), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
            print(p_values)

            with matplotlib.rc_context({"lines.linewidth": 0.5}):
                # if 'calcium' in measure:
                #     sns.barplot(x='charge_bin', y=measure, ci=95, palette='YlOrRd', data=df_stim)
                #     nr_rates = 1

                # else:
                sns.barplot(x='charge_bin', y=measure, hue='rate', ci=95, palette='YlOrRd', data=df_stim)
                nr_rates = len(df_stim['rate_bin'].unique())
            width = 0.8/nr_rates 
            locs = np.cumsum(np.repeat(width,nr_rates))-np.cumsum(np.repeat(width,nr_rates)).mean()               

            for i in range(p_values.shape[0]):
                if p_values[i] < 0.001:
                    sig = '***'
                elif p_values[i] < 0.01:
                    sig = '**'
                elif p_values[i] < 0.05:
                    sig = '*'
                else:
                    sig = 'n.s.'
                ax.text(x=int(i/nr_rates)+locs[(i%nr_rates)]+width/4, y=0.75*ylim[1], s=sig, size=6, horizontalalignment='center', verticalalignment='center', rotation=90)

            try:
                ax.get_legend().remove()
            except:
                pass
        else:
            x = np.array(means[x_measure])
            y = np.array(means[measure])

            if 'charge' in x_measure:
                x = np.log10(x)
            for i in range(len(x)):
                plt.plot(x[i],y[i], 'o', mfc='lightgrey', color=colors[i], zorder=1)
            ax.errorbar(x=x, y=y, yerr=sems[measure], fmt='none', elinewidth=0.5, markeredgewidth=0.5, ecolor='k', capsize=2, zorder=2)
            y = y[np.argsort(x)]
            x = x[np.argsort(x)]
            
            try:
                if 'charge' in x_measure:
                    popt, pcov = curve_fit(fsigmoid, x, y, method='dogbox', bounds=([-100, -0.75, y.max()/2],[100, 0.75, y.max()*2]))
                    ax.plot(np.linspace(0, max(x), 100), fsigmoid(np.linspace(0, max(x), 100), *popt), color='k', ls='-', zorder=10)
                else:
                    popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
                    ax.plot(np.linspace(0, max(x), 100), log_logistic(np.linspace(0, max(x), 100), *popt), color='k', ls='-', zorder=10)
            except:
                pass

            # add stats:
            p_values = []
            for df, d in df_stim.groupby(bin_by):
                if p0:

                    if len(bin_by) == 1:
                        bb = bin_by[0]
                    else:
                        bb = bin_by
                    print(bb)
                    baseline = df_stim.loc[df_stim[bb]==min(df_stim[bb]), measure]
                    t,p = sp.stats.ttest_ind(d[measure], baseline)
                else:
                    t,p = sp.stats.ttest_1samp(d[measure], 0)
                p_values.append(p)
            if p0:
                p_values = np.concatenate((np.array([1]), sm.stats.multitest.multipletests(np.array(p_values)[1:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]))
            else:
                p_values = sm.stats.multitest.multipletests(np.array(p_values), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
            print(min(p_values.ravel()))
            for i in range(len(x)):
                if p_values[i] < 0.001:
                    sig = '***'
                elif p_values[i] < 0.01:
                    sig = '**'
                elif p_values[i] < 0.05:
                    sig = '*'
                else:
                    sig = 'n.s.'
                ax.text(x=x[i], y=0.75*ylim[1], s=sig, size=6, horizontalalignment='center', verticalalignment='center')

        if isinstance(bin_by, str):
            plt.xticks(ticks=np.round(np.sort(df_stim[x_measure].unique()),1), labels=np.round(np.sort(df_stim[x_measure].unique()),1))

        plt.ylim(ylim)
        plt.title('{}'.format(param))
        plt.xlabel(x_measure)
        plt.ylabel(ylabel)
        # plt.legend()

        # ax.get_legend().remove()
        plot_nr += 1
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

# def plot_scalars2(df_stim, measure, ylabel='Pupil response', ylim=(None, None)):

#     fig = plt.figure(figsize=(6,4))
#     plot_nr = 1
#     for bin_by, x_measure, param in zip([['amplitude_bin', 'width_bin'], ['width_bin', 'rate_bin'], ['rate_bin', 'amplitude_bin'], ['amplitude_bin', 'rate_bin'], ['width_bin', 'amplitude_bin'], ['rate_bin', 'width_bin']], 
#                                         ['amplitude', 'width', 'rate', 'amplitude', 'width', 'rate',], 
#                                         ['amplitude', 'width', 'rate','amplitude', 'width', 'rate',]):
        
#         means = df_stim.groupby(bin_by)[[measure, x_measure]].mean().reset_index()
#         sems = df_stim.groupby(bin_by)[[measure, x_measure]].sem().reset_index()
#         sems[measure] = sems[measure]*1.96

#         ax = fig.add_subplot(2,3,plot_nr)
#         for i, m2 in enumerate(means[bin_by[1]].unique()):
            
#             x = np.array(means.loc[means[bin_by[1]]==m2, x_measure])
#             y = np.array(means.loc[means[bin_by[1]]==m2, measure])
#             colors = sns.color_palette("YlOrRd", len(means[bin_by[1]].unique()))
#             for j in range(len(x)):
#                 plt.plot(x[j],y[j], 'o', mfc='lightgrey', color=colors[i], zorder=1)
#             ax.errorbar(x=x, y=y, yerr=np.array(sems.loc[sems[bin_by[1]]==m2, measure]), elinewidth=0.5, markeredgewidth=0.5, 
#                             fmt='none', ecolor='darkgrey', capsize=2, zorder=2)
#             try:
#                 y = y[np.argsort(x)]
#                 x = x[np.argsort(x)]
#                 popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
#                 ax.plot(np.linspace(0, max(x), 100), log_logistic(np.linspace(0, max(x), 100), *popt), color=colors[i], ls='-', zorder=10)
#             except:
#                 pass
#         plt.ylim(ylim)
#         plt.title('{}'.format(param))
#         plt.xlabel(x_measure)
#         plt.ylabel(ylabel)
#         # plt.legend()

#         # ax.get_legend().remove()
#         plot_nr += 1
#     sns.despine(trim=False, offset=3)
#     plt.tight_layout()
#     return fig


def plot_scalars2(df_stim, measure, ylabel='Pupil response', ylim=(None, None)):

    fig = plt.figure(figsize=(6,4))
    plot_nr = 1
    for bin_by, x_measure, param in zip([['amplitude_bin', 'width_bin'], ['width_bin', 'rate_bin'], ['rate_bin', 'amplitude_bin'], ['amplitude_bin', 'rate_bin'], ['width_bin', 'amplitude_bin'], ['rate_bin', 'width_bin']], 
                                        ['amplitude', 'width', 'rate', 'amplitude', 'width', 'rate',], 
                                        ['amplitude', 'width', 'rate','amplitude', 'width', 'rate',]):
        
        means = df_stim.groupby(bin_by)[[measure, x_measure]].mean().reset_index()
        sems = df_stim.groupby(bin_by)[[measure, x_measure]].sem().reset_index()
        sems[measure] = sems[measure]*1.96

        ax = fig.add_subplot(2,3,plot_nr)
        for i, m2 in enumerate(means[bin_by[1]].unique()):
            
            x = np.array(means.loc[means[bin_by[1]]==m2, x_measure])
            y = np.array(means.loc[means[bin_by[1]]==m2, measure])
            colors = sns.color_palette("YlOrRd", len(means[bin_by[1]].unique()))
            for j in range(len(x)):
                plt.plot(x[j],y[j], 'o', mfc='lightgrey', color=colors[i], zorder=1)
            ax.errorbar(x=x, y=y, yerr=np.array(sems.loc[sems[bin_by[1]]==m2, measure]), elinewidth=0.5, markeredgewidth=0.5, 
                            fmt='none', ecolor='darkgrey', capsize=2, zorder=2)
            try:
                y = y[np.argsort(x)]
                x = x[np.argsort(x)]
                popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
                ax.plot(np.linspace(0, max(x), 100), log_logistic(np.linspace(0, max(x), 100), *popt), color=colors[i], ls='-', zorder=10)
            except:
                pass
        plt.ylim(ylim)
        plt.title('{}'.format(param))
        plt.xlabel(x_measure)
        plt.ylabel(ylabel)
        # plt.legend()

        plt.xticks(ticks=np.round(np.sort(df_stim[x_measure].unique()),1), labels=np.round(np.sort(df_stim[x_measure].unique()),1))

        # ax.get_legend().remove()
        plot_nr += 1
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

def plot_scalars3(df_stim, measure, ylabel='Pupil response', ylim=(None, None)):

    fig = plt.figure(figsize=(6,2))
    plot_nr = 1
    for bin_by, x_measure, param in zip([['amplitude_bin', 'width_bin'], ['amplitude_bin', 'width_bin'], ['amplitude_bin', 'width_bin'],], 
                                        ['charge', 'charge', 'charge',], 
                                        ['5Hz', '10Hz', '20Hz',]):
        
        if plot_nr == 1:
            means = df_stim.loc[df_stim['rate']==5,:].groupby(bin_by)[[measure, x_measure]].mean().reset_index()
            sems = df_stim.loc[df_stim['rate']==5,:].groupby(bin_by)[[measure, x_measure]].sem().reset_index()
            sems[measure] = sems[measure]*1.96
        if plot_nr == 2:
            means = df_stim.loc[df_stim['rate']==10,:].groupby(bin_by)[[measure, x_measure]].mean().reset_index()
            sems = df_stim.loc[df_stim['rate']==10,:].groupby(bin_by)[[measure, x_measure]].sem().reset_index()
            sems[measure] = sems[measure]*1.96
        if plot_nr == 3:
            means = df_stim.loc[df_stim['rate']==20,:].groupby(bin_by)[[measure, x_measure]].mean().reset_index()
            sems = df_stim.loc[df_stim['rate']==20,:].groupby(bin_by)[[measure, x_measure]].sem().reset_index()
            sems[measure] = sems[measure]*1.96

        ax = fig.add_subplot(1,3,plot_nr)
        for i, m2 in enumerate(means[bin_by[1]].unique()):
            
            x = np.array(means.loc[means[bin_by[1]]==m2, x_measure])
            y = np.array(means.loc[means[bin_by[1]]==m2, measure])
            colors = sns.color_palette("YlOrRd", len(means[bin_by[1]].unique()))
            for j in range(len(x)):
                plt.plot(x[j],y[j], 'o', mfc='lightgrey', color=colors[i], zorder=1)
            ax.errorbar(x=x, y=y, yerr=np.array(sems.loc[sems[bin_by[1]]==m2, measure]), elinewidth=0.5, markeredgewidth=0.5, 
                            fmt='none', ecolor='darkgrey', capsize=2, zorder=2)
            try:
                y = y[np.argsort(x)]
                x = x[np.argsort(x)]
                popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
                ax.plot(np.linspace(0.01, max(x), 100), log_logistic(np.linspace(0.01, max(x), 100), *popt), color=colors[i], ls='-', zorder=10)
            except:
                pass
        plt.ylim(ylim)
        plt.title('{}'.format(param))
        plt.xlabel(x_measure)
        plt.ylabel(ylabel)
        # plt.legend()
        ax.set_xlim(xmin=0.01)
        ax.set_xscale('log')

        # ax.get_legend().remove()
        plot_nr += 1
    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

def plot_pupil_responses_matrix(df_stim, measure, vmin=-0.1, vmax=0.1):

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)

    fig = plt.figure(figsize=(6,2))
    plt_nr = 1

    # stats:
    p_values = []
    for rate_bin in [0,1,2]:
        t_test = df_stim.loc[df_stim['rate_bin']==rate_bin,:].groupby(['amplitude_bin', 'width_bin'])[measure].apply(sp.stats.ttest_1samp, 0).as_matrix().reshape((5,4))
        p_values.append( np.array([t_test[i,j][1] for i in range(t_test.shape[0]) for j in range(t_test.shape[1])]).reshape((5,4)) )
    p_values = np.stack((p_values))
    p_values = sm.stats.multitest.multipletests(p_values.ravel(), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1].reshape(p_values.shape)

    # panels:
    for rate_bin in [0,1,2,]:
        ax = fig.add_subplot(1,3,plt_nr)
        a = df_stim.loc[df_stim['rate_bin']==rate_bin,:].groupby(['amplitude_bin', 'width_bin']).mean()[measure].as_matrix().reshape((5,4))

        if measure == 'distance_resp_0c':
             p_values[rate_bin, (a<0)] = 1

        cax = ax.pcolormesh(np.arange(5), np.arange(6), a, vmin=vmin, vmax=vmax, cmap=cmap)
        for i in range(p_values.shape[1]):
            for j in range(p_values.shape[2]):
                if p_values[rate_bin,i,j] < 0.001:
                    sig = '***'
                elif p_values[rate_bin,i,j] < 0.01:
                    sig = '**'
                elif p_values[rate_bin,i,j] < 0.05:
                    sig = '*'
                else:
                    sig = ''
                ax.text(x=j+0.5, y=i+0.5, s=sig, size=6, horizontalalignment='center', verticalalignment='center')
        plt.title('Rate bin {}'.format(rate_bin))
        plt.xlabel('Width (ms)')
        plt.ylabel('Amplitude (mA)')
        plt.xticks(ticks=np.arange(4)+0.5, labels=[0.1, 0.2, 0.4, 0.8])
        plt.yticks(ticks=np.arange(5)+0.5, labels=[0.1, 0.3, 0.5, 0.7, 0.9])
        plt.colorbar(cax, ticks=[vmin, 0, vmax])
        plt_nr+=1    
    plt.tight_layout()
    return fig

def plot_pupil_responses_matrix_():

    charge = []
    amp = []
    width = []
    widths = np.array([0.1, 0.2, 0.4, 0.8])
    amps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for a in amps:
        for w in widths:
            amp.append(a)
            width.append(w)
            charge.append(a*w)
    charge = np.array(charge)

    # sorting = np.argsort(np.array(charge))
    # charge = charge[sorting]
    # amp = np.array(amp)[sorting]
    # width = np.array(width)[sorting]

    X = np.array(charge).reshape((5,4))

    rates = np.array([5,10,20])

    fig = plt.figure(figsize=(8,2))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
    plt_nr = 1
    for rate_bin in [0,1,2,3]:
        ax = fig.add_subplot(1,4,plt_nr)
        if rate_bin < 3:
            vmin = -1
            vmax = 1
            cax = ax.pcolormesh(np.arange(5), np.arange(6), np.log10(X*rates[rate_bin]), vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            vmin = -2
            vmax = 0
            cax = ax.pcolormesh(np.arange(5), np.arange(6), np.log10(X), vmin=vmin, vmax=vmax, cmap=cmap)
        plt.title('Rate bin {}'.format(rate_bin))
        plt.xlabel('Width (ms)')
        plt.ylabel('Amplitude (mA)')
        plt.xticks(ticks=np.arange(4)+0.5, labels=[0.1, 0.2, 0.4, 0.8])
        plt.yticks(ticks=np.arange(5)+0.5, labels=[0.1, 0.3, 0.5, 0.7, 0.9])
        plt.colorbar(cax, ticks=[vmin, 0, vmax])
        plt_nr+=1
    plt.tight_layout()
    return fig


def plot_pupil_walking_relationship(df_meta, n_bins=8, distance_cutoff=0.005, ylim=(None, None)):

    fig = plt.figure(figsize=(8,3))

    plot_nr = 1
    for bin_by, x_measure, param in zip(['amplitude_bin', 'width_bin', 'rate_bin', ['charge_bin', 'rate']], 
                                        ['amplitude', 'width', 'rate', 'charge'], 
                                        ['amplitude', 'width', 'rate', 'charge']):
        

        colors = sns.color_palette("YlOrRd", df_meta.groupby(bin_by).mean().shape[0])
        print()
        print()
        ax = fig.add_subplot(1,4,plot_nr)
        
        i = 0
        for par, df in df_meta.groupby(bin_by):
            df = df.loc[df['distance']>distance_cutoff,:]
            ind = (df['distance_r']<distance_cutoff)
            print(np.mean(ind))
            # df = df.loc[df['distance_resp_default']<distance_cutoff,:]
            df['distance_bin'] = pd.qcut(df['distance'], q=n_bins, labels=False)
            d = df.groupby('distance_bin').mean()[['distance', 'pupil']].reset_index()
            x = np.array(d['distance'])
            y = np.array(d['pupil'])
            plt.plot(x, y, 'o', color=colors[i])
            try:
                popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
                plt.plot(np.linspace(0, 1, 100), log_logistic(np.linspace(0, max(x), 100), *popt), color=colors[i], ls='-', zorder=10)
            except:
                pass
            
            i+=1

        df = df_meta.loc[df_meta['distance_r']>distance_cutoff,:]
        ind = (df['distance_resp_default_b']<distance_cutoff)
        print(np.mean(ind))
        # df = df.loc[df['distance_resp_default_b']<distance_cutoff,:]
        df['distance_bin'] = pd.qcut(df['distance_resp_default'], q=n_bins, labels=False)
        d = df.groupby('distance_bin').mean()[['distance_resp_default', 'pupil_resp_default']].reset_index()
        x = np.array(d['distance_resp_default'])
        y = np.array(d['pupil_resp_default'])
        plt.plot(x, y, 'o', color='b',)
        popt, pcov = curve_fit(log_logistic, x, y, method='trf', bounds=([0, 0, y.max()/2],[40, 2, y.max()*2]))
        plt.plot(np.linspace(0, 1, 100), log_logistic(np.linspace(0, max(x), 100), *popt), color='b', ls='-', zorder=10)

        ax.set_ylim(ylim)
        ax.set_title('{}'.format(param))

        plot_nr += 1

    plt.xlabel('Distance (m)')
    plt.ylabel('Pupil response')

    sns.despine(trim=False, offset=3)
    plt.tight_layout()

    return fig

def plot_pupil_responses(df_meta, epochs, bins, ylabel='Pupil response', ylim=(-0.1, 0.2)):

    x = np.round(np.array(epochs.columns, dtype=float), 2)

    epochs_r = epochs.loc[:,(x>=-60)&(x<-20)]
    epochs_v = epochs.loc[:,(x>=-20)&(x<20)]

    if isinstance(bins, int):
        bins_r = pd.qcut(epochs_r.loc[:,(epochs_r.columns>=-45)&(epochs_r.columns<-40)].mean(axis=1), q=bins, labels=False)
        bins_v = pd.qcut(epochs_v.loc[:,(epochs_v.columns>=-5)&(epochs_v.columns<-0)].mean(axis=1), q=bins, labels=False)
    else:
        bins_r = pd.cut(epochs_r.loc[:,(epochs_r.columns>=-45)&(epochs_r.columns<-40)].mean(axis=1), bins=bins, labels=False)
        bins_v = pd.cut(epochs_v.loc[:,(epochs_v.columns>=-5)&(epochs_v.columns<-0)].mean(axis=1), bins=bins, labels=False)
    
    # shell()

    fig = plt.figure(figsize=(6,2))

    ax = fig.add_subplot(131)
    means = np.array(epochs.mean(axis=0))
    sems = np.array(epochs.sem(axis=0))
    plt.fill_between(np.array(epochs.columns) , means-sems, means+sems, color=sns.color_palette("YlOrRd", 5)[2], alpha=0.2)
    plt.plot(np.array(epochs.columns) , means, lw=1, color=sns.color_palette("YlOrRd", 5)[2]) #, label='bin {}'.format(i+1))
    plt.axvline(0, lw=0.5, color='k')
    plt.axvspan(-27.5, -22.5, color='k', alpha=0.1)
    plt.axvspan(-20, -15, color='k', alpha=0.1)
    plt.axvspan(-12.5, -7.5, color='k', alpha=0.1)
    plt.axvspan(-5, 0, color='k', alpha=0.1)
    plt.axvspan(2.5, 7.5, color='k', alpha=0.1)
    plt.xlabel('Time from pulse (s)')
    plt.ylabel('Pupil size')

    ax = fig.add_subplot(132)
    means = epochs_r.groupby(bins_r).mean()
    sems = epochs_r.groupby(bins_r).sem()
    for i in range(means.shape[0]):
        plt.fill_between(np.array(epochs_v.columns), means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color='cornflowerblue', alpha=0.2)
        plt.plot(np.array(epochs_v.columns), means.iloc[i], lw=1, color='cornflowerblue') #, label='bin {}'.format(i+1))
    plt.ylim(0,1)
    plt.axvline(0, lw=0.5, color='k')
    plt.axhline(0, lw=0.5, color='k')
    plt.axvspan(-5, 0, color='k', alpha=0.1)
    plt.axvspan(2.5, 7.5, color='k', alpha=0.1)
    plt.xlabel('Time from pulse (s)')
    plt.ylabel('Pupil size')

    ax = fig.add_subplot(133)
    means = epochs_r.groupby(bins_r).mean()
    sems = epochs_r.groupby(bins_r).sem()
    for i in range(means.shape[0]):
        plt.fill_between(np.array(epochs_v.columns), means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color='cornflowerblue', alpha=0.2)
        plt.plot(np.array(epochs_v.columns), means.iloc[i], lw=1, color='cornflowerblue') #, label='bin {}'.format(i+1))
    means = epochs_v.groupby(bins_v).mean()
    sems = epochs_v.groupby(bins_v).sem()
    colors = sns.color_palette("YlOrRd", means.shape[0])
    for i in range(means.shape[0]):
        plt.fill_between(np.array(epochs_v.columns), means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color=colors[i], alpha=0.2)
        plt.plot(np.array(epochs_v.columns), means.iloc[i], lw=1, color=colors[i]) #, label='bin {}'.format(i+1))
    plt.ylim(0,1)
    plt.axvline(0, lw=0.5, color='k')
    plt.axhline(0, lw=0.5, color='k')
    plt.axvspan(-5, 0, color='k', alpha=0.1)
    plt.axvspan(2.5, 7.5, color='k', alpha=0.1)
    plt.xlabel('Time from pulse (s)')
    plt.ylabel('Pupil size')

    # ax = fig.add_subplot(133)
    # means = epochs.groupby([df_meta['bins']]).mean()
    # means.loc[:,:] = means.loc[:,:] - epochs_r.groupby([df_meta['pupilr_0_bin']]).mean().get_values()
    # sems = epochs.groupby([df_meta['bins']]).sem()
    
    # # smooth:
    # for i in range(means.shape[0]):
    #     means.iloc[i] = means.iloc[i].rolling(window=10, center=True).mean()
    #     sems.iloc[i] = sems.iloc[i].rolling(window=10, center=True).mean()

    # colors = sns.color_palette("YlOrRd", means.shape[0])
    # for i in range(means.shape[0]):
    #     plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], color=colors[i], alpha=0.2)
    #     plt.plot(x, means.iloc[i], color=colors[i]) #, label='bin {}'.format(i+1))
    # # plt.ylim(ylim)
    # plt.axvline(0, lw=0.5, color='k')
    # plt.axhline(0, lw=0.5, color='k')
    # plt.axvspan(-7, -4, color='k', alpha=0.1)
    # plt.axvspan(-3, 0, color='k', alpha=0.1)
    # plt.axvspan(1, 4, color='k', alpha=0.1)
    # plt.xlabel('Time from pulse (s)')

    sns.despine(trim=False, offset=3)
    plt.tight_layout()
    return fig

def plot_reversion_to_mean_correction(df, measure, bin_measure, func):

    lp = 0
    hp = 100

    nr_subjects = len(df.subj_idx.unique())
    n_rows = np.ceil(nr_subjects/4)

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    means = df.groupby('bins_{}'.format(bin_measure)).mean()
    sems = df.groupby('bins_{}'.format(bin_measure)).sem()
    x = np.array(means['{}_b'.format(bin_measure)])
    y = np.array(means['{}_change'.format(measure)])
    x_pred = np.linspace(min(x), max(x), 100)
    plt.scatter(df['{}_b'.format(bin_measure)], df['{}_change'.format(measure)], s=3, color='grey', alpha=0.2, rasterized=True)
    plt.errorbar(x, y, yerr=sems['{}_change'.format(measure)], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
    popt, pcov = curve_fit(func, df.loc[~np.isnan(df['{}_b'.format(bin_measure)]),'{}_b'.format(bin_measure)], df.loc[~np.isnan(df['{}_b'.format(bin_measure)]),'{}_change'.format(measure)],)
    # popt2, pcov = curve_fit(func, df['{}_b'.format(measure)], df['{}_change'.format(measure)], sigma=1/df['bins_{}_count'.format(measure)])
    # popt3, pcov = curve_fit(func, df.groupby('bins_{}'.format(measure))['{}_b'.format(measure)].mean(), df.groupby('bins_{}'.format(measure))['{}_change'.format(measure)].mean(), sigma=df.groupby('bins_{}'.format(measure)).count()[measure])
    plt.plot(x_pred, func(x_pred, *popt), color='red', alpha=0.75, ls='-', zorder=10)
    # plt.ylim(np.percentile(df['{}_change'.format(measure)],lp), np.percentile(df['{}_change'.format(measure)],hp))
    plt.ylim(np.percentile(df.loc[~np.isnan(df['{}_change'.format(measure)]),'{}_change'.format(measure)],lp),
        np.percentile(df.loc[~np.isnan(df['{}_change'.format(measure)]),'{}_change'.format(measure)],hp))
    
    plt.xlabel('Measure (t)')
    plt.ylabel('Measure (t+1)')
    
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    sns.despine(trim=False, offset=3)
    
    fig2 = plt.figure(figsize=(8,2*n_rows))
    plt_nr = 1
    for s, d in df.groupby(['subj_idx']):
        ax = fig2.add_subplot(n_rows,4,plt_nr)
        
        try:
            means = d.groupby('bins_{}'.format(bin_measure)).mean()
            sems = d.groupby('bins_{}'.format(bin_measure)).sem()
            x = np.array(means['{}_b'.format(bin_measure)])
            y = np.array(means['{}_change'.format(measure)])
            x_pred = np.linspace(min(x), max(x), 100)
            
            plt.scatter(d['{}_b'.format(bin_measure)], d['{}_change'.format(measure)], s=3, color='grey', alpha=0.2, rasterized=True)

            # d['bins'] = pd.qcut(d[bin_measure], q=50)
            # m = d.groupby('bins').mean()
            # plt.scatter(m['{}_b'.format(bin_measure)], m['{}_change'.format(measure)], s=2, color='grey', alpha=0.2, rasterized=True)
            
            plt.errorbar(x, y, yerr=sems['{}_change'.format(measure)], color='k', elinewidth=0.5, mfc='lightgrey', fmt='o', ecolor='lightgray', capsize=0)
            popt_, pcov_ = curve_fit(func, d.loc[~np.isnan(d['{}_b'.format(bin_measure)]),'{}_b'.format(bin_measure)], d.loc[~np.isnan(d['{}_b'.format(bin_measure)]),'{}_change'.format(measure)],)

            plt.plot(x_pred, func(x_pred, *popt), color='red', alpha=0.75, ls='-', zorder=10)
            plt.plot(x_pred, func(x_pred, *popt_), color='orange', alpha=0.75, ls='-', zorder=11)
            plt.ylim(np.percentile(d.loc[~np.isnan(d['{}_change'.format(measure)]),'{}_change'.format(measure)],lp),
                    np.percentile(d.loc[~np.isnan(d['{}_change'.format(measure)]),'{}_change'.format(measure)],hp))
            plt.xlabel('Measure (t)')
            plt.ylabel('Measure (t+1)')
        except:
            pass
        plt_nr += 1
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    sns.despine(trim=False, offset=3)    
    
    return fig, fig2

def load_stim_parameters(file_meta):

    # open meta data:
    file = h5py.File(file_meta, 'r')
    df_stim = pd.DataFrame({
        'time': file['train_starts'][()].ravel(),
        'width': file['train_widths'][()].ravel(),
        'amplitude_m': file['train_amps'][()].ravel(),
        'rate': file['train_rate_round'][()].ravel(),
        })
    frames = file['frame_numbers'][()].ravel().astype(int)
    pulses =  file['Pulse_idxs'][()].ravel().astype(int)
    stim_si = file['StimSI'][()].ravel()
        
    return df_stim, frames, pulses, stim_si

def correct_scalars(df_meta, group, velocity_cutoff, ind_clean_w):

    # correct pupil and eyelid scalars:
    # --------------------------------

    for m in ['pupil', 'eyelid']:

        # make random df:
        dfs = []
        for i in [0,-1,-2]:
            df = df_meta.loc[group,['{}_{}'.format(m, str(i)), '{}_{}'.format(m, str(i-1)), 'subj_idx', 'session']]
            df = df.rename(columns={'{}_{}'.format(m, str(i)): m, '{}_{}'.format(m, str(i-1)): '{}_b'.format(m)}).reset_index(drop=True)
            dfs.append(df)
        df = pd.concat(dfs)
        df['{}_change'.format(m)] = df[m] - df['{}_b'.format(m)]
        # df['pupil_change'] = df['pupil']

        # bins:
        bins = np.array([-10,0.2,0.3,0.4,0.5,0.6,0.7,0.8,10])
        df_meta['bins_{}'.format(m)] = pd.cut(df_meta['{}_0'.format(m)], bins=bins, labels=False)
        df['bins_{}'.format(m)] = pd.cut(df['{}_b'.format(m)], bins=bins, labels=False)

        # # bin count:
        # pupil_counts = df.groupby(['bins_pupil']).count().reset_index()
        # df['bins_pupil_count'] = 0
        # for b in df['bins_pupil'].unique():
        #     df.loc[df['bins_pupil']==b, 'bins_pupil_count'] = int(pupil_counts.loc[pupil_counts['bins_pupil']==b, 'pupil'])

        # function:
        func = qubic

        # plot:
        if m == 'pupil':
            fig1, fig2 = plot_reversion_to_mean_correction(df, measure='pupil', bin_measure='pupil', func=func)
        if m == 'eyelid':
            fig3, fig4 = plot_reversion_to_mean_correction(df, measure='eyelid', bin_measure='eyelid', func=func)

        # uncorrected:
        df_meta.loc[group, m] = df_meta.loc[group, '{}_1'.format(m)] - df_meta.loc[group, '{}_0'.format(m)]
        # df_meta['pupil'] = df_meta['pupil_1']

        # correct across subjects:
        popt, pcov = curve_fit(func, df.loc[~np.isnan(df['{}_b'.format(m)]), '{}_b'.format(m)], df.loc[~np.isnan(df['{}_b'.format(m)]), '{}_change'.format(m)])
        df_meta.loc[group, '{}_c'.format(m)] = df_meta.loc[group, m] - func(df_meta.loc[group, '{}_0'.format(m)], *popt)
        
        # correct per subject:
        df_meta.loc[group, '{}_c2'.format(m)] = np.repeat(np.NaN, df_meta.loc[group,:].shape[0])
        for subj in df['subj_idx'].unique():
            try:
                ind = df['subj_idx'] == subj
                popt, pcov = curve_fit(func, df.loc[ind&~np.isnan(df['{}_b'.format(m)]),'{}_b'.format(m)], df.loc[ind&~np.isnan(df['{}_b'.format(m)]),'{}_change'.format(m)])
                ind = (df_meta['subj_idx'] == subj) & group
                df_meta.loc[ind,'{}_c2'.format(m)] = df_meta.loc[ind,m] - func(df_meta.loc[ind,'{}_0'.format(m)], *popt)
            except:
                ind = (df_meta['subj_idx'] == subj) & group
                df_meta.loc[ind,'{}_c2'.format(m)] = np.NaN

        # plt.figure()
        # plt.scatter(df_meta.loc[group,'pupil'], df_meta.loc[group,'pupil_c'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group,'pupil'], df_meta.loc[group,'pupil_c2'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group,'pupil_c'], df_meta.loc[group,'pupil_c2'])

    if sum(ind_clean_w) > 0:

        # correct velocity scalars:
        # -------------------------

        # make random df:
        dfs = []
        for i in [0,-1,-2]:
            df = df_meta.loc[group&ind_clean_w,['velocity_{}'.format(str(i)), 'velocity_{}'.format(str(i-1)), 'subj_idx', 'session']]
            df = df.rename(columns={'velocity_{}'.format(str(i)): 'velocity', 'velocity_{}'.format(str(i-1)): 'velocity_b'}).reset_index(drop=True)
            dfs.append(df)
        df = pd.concat(dfs)
        # df['walk_b'] = ((df['velocity_b'] <= velocity_cutoff[0])|(df['velocity_b'] > velocity_cutoff[1])).astype(int)
        # df['walk'] = ((df['velocity'] <= velocity_cutoff[0])|(df['velocity'] > velocity_cutoff[1])).astype(int)
        df['walk_b'] = (df['velocity_b'] > velocity_cutoff[1]).astype(int)
        df['walk'] = (df['velocity'] > velocity_cutoff[1]).astype(int)

        # df['velocity_change'] = df['velocity'] - df['velocity_b']
        # df['walk_change'] = df['walk'] - df['walk_b']

        df['velocity_change'] = df['velocity']
        df['walk_change'] = df['walk'] 

        # bins:
        bins = np.array([-10,0.005,0.05,0.1,0.2,0.5,1,2,10])
        df['bins_velocity'] = pd.cut(df['velocity_b'], bins=bins, labels=False)

        # correct velocity scalars:
        # -------------------------

        # function:
        func = quadratic

        # plot:
        fig5, fig6 = plot_reversion_to_mean_correction(df.loc[df['velocity_b']>0.005,:], measure='velocity', bin_measure='velocity', func=func)
        fig7, fig8 = plot_reversion_to_mean_correction(df.loc[df['velocity_b']<=0.005,:], measure='velocity', bin_measure='velocity', func=func)

        # uncorrected:
        df_meta.loc[group, 'velocity'] = df_meta.loc[group, 'velocity_1'].copy()

        # correct across subjects:
        df_meta.loc[group, 'velocity_c'] = np.repeat(np.NaN, df_meta.loc[group,:].shape[0])
        ind_r = (df['velocity_b']>0.005)
        ind = (df_meta['velocity_0']>0.005) & group
        popt, pcov = curve_fit(func, df.loc[ind_r,'velocity_b'], df.loc[ind_r,'velocity_change'])
        df_meta.loc[ind,'velocity_c'] = df_meta.loc[ind,'velocity'] - func(df_meta.loc[ind,'velocity_0'], *popt)
        ind_r = (df['velocity_b']<=0.005)
        ind = (df_meta['velocity_0']<=0.005) & group
        df_meta.loc[ind,'velocity_c'] = df_meta.loc[ind,'velocity'] - df.loc[ind_r,'velocity_change'].mean()

        # correct per subject:
        df_meta.loc[group, 'velocity_c2'] = np.repeat(np.NaN, df_meta.loc[group,:].shape[0])
        nr_subjects = 0
        for subj in df_meta.loc[group&ind_clean_w,'subj_idx'].unique():
            ind_r = (df['subj_idx'] == subj)&(df['velocity_b']>0.005)
            ind = (df_meta['subj_idx'] == subj)&(df_meta['velocity_0']>0.005)&group&ind_clean_w
            popt, pcov = curve_fit(func, df.loc[ind_r,'velocity_b'], df.loc[ind_r,'velocity_change'])
            df_meta.loc[ind,'velocity_c2'] = df_meta.loc[ind,'velocity'] - func(df_meta.loc[ind,'velocity_0'], *popt)
            ind_r = (df['subj_idx'] == subj)&(df['velocity_b']<=0.005)
            ind = (df_meta['subj_idx'] == subj)&(df_meta['velocity_0']<=0.005)&group&ind_clean_w
            df_meta.loc[ind,'velocity_c2'] = df_meta.loc[ind,'velocity'] - df.loc[ind_r,'velocity_change'].mean()

        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'velocity'], df_meta.loc[group&ind_clean_w,'velocity_c'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'velocity'], df_meta.loc[group&ind_clean_w,'velocity_c2'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'velocity_c'], df_meta.loc[group&ind_clean_w,'velocity_c2'])

        # correct walk probability scalars:
        # ---------------------------------

        # function:
        func = linear

        # plot:
        fig9, fig10 = plot_reversion_to_mean_correction(df.loc[df['velocity_b']>0.005,:], measure='walk', bin_measure='velocity', func=func)
        fig11, fig12 = plot_reversion_to_mean_correction(df.loc[df['velocity_b']<=0.005,:], measure='walk', bin_measure='velocity', func=func)
        
        # uncorrected:
        df_meta.loc[group, 'walk'] = (df_meta.loc[group, 'velocity_1'] > velocity_cutoff[1]).astype(int)

        # correct across subjects:
        df_meta.loc[group, 'walk_c'] = np.repeat(np.NaN, df_meta.loc[group,:].shape[0])
        
        ind_r = (df['velocity_b']>0.005)
        ind = (df_meta['velocity_0']>0.005) & group
        popt, pcov = curve_fit(func, df.loc[ind_r,'walk_b'], df.loc[ind_r,'walk_change'])
        df_meta.loc[ind,'walk_c'] = df_meta.loc[ind,'walk'] - func(df_meta.loc[ind,'walk_0'], *popt)
        
        ind_r = (df['velocity_b']<=0.005)
        ind = (df_meta['velocity_0']<=0.005) & group
        df_meta.loc[ind,'walk_c'] = df_meta.loc[ind,'walk'] - df.loc[ind_r,'walk_change'].mean()

        # correct per subject:
        df_meta.loc[group, 'walk_c2'] = np.repeat(np.NaN, df_meta.loc[group,:].shape[0])
        nr_subjects = 0
        for subj in df_meta.loc[group&ind_clean_w,'subj_idx'].unique():
            ind_r = (df['subj_idx'] == subj)&(df['walk_b']>0.005)
            ind = (df_meta['subj_idx'] == subj)&(df_meta['walk_0']>0.005)&group&ind_clean_w
            popt, pcov = curve_fit(func, df.loc[ind_r,'walk_b'], df.loc[ind_r,'walk_change'])
            df_meta.loc[ind,'walk_c2'] = df_meta.loc[ind,'walk'] - func(df_meta.loc[ind,'walk_0'], *popt)
            ind_r = (df['subj_idx'] == subj)&(df['walk_b']<=0.005)
            ind = (df_meta['subj_idx'] == subj)&(df_meta['walk_0']<=0.005)&group&ind_clean_w
            df_meta.loc[ind,'walk_c2'] = df_meta.loc[ind,'walk'] - df.loc[ind_r,'walk_change'].mean()

        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'walk'], df_meta.loc[group&ind_clean_w,'walk_c'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'walk'], df_meta.loc[group&ind_clean_w,'walk_c2'])
        # plt.figure()
        # plt.scatter(df_meta.loc[group&ind_clean_w,'walk_c'], df_meta.loc[group&ind_clean_w,'walk_c2'])

        return df_meta, (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12)

    else:

        return df_meta, (fig1, fig2, fig3, fig4)


def process_meta_data(file_meta, stim_overview, cuff_type, subj, session, fig_dir):

    bad_session = False

    df_stim, frames, pulses, stim_si = load_stim_parameters(file_meta)

    # add intended params:
    try:
        stim_overview_ses = stim_overview.loc[(stim_overview['subject']==subj)&(stim_overview['session']==int(session)),:]
    except:
        stim_overview_ses = stim_overview.loc[(stim_overview['subject']==subj)&(stim_overview['session']==session),:]
    if stim_overview_ses.shape[0] == 0:
        stim_overview_ses = stim_overview.loc[(stim_overview['subject']==subj)&(stim_overview['session']==int(session[:-1])),:]

    # check if workable session:
    if not df_stim.shape[0] == 60:
        bad_session = True
        print('ERROR!!')
        print('Not 60 stimulations!')
    if not len(np.unique(df_stim['rate'])) == 3:
        bad_session = True
        print('ERROR!!')
        print('Not 3 unique train rates!')
    if sum(stim_overview_ses['stim'].isna()) > 0:
        bad_session = True
        print('ERROR!!')
        print('No info in excel sheet!')
    if stim_overview_ses.shape[0] == 0:
        bad_session = True
        print('ERROR!!')
        print('No info in excel sheet!')
    if bad_session:
        print('ERROR!!')
        return pd.DataFrame([])
    else:
        
        df_stim['subj_idx'] = subj
        df_stim['session'] = session
        df_stim['cuff_type'] = cuff_type
        df_stim['width'] = np.round(df_stim['width']*1000,2)
        df_stim['rate'] = df_stim['rate'].astype(int)
        
        df_stim['ones'] = 1
        counts = df_stim.groupby(['width', 'rate'])['ones'].sum()

        # make bins!!
        df_stim['width_bin'] = df_stim.groupby(['rate'])['width'].apply(pd.qcut, q=4, labels=False)
        for b in np.unique(df_stim['width_bin']):
            df_stim.loc[df_stim['width_bin']==b, 'width'] = round(df_stim.loc[df_stim['width_bin']==b, 'width'].median(), 4)
        # check if workable session:
        if not len(np.unique(df_stim['width'])) == 4:
            bad_session = True
            print('ERROR!!')
            print('Not 4 unique train widths!')
            return pd.DataFrame([])
        df_stim['rate_bin'] = df_stim.groupby(['width'])['rate'].apply(pd.qcut, q=3, labels=False)
        df_stim['amplitude_m_rank'] = df_stim['amplitude_m'].rank(method='first') # takes care of repeating values...
        df_stim['amplitude_m_bin'] = df_stim.groupby(['width', 'rate'])['amplitude_m_rank'].apply(pd.qcut, q=5, labels=False,)
        
        # step 1 -- determine leak fraction:
        widths = np.unique(df_stim['width'])
        ind = (df_stim['width']==widths[-2])|(df_stim['width']==widths[-1])
        d = df_stim.loc[ind,:].groupby(['width', 'rate']).min().reset_index()
        df_stim['amplitude_i_min'] = float(stim_overview_ses['stim'].iloc[0][0:3])
        df_stim['amplitude_i_max'] = float(stim_overview_ses['stim'].iloc[0][-3:])
        df_stim['amplitude_m_mean'] = d['amplitude_m'].mean()
        df_stim['leak'] = 1 - (df_stim['amplitude_m_mean']/df_stim['amplitude_i_min'])

        # step 2 -- fit logistic functions:
        X = np.array(df_stim.groupby(['width']).mean().reset_index()['width'])
        y = np.array(df_stim.groupby(['width']).mean().reset_index()['amplitude_m'])
        try:
            popt, pcov = curve_fit(fsigmoid, X, y, method='dogbox', bounds=([2, -0.75, y.max()/2],[20, 0.75, y.max()*2]))
            fit_success = True
        except:
            popt = np.array([10000,0,y.mean()])
            fit_success = False

        # print(fit_success)
        # print(popt)
        # x = np.linspace(-1, 1, 1000)
        # y_ = fsigmoid(x, 20, 0, 0.5)
        # plt.plot(X, y, 'o', label='data')
        # plt.plot(x,y_)
        # plt.show()
        
        popts = []
        for b in np.unique(df_stim['amplitude_m_bin']):
            if fit_success:
                _popt, _pcov = curve_fit(lambda x, s: fsigmoid(x, popt[0], popt[1], s),
                                            np.array(df_stim.loc[df_stim['amplitude_m_bin']==b,:].groupby(['width']).mean().reset_index()['width']), 
                                            np.array(df_stim.loc[df_stim['amplitude_m_bin']==b,:].groupby(['width']).mean().reset_index()['amplitude_m']))
            else:
                _popt = np.array([np.array(df_stim.loc[df_stim['amplitude_m_bin']==b,:].groupby(['width']).mean().reset_index()['amplitude_m']).mean()])
            popts.append(_popt)

        # step 3 -- correct for leak and filter:
        df_stim['amplitude_c'] = 0
        for i, b in enumerate(np.unique(df_stim['amplitude_m_bin'])):
            for j, w in enumerate(np.unique(df_stim.loc[df_stim['amplitude_m_bin']==b, 'width'])):
                ind = (df_stim['amplitude_m_bin']==b)&(df_stim['width']==w)
                df_stim.loc[ind, 'amplitude_c'] = df_stim.loc[ind, 'amplitude_m'] + fsigmoid(widths[-1], popt[0], popt[1], popts[i]) - fsigmoid(w, popt[0], popt[1], popts[i])
        df_stim.loc[:, 'amplitude_c'] = df_stim.loc[:, 'amplitude_c'] / (1 - df_stim['leak'].iloc[0])
        
        # step 4 -- compute saturation:
        df_stim['saturation'] = df_stim['amplitude_i_max'].iloc[0] - df_stim.loc[df_stim['amplitude_m_bin']==max(df_stim['amplitude_m_bin']), 'amplitude_c'].mean()

        # normalize before plotting:
        df_stim['amplitude_c_norm'] = 0
        for i, b in enumerate(np.unique(df_stim['amplitude_m_bin'])):
            df_stim.loc[df_stim['amplitude_m_bin']==b, 'amplitude_c_norm'] = df_stim.loc[df_stim['amplitude_m_bin']==b, 'amplitude_c'] / df_stim.loc[df_stim['amplitude_m_bin']==b, 'amplitude_c'].mean()

    fig = plot_param_preprocessing(df_stim, popt, popts)
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'stimulation_params_{}_{}.png'.format(subj, session)))

    return df_stim

def process_walk_data(file_tdms, fs_resample='2L'):

    # # load stim data:
    # df_stim, frames, pulses, stim_si = load_stim_parameters(file_meta)
    # frame_times = pulses * stim_si

    # load velocity:
    df_walk = utils.get_velocity_tdms(file_tdms)
    
    # correct:
    df_walk['velocity'] = df_walk['velocity'] * (5 / 1000) / df_walk['time'].diff()
    df_walk.loc[np.isnan(df_walk['velocity']), 'velocity'] = 0
    df_walk['distance'] = (df_walk['velocity'] * df_walk['time'].diff()).cumsum() 
    df_walk.loc[np.isnan(df_walk['distance']), 'distance'] = 0

    only_zeros = (np.mean(df_walk['velocity']==0) == 1)

    # resample to 500 Hz:
    index = pd.to_datetime(df_walk['time'], unit='s')
    df_walk = df_walk.loc[:,['velocity', 'distance']]
    df_walk = df_walk.set_index(index)
    df_walk = df_walk.resample(fs_resample).mean().interpolate('linear').reset_index()

    # df_walk = df_walk.resample('2L').fillna('pad').reset_index()
    # df_walk = df_walk.fillna(method='backfill') 
    df_walk['time'] = pd.to_timedelta(df_walk['time']).dt.total_seconds() 

    # df_walk['velocity_lp'] = preprocess_pupil._butter_lowpass_filter(data=df_walk['velocity'], highcut=1, fs=fs, order=3)

    if only_zeros:
        df_walk['velocity'] = np.NaN
        df_walk['distance'] = np.NaN

    return df_walk
    
def process_eye_data(file_meta, file_tdms, file_pupil, subj, ses, fig_dir, use_dlc_blinks=False, fs_resample='2L'):

    # load stim data:
    if file_meta.split('.')[-1] == 'txt':
        frames = np.loadtxt(file_meta).astype(int)
    else:
        df_stim, frames, pulses, stim_si = load_stim_parameters(file_meta)
    
    # frames must be in ascending order:
    frames = frames[np.concatenate((np.array([True]), np.diff(frames)>=0))]

    # open eye data:
    df_eye = pd.read_hdf(file_pupil)
    df_eye = df_eye.loc[:,['pupil', 'eyelid', 'pupil_x', 'pupil_y', 'is_blink_dlc']]

    # # add timestamps:
    # if frames[-1] > pulses.shape[0]:
    #     pulses = pulses[frames[frames<pulses.shape[0]]]
    # else:
    #     try:
    #         pulses = pulses[frames]
    #     except:
    #         print('ERROR!!')
    #         print('Alligenment issue!')
    #         return pd.DataFrame([])
    # frame_times = pulses * stim_si
    # frame_times = np.linspace(frame_times[0], frame_times[-1], frame_times.shape[0])

    # get eye timestamps:
    eye_timestamps = utils.get_pupil_timestamps_tdms(file_tdms)

    if len(eye_timestamps) == 0:

        fs = utils.get_frame_rate(glob.glob(os.path.join(os.path.dirname(file_tdms), '*.mp4'))[0])
        eye_timestamps = pd.DataFrame({'time': np.linspace(0, df_eye.shape[0]/fs, df_eye.shape[0])})
        print('{} hrs of pupil data!'.format(float(eye_timestamps.iloc[-1]/60/60)))
        
    if (len(frames) / len(eye_timestamps)) < 0.05:
        return pd.DataFrame([])
    if frames[-1] > eye_timestamps.shape[0]:
        eye_timestamps = eye_timestamps.iloc[frames[frames<eye_timestamps.shape[0]]]
    else:
        try:
            eye_timestamps = eye_timestamps.iloc[frames]
        except:
            print('ERROR!!')
            print('Alligenment issue!')
            return pd.DataFrame([])
    # frame_times = pulses * stim_si
    # frame_times = np.linspace(frame_times[0], frame_times[-1], frame_times.shape[0])

    # force to be same length:
    df_eye = df_eye.iloc[:eye_timestamps.shape[0],:]
    eye_timestamps = eye_timestamps[:df_eye.shape[0]]
    
    # add time:
    df_eye['time'] = np.array(eye_timestamps['time'])

    # resample:
    index = pd.to_datetime(np.array(df_eye['time']), unit='s') 
    df_eye = df_eye.set_index(index)
    df_eye = df_eye.resample(fs_resample).mean().interpolate('linear').reset_index()
    # df_eye = df_eye.resample('2L').fillna('pad').reset_index()
    # df_eye = df_eye.fillna(method='backfill') 
    df_eye['time'] = pd.to_timedelta(df_eye['index']).dt.total_seconds() 
    df_eye = df_eye.drop(labels=['index'], axis=1)

    # preprocess pupil:
    if fs_resample == '20L':
        fs = 50
    elif fs_resample == '2L':
        fs = 500
    print(fs)
    
    blink_measure = 'pupil'
    blink_cutoff = 30

    for measure in ['pupil', 'eyelid']:
        preprocess_pupil.interpolate_blinks(df=df_eye, fs=fs, measure=measure, 
                                            blink_detection_measures=[blink_measure], cutoffs=[blink_cutoff], 
                                            coalesce_period=0.75, buffer=0.15, use_dlc_blinks=use_dlc_blinks)
        preprocess_pupil.temporal_filter(df=df_eye, measure='{}_int'.format(measure), fs=fs, hp=0.1, lp=3, order=3)
        preprocess_pupil.fraction(df=df_eye, measure='{}_int_lp'.format(measure))
        preprocess_pupil.slope(df=df_eye, measure='{}_int_lp_frac'.format(measure))

    # preprocess xy:
    try:
        preprocess_pupil.interpolate_blinks(df=df_eye, fs=fs, measure='pupil_x', 
                                            blink_detection_measures=[blink_measure], cutoffs=[blink_cutoff], 
                                            coalesce_period=0.75, buffer=0.15, use_dlc_blinks=use_dlc_blinks)
        preprocess_pupil.interpolate_blinks(df=df_eye, fs=fs, measure='pupil_y', 
                                            blink_detection_measures=[blink_measure], cutoffs=[blink_cutoff],
                                            coalesce_period=0.75, buffer=0.15, use_dlc_blinks=use_dlc_blinks)
        
        df_eye['x_int_lp'] = preprocess_pupil._butter_lowpass_filter(data=df_eye['pupil_x_int'], highcut=10, fs=fs, order=3)
        df_eye['y_int_lp'] = preprocess_pupil._butter_lowpass_filter(data=df_eye['pupil_y_int'], highcut=10, fs=fs, order=3)
        df_eye['x_z'] = (df_eye['x_int_lp']-df_eye['x_int_lp'].mean()) / df_eye['x_int_lp'].std()
        df_eye['y_z'] = (df_eye['y_int_lp']-df_eye['y_int_lp'].mean()) / df_eye['y_int_lp'].std()
    except:
        pass

    fig = preprocess_pupil.plot_preprocessing(df_eye)
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'pupil_data_{}_{}.png'.format(subj, ses)))

    # shell()

    # measure = 'pupil'
    # ts = (np.array(df_eye[measure]) - np.median(df_eye[measure])) / np.std(df_eye[measure]) # z-score
    # ts = np.concatenate(( np.array([0]), np.diff(ts) )) / (1/fs)  # z-score / s

    return df_eye

def analyse_baseline_session(file_meta, file_pupil, file_tdms, fig_dir, subj, cuff_type, fs_resample='20L'):
    
    # session:
    ses = file_meta.split('/')[-1].split('_')[1]
    import re
    ses = re.sub('[OTVNS]', '', ses)

    # print:
    print()
    print((subj, ses))

    ### GET META DATA ###
    df_stim, frames, pulses, stim_si = load_stim_parameters(file_meta)
    if df_stim.shape[0] == 0:
        return [pd.DataFrame([]) for _ in range(7)]
    df_stim['subj_idx'] = subj
    df_stim['session'] = ses
    df_stim['cuff_type'] = cuff_type
    df_stim['width'] = np.round(df_stim['width']*1000,2)
    df_stim['rate'] = df_stim['rate'].astype(int)
    df_stim['amplitude_m_bin'] = pd.qcut(df_stim['amplitude_m'], 1, labels=False)
    df_stim['width_bin'] = pd.qcut(df_stim['width'], 1, labels=False)
    df_stim['rate_bin'] = pd.qcut(df_stim['rate'], 1, labels=False)
    
    ### GET WALKING DATA ###
    df_walk = process_walk_data(file_tdms, fs_resample=fs_resample)

    ### GET EYE DATA ###
    df_eye = process_eye_data(file_meta, file_tdms, file_pupil, subj, ses, fig_dir, fs_resample=fs_resample)
    if df_eye.shape[0] == 0:
        return [pd.DataFrame([]) for _ in range(7)]

    # make epochs:
    fs = 50
    epochs_v = utils.make_epochs(df_walk, df_stim, locking='time', start=-60, dur=120, measure='distance', fs=fs)
    epochs_p = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='pupil_int_lp_frac', fs=fs)
    epochs_l = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='eyelid_int_lp_frac', fs=fs)
    epochs_x = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='x_z', fs=fs)
    epochs_y = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='y_z', fs=fs)
    epochs_b = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='is_blink', fs=fs)
    
    xmax = min((max(df_eye['time']/60), max(df_walk['time']/60)))

    fig = plt.figure(figsize=(24,18))
    
    row = 0
    for measure in ['pupil_int_lp_frac', 'eyelid_int_lp_frac', 'x_z', 'y_z', 'is_blink', 'velocity', 'distance',]:
        
        if ('velocity' in measure) | ('distance' in measure):
            df = df_walk.copy()
        else:
            df = df_eye.copy()

        ax = plt.subplot2grid((7, 6), (row, 0), colspan=4)
        ax.plot(df['time']/60, df[measure]) 
        for p in np.array(df_stim['time']/60):
            ax.axvline(p, color='r', lw=0.5)
        plt.xlim(left=0, right=xmax)
        plt.ylabel(measure)
        ax = plt.subplot2grid((7, 6), (row, 4))
        if ('velocity' in measure) | ('distance' in measure):
            try:
                ax.hist(df.loc[df[measure]>0.01, measure], bins=100, orientation="horizontal")
            except:
                pass
        else:
            try:
                ax.hist(df[measure], bins=100, orientation="horizontal")
            except:
                pass
        ax = plt.subplot2grid((7, 6), (row, 5))

        for i, p in enumerate(df_stim['time']):
            ind = (df['time']>(p-10)) & (df['time']<(p+20))
            resp = np.array(df.loc[ind, measure])
            if (measure == 'pupil_int_lp_frac') | (measure == 'eyelid_int_lp_frac') | (measure == 'x_z') | (measure == 'y_z') | (measure == 'distance'):
                if (measure == 'distance'):
                    ind_b = (df['time']>=(p-0.1)) & (df['time']<=(p+0.1))
                else:
                    ind_b = (df['time']>=(p-7.5)) & (df['time']<=(p))
                baseline = float(df.loc[ind_b, measure].mean())
                resp = resp - baseline
            x = np.linspace(-10,20,len(resp))
            color = sns.dark_palette("red", 5)[int(df_stim['amplitude_m_bin'].iloc[i])]
            if (measure == 'pupil_int_lp_frac') | (measure == 'eyelid_int_lp_frac') | (measure == 'x_z') | (measure == 'y_z'):
                if df.loc[ind, 'is_blink'].mean() < 0.1:
                    plt.plot(x[::10], resp[::10], color=color, lw=0.5, alpha=0.25)
            else:
                plt.plot(x[::10], resp[::10], color=color, lw=0.5, alpha=0.25)
        ax.axvline(0, color='r', lw=0.5)
        
        row += 1
    
    sns.despine(trim=False, offset=2)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'session_data_{}_{}.pdf'.format(subj, ses)))
    
    plt.close('all')

    return df_stim, epochs_v, epochs_p, epochs_l, epochs_x, epochs_y, epochs_b

def analyse_exploration_session(file_meta, file_pupil, file_tdms, fig_dir, subj, cuff_type, stim_overview, fs_resample='20L'):

    # session:
    ses = file_meta.split('/')[-1].split('_')[1]
    import re
    ses = re.sub('[OTVNS]', '', ses)

    # print:
    print()
    print((subj, ses))

    ### GET META DATA ###
    df_stim = process_meta_data(file_meta, stim_overview, cuff_type, subj, ses, fig_dir)
    if df_stim.shape[0] == 0:
        return [pd.DataFrame([]) for _ in range(7)]

    ### GET WALKING DATA ###
    df_walk = process_walk_data(file_tdms, fs_resample=fs_resample)

    ### GET EYE DATA ###
    df_eye = process_eye_data(file_meta, file_tdms, file_pupil, subj, ses, fig_dir, fs_resample=fs_resample)
    if df_eye.shape[0] == 0:
        return [pd.DataFrame([]) for _ in range(7)]

    # make epochs:
    fs = 50
    epochs_v = utils.make_epochs(df_walk, df_stim, locking='time', start=-60, dur=120, measure='distance', fs=fs)
    epochs_p = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='pupil_int_lp_frac', fs=fs)
    epochs_l = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='eyelid_int_lp_frac', fs=fs)
    epochs_x = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='x_z', fs=fs)
    epochs_y = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='y_z', fs=fs)
    epochs_b = utils.make_epochs(df_eye, df_stim, locking='time', start=-60, dur=120, measure='is_blink', fs=fs)

    xmax = min((max(df_eye['time']/60), max(df_walk['time']/60)))

    fig = plt.figure(figsize=(24,18))
    
    row = 0
    for measure in ['pupil_int_lp_frac', 'eyelid_int_lp_frac', 'x_z', 'y_z', 'is_blink', 'velocity', 'distance',]:
        
        if ('velocity' in measure) | ('distance' in measure):
            df = df_walk.copy()
        else:
            df = df_eye.copy()

        ax = plt.subplot2grid((7, 6), (row, 0), colspan=4)
        ax.plot(df['time']/60, df[measure]) 
        for p in np.array(df_stim['time']/60):
            ax.axvline(p, color='r', lw=0.5)
        plt.xlim(left=0, right=xmax)
        plt.ylabel(measure)
        ax = plt.subplot2grid((7, 6), (row, 4))
        if ('velocity' in measure) | ('distance' in measure):
            try:
                ax.hist(df.loc[df[measure]>0.01, measure], bins=100, orientation="horizontal")
            except:
                pass
        else:
            try:
                ax.hist(df[measure], bins=100, orientation="horizontal")
            except:
                pass
        ax = plt.subplot2grid((7, 6), (row, 5))

        for i, p in enumerate(df_stim['time']):
            ind = (df['time']>(p-10)) & (df['time']<(p+20))
            resp = np.array(df.loc[ind, measure])
            if (measure == 'pupil_int_lp_frac') | (measure == 'eyelid_int_lp_frac') | (measure == 'x_z') | (measure == 'y_z') | (measure == 'distance'):
                if (measure == 'distance'):
                    ind_b = (df['time']>=(p-0.1)) & (df['time']<=(p+0.1))
                else:
                    ind_b = (df['time']>=(p-7.5)) & (df['time']<=(p))
                baseline = float(df.loc[ind_b, measure].mean())
                resp = resp - baseline
            x = np.linspace(-10,20,len(resp))
            color = sns.dark_palette("red", 5)[int(df_stim['amplitude_m_bin'].iloc[i])]
            if (measure == 'pupil_int_lp_frac') | (measure == 'eyelid_int_lp_frac') | (measure == 'x_z') | (measure == 'y_z'):
                if df.loc[ind, 'is_blink'].mean() < 0.1:
                    plt.plot(x[::10], resp[::10], color=color, lw=0.5, alpha=0.25)
            else:
                plt.plot(x[::10], resp[::10], color=color, lw=0.5, alpha=0.25)
        ax.axvline(0, color='r', lw=0.5)
        
        row += 1
    
    sns.despine(trim=False, offset=2)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'session_data_{}_{}.pdf'.format(subj, ses)))
    
    plt.close('all')

    # # save stuff:
    # leak_cut_off = 0.25
    # if (df_stim['session_amp_leak'].iloc[0]>=0)&(df_stim['session_amp_leak'].iloc[0]<leak_cut_off):
    #     dd = pd.merge(df_walk, df_eye.loc[:,['time', 'pupil', 'pupil_int_lp_frac']], on='time')
    #     dd = dd.rename(columns={'pupil_int_lp_frac': 'pupil'})
    #     dd.to_csv('/media/external2/forMatt/{}_{}.csv'.format(subj, ses))
    
    return df_stim, epochs_v, epochs_p, epochs_l, epochs_x, epochs_y, epochs_b

def analyse_imaging_session(raw_dir, imaging_dir, fig_dir, subj, ses, fs_resample='20L'):

    # get image timestamps:
    file_tdms = [fn for fn in glob.glob(os.path.join(raw_dir, subj, ses, '*.tdms')) if not 'h264' in fn][0]
    file_frames = [fn for fn in glob.glob(os.path.join(raw_dir, subj, ses, '*frame*.txt')) if not 'h264' in fn][0]
    file_pupil = [fn for fn in glob.glob(os.path.join(raw_dir, subj, ses, 'pupil_preprocess', '*.hdf'))][0]
    file_tiff = [fn for fn in glob.glob(os.path.join(raw_dir, subj, ses, '*.tif')) if not 'h264' in fn][0]

    ### get time:
    time = utils.get_image_timestamps_tdms(file_tdms)

    ### get pulses:
    df_stim = utils.get_vns_timestamps_tdms(file_tdms)
    df_stim['width'] = df_stim['width'].round(1)
    df_stim['rate'] = df_stim['rate'].astype(int)+1
    df_stim['amplitude_bin'] = pd.cut(df_stim['amplitude'], bins=np.array([0,0.2,0.4,0.6,0.8,1]), labels=False,)
    df_stim['amplitude'] = np.round((df_stim['amplitude_bin']*0.2)+0.1,1)
    
    ### GET WALKING DATA ###
    df_walk = process_walk_data(file_tdms, fs_resample=fs_resample)

    ### GET EYE DATA ###
    df_eye = process_eye_data(file_frames, file_tdms, file_pupil, subj, ses, fig_dir, use_dlc_blinks=False, fs_resample=fs_resample)
    # if df_eye.shape[0] == 0:
    #     print()
    #     print(subj, ses)
    #     print("THIS IS REALLY BAD!! == WE'RE LOOSING A SESSION DUE TO EYE DATA....!")
    #     print()
    #     return [pd.DataFrame([]) for _ in range(6)]

    # fig = plt.figure()   
    # ax = fig.add_subplot(211)
    # plt.plot(df_eye.time, df_eye.eyelid)  
    # ax = fig.add_subplot(212)
    # plt.plot(df_walk.time, df_walk.velocity)  

    ### get zoom factor:
    zoom = utils.get_image_zoom_factor(file_tiff)
    px_per_micron = 0.0535+0.44*zoom

    ### get suite2p output:
    if not os.path.exists(os.path.join(imaging_dir, subj, ses, 'suite2p', 'ops1.npy')):
        print('ERROR2!')
        return [pd.DataFrame([]) for _ in range(6)]
    ops = np.load(os.path.join(imaging_dir, subj, ses, 'suite2p', 'ops1.npy'))[0]
    
    # update time:
    if time.shape[0] > ops['nframes']:
        time = time.iloc[:ops['nframes'],:].reset_index(drop=True)
    
    # get image motion information
    df_motion = pd.DataFrame({
                        'xoff': ops['xoff'] / px_per_micron,
                        'yoff': ops['yoff'] / px_per_micron,
                        'corrXY': ops['corrXY'],
                        'badframes': ops['badframes'],})
    df_motion = df_motion.iloc[0:len(time),:]
    df_motion['time'] = np.array(time)

    # resample:
    index = pd.to_datetime(np.array(df_motion['time']), unit='s') 
    df_motion = df_motion.set_index(index)
    df_motion = df_motion.resample(fs_resample).mean().interpolate('linear').reset_index()
    df_motion['time'] = pd.to_timedelta(df_motion['index']).dt.total_seconds() 
    df_motion = df_motion.drop(labels=['index'], axis=1)

    # make epochs:
    fs = 50
    df_stim['time2'] = df_stim['time']-5
    epochs_x = utils.make_epochs(df=df_motion, df_meta=df_stim, locking='time', start=-10, dur=30, measure='xoff', fs=fs,)
    epochs_y = utils.make_epochs(df=df_motion, df_meta=df_stim, locking='time', start=-10, dur=30, measure='yoff', fs=fs,)
    epochs_w = utils.make_epochs(df=df_walk, df_meta=df_stim, locking='time', start=-10, dur=30, measure='distance', fs=fs,)
    epochs_v = utils.make_epochs(df_walk, df_stim, locking='time', start=-60, dur=120, measure='distance', fs=fs)
    epochs_p = utils.make_epochs(df_eye, df_stim, locking='time2', start=-60, dur=120, measure='pupil_int_lp_frac', fs=fs)
    epochs_l = utils.make_epochs(df_eye, df_stim, locking='time2', start=-60, dur=120, measure='eyelid_int_lp_frac', fs=fs)
    epochs_b = utils.make_epochs(df_eye, df_stim, locking='time2', start=-60, dur=120, measure='is_blink', fs=fs)

    # fluorescence:
    f = np.load(os.path.join('/media/internal1/vns/', subj, ses, 'F.npy'))
    t = np.array(time['time'])
    if len(f) > len(t):
        f = f[:len(t)]
    fluorescence = pd.DataFrame({'F':f})
    fluorescence['time'] = t

    # resample:
    index = pd.to_datetime(np.array(fluorescence['time']), unit='s') 
    fluorescence = fluorescence.set_index(index)
    fluorescence = fluorescence.resample(fs_resample).mean().interpolate('linear').reset_index()
    fluorescence['time'] = pd.to_timedelta(fluorescence['index']).dt.total_seconds() 
    fluorescence = fluorescence.drop(labels=['index'], axis=1)

    # preprocess:
    fluorescence['F2'] = utils._butter_lowpass_filter(fluorescence['F'], highcut=3, fs=fs, order=3)
    fluorescence['F2'] = ((fluorescence['F2'] - np.percentile(fluorescence['F2'], 0.01)) / 
                            (np.percentile(fluorescence['F2'], 99.9) - np.percentile(fluorescence['F2'], 0.01)))

    x = fluorescence['time']
    f = fluorescence['F2']

    # (subj == 'C1772') & (session == 7)
    # (subj == 'C1773') & (session == 6)
    # (subj == 'C1773') & (session == 8)

    # linear detrend:
    func = linear
    popt, pcov = curve_fit(func, x, f,)
    f_filt = f - func(x, *popt)
    scaler_orig = func(x, *popt)
    scaler = (max(scaler_orig) - scaler_orig) / max(scaler_orig)

    # # highpass detrend:
    # f_filt = utils._butter_highpass_filter(f, lowcut=0.001, fs=fs, order=3)
    # scaler_orig = f-f_filt
    # scaler = (max(scaler_orig) - scaler_orig) / max(scaler_orig)

    f_corrected = f_filt + (f_filt * scaler)
    f_corrected = utils._butter_lowpass_filter(f_corrected, highcut=6, fs=fs, order=3)
    f_corrected = ((f_corrected - np.percentile(f_corrected, 0.01)) / 
                            (np.percentile(f_corrected, 99.9) - np.percentile(f_corrected, 0.01)))
    fluorescence['F'] = f_corrected

    fig = plt.figure(figsize=(6,9))
    ax = fig.add_subplot(311)
    plt.plot(x,f)
    plt.plot(x, scaler_orig, color='k', ls='-', zorder=10)
    ax = fig.add_subplot(312)
    plt.plot(x,f_filt)
    ax = fig.add_subplot(313)
    plt.plot(x,f_corrected)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'preprocess_calcium_{}_{}.png'.format(subj, ses)), dpi=300)

    # epochs:
    epochs_c = utils.make_epochs(df=fluorescence, df_meta=df_stim, locking='time', start=-10, dur=30, measure='F', fs=fs,)
    # epochs_c = pd.read_hdf(os.path.join('/media/internal1/vns/', subj, ses, 'epochs_c.hdf'))
    epochs_c['cell'] = np.ones(epochs_c.shape[0])
    epochs_c['amplitude'] = np.array(df_stim['amplitude'])
    epochs_c['rate'] = np.array(df_stim['rate'])
    epochs_c['width'] = np.array(df_stim['width'])
    epochs_c['subj_idx'] = subj
    epochs_c['session'] = ses
    epochs_c['trial'] = np.arange(epochs_c.shape[0]) 
    epochs_c = epochs_c.set_index(['subj_idx', 'session', 'trial', 'cell', 'amplitude', 'rate', 'width'])

    # summary figure: 
    fig = preprocess_calcium.fig_image_quality(ops['refImg'], ops['meanImg'], df_motion.iloc[::10], fluorescence.iloc[::10], df_eye.iloc[::10], df_walk.iloc[::10], zoom, subj, ses, df_stim)
    fig.savefig(os.path.join(fig_dir, 'preprocess', 'preprocess_{}_{}_a.pdf'.format(subj, ses)), dpi=300)

    # fig = plt.figure(figsize=(4,2))
    # ax = fig.add_subplot(121)
    # ax.scatter(image_pos_x, image_pos_y)
    # ax.set_xlabel('Image position x')
    # ax.set_ylabel('Image position y')
    # ax = fig.add_subplot(122)
    # ax.scatter(image_motion_x, image_motion_y)
    # ax.set_xlabel('Image motion x')
    # ax.set_ylabel('Image motion y')
    # plt.tight_layout()
    # fig.savefig(os.path.join(fig_dir, 'preprocess', 'preprocess_{}_{}_b.pdf'.format(subj, ses)), dpi=300)

    # d = pd.DataFrame({
    #     'x_shift': abs(ops['xoff']),
    #     'y_shift': abs(ops['yoff']),
    #     'corr': abs(ops['corrXY']),
    # })
    # d.to_csv(os.path.join(fig_dir, 'preprocess', 'preprocess_{}_{}.csv'.format(subj, ses)))

    # inds = ((df_stim['time'] - float(time.iloc[0])) * fs).astype(int)
    # inds.to_csv(os.path.join(fig_dir, 'preprocess', 'preprocess_{}_{}_indices.csv'.format(subj, ses)))

    # # cells:
    # iscell = np.load(os.path.join(imaging_dir, subj, ses, 'suite2p', 'plane0', 'iscell.npy'))
    # cell_indices = (iscell[:,0] == 1)
    
    # # if sum(iscell[:,0]) > 0:
        
    #     # load data:
    #     fluorescence = pd.DataFrame(np.load(os.path.join(imaging_dir, subj, ses, 'suite2p', 'plane0', 'F.npy'))[cell_indices,:][:,:len(time)])
    #     neuropil = pd.DataFrame(np.load(os.path.join(imaging_dir, subj, ses, 'suite2p', 'plane0', 'Fneu.npy'))[cell_indices,:][:,:len(time)])
    #     fluorescence = fluorescence - (0.7 * neuropil)

    #     # spikes = pd.DataFrame(np.load(os.path.join(imaging_dir, subj, ses, 'suite2p', 'plane0', 'spks.npy'))[cell_indices,:][:,:len(time)])
    #     time = time[:fluorescence.shape[1]]

    #     # add timestamps:
    #     fluorescence.columns = time.values.ravel() 
    #     # spikes.columns = time.values.ravel() 
    #     # neuropil.columns = time.values.ravel() 

    #     # filter:
    #     # fluorescence = fluorescence.rolling(window=3, center=True, min_periods=1, axis='columns').median()
    #     # fluorescence = fluorescence.rolling(window=5, center=True, min_periods=1, axis='columns').mean()
    #     for i in range(fluorescence.shape[0]):
    #         fluorescence.iloc[i] = utils._butter_lowpass_filter(fluorescence.iloc[i], highcut=1, fs=fs, order=3)

    #     # fnorm:
    #     fluorescence.loc[:,:] = ((fluorescence.values - np.atleast_2d(np.min(fluorescence.values, axis=1)).T) / 
    #                         (np.atleast_2d(np.percentile(fluorescence.values, 99, axis=1)).T - np.atleast_2d(np.min(fluorescence.values, axis=1)).T))

    #     # first check for activity (excluding bad frames) --> power of asynchronous sampled
        
    #     # second correlation pooling  (excluding bad frames):
    #     badframes = ops['badframes'][:len(time)]
    #     fluorescence_clustered = hierarchical_cluster(fluorescence.T, min_r=0.75, include=~badframes)
    #     fig = plot_corr(fluorescence_clustered.corr())
    #     fig.savefig(os.path.join(fig_dir, 'preprocess', 'preprocess_{}_{}_c.pdf'.format(subj, ses)), dpi=300)

    #     # take mean per cluster:
    #     fluorescence = fluorescence_clustered.groupby(by=fluorescence_clustered.columns, axis=1).mean().T
        
    #     # check if axon has sufficient SNR (check supplement)
        
    #     # make epochs:
    #     epochs = []
    #     for i in range(fluorescence.shape[0]):
    #         df = fluorescence.iloc[i,:].reset_index()
    #         df.columns = ['time', 'F']
    #         epoch = utils.make_epochs(df=df, df_meta=df_stim, locking='time', start=-10, dur=30, measure='F', fs=fs,)
    #         epoch['cell'] = (np.ones(epoch.shape[0]) * i).astype(int)
    #         epoch['amplitude'] = np.array(df_stim['amplitude_b'])
    #         epoch['rate'] = np.array(df_stim['rate'])
    #         epoch['width'] = np.array(df_stim['width'])
    #         epoch['subj_idx'] = subj
    #         epoch['session'] = ses
    #         epoch['trial'] = np.arange(epoch.shape[0]) 
    #         epoch = epoch.set_index(['subj_idx', 'session', 'trial', 'cell', 'amplitude', 'rate', 'width'])
    #         epochs.append(epoch)
    #     epochs = pd.concat(epochs, axis=0)
    #     x = np.array(epochs.columns, dtype=float)

    #     # session figure:
    #     fig = plt.figure(figsize=(9,2+(fluorescence.shape[0]/2)))
        
    #     colors = sns.color_palette(n_colors=fluorescence.shape[0])
        
    #     gs = gridspec.GridSpec(1, 5)
    #     ax = plt.subplot(gs[0:1,0:4])
    #     offset = 0
    #     for i in range(fluorescence.shape[0]):
    #         ax.plot(time/60, fluorescence.iloc[i]+offset, alpha=0.75, color=colors[i])
    #         offset += np.percentile(np.array(fluorescence).ravel(), 99)
    #     for t in df_stim['time']:
    #         plt.axvline(t/60, lw=0.5, color='r')
    #     plt.ylabel('Fluorescence')
    #     plt.xlabel('Time (min)')
    #     ylim = ax.get_ylim() 

    #     ax = plt.subplot(gs[0:1,4:5])
    #     offset = 0
    #     for i in range(fluorescence.shape[0]):
    #         for p in df_stim['time']:
    #             ind = (fluorescence.columns>(p-10)) & (fluorescence.columns<(p+20))
    #             ind_b = (fluorescence.columns>(p-5.5)) & (fluorescence.columns<(p+0.5))
    #             resp = fluorescence.iloc[i,ind] - fluorescence.iloc[i,ind_b].mean()
    #             x = np.linspace(-10,20,len(resp))
    #             ax.plot(x, resp+offset, lw=0.5, alpha=1, color=colors[i])
    #         offset += np.percentile(np.array(fluorescence).ravel(), 99)
    #     plt.axvline(0, lw=0.5, color='r')
    #     ax.set_ylim(ylim)
    #     plt.ylabel('Fluorescence')
    #     plt.xlabel('Time (min)')
        
    #     sns.despine(trim=False, offset=3)
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(fig_dir, 'vns_{}_{}_session.pdf'.format(subj, ses)))
        
    return epochs_c, epochs_x, epochs_y, epochs_v, epochs_p, epochs_l, epochs_b


