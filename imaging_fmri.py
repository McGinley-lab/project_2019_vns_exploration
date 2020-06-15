import os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
from scipy import stats
import mne
from joblib import Parallel, delayed
from tqdm import tqdm 
from functools import partial
from mne.stats import (ttest_1samp_no_p, bonferroni_correction, fdr_correction,
        permutation_t_test, permutation_cluster_1samp_test)

from IPython import embed as shell

from tools_mcginley import utils

def compute_snr(x, coord, fs=15):

    f, Pxx = sp.signal.welch(x, fs=fs)
    snr = np.log10(Pxx[(f>=0.05)&(f<=0.5)].mean() / Pxx[(f>=1)&(f<=3)].mean())
    return snr

def sample_frames(ops, ix, reg_file, crop=True):
    """ get frames ix from reg_file
        frames are cropped by ops['yrange'] and ops['xrange']

    Parameters
    ----------
    ops : dict
        requires 'yrange', 'xrange', 'Ly', 'Lx'
    ix : int, array
        frames to take
    reg_file : str
        location of binary file to read (frames x Ly x Lx)
    crop : bool
        whether or not to crop by 'yrange' and 'xrange'

    Returns
    -------
        mov : int16, array
            frames x Ly x Lx
    """
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread =  np.int64(Ly*Lx*2)
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    if crop:
        mov = np.zeros((len(ix), Lyc, Lxc), np.int16)
    else:
        mov = np.zeros((len(ix), Ly, Lx), np.int16)
    # load and bin data
    with open(reg_file, 'rb') as reg_file:
        for i in range(len(ix)):
            reg_file.seek(nbytesread*ix[i], 0)
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            data = np.reshape(data, (Ly, Lx))
            if crop:
                mov[i,:,:] = data[ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
            else:
                mov[i,:,:] = data
    return mov

raw_dir = '/media/external4/2p_imaging/vns/'
# temp_dir = '/media/internal1/vns/'
temp_dir = '/media/internal2/vns/'

run_stats = True

subjects = {
    # 'C7A2':  ['1',      '3', '4', '5', '6', '7', '8'],
    # 'C7A6':  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'C1772': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'C1773': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
}
for subj in subjects.keys():
    for session in subjects[subj]:
        
        try:

            print()
            print(subj)
            print(session)

            # directories:
            raw_directory = os.path.join(raw_dir, subj, session)
            temp_directory = os.path.join(temp_dir, subj, session)

            # load suite2p:
            ops = np.load(os.path.join(temp_directory, 'suite2p', 'ops1.npy'), allow_pickle=True)[0]
            reg_file = ops['raw_file']
            bad_frames = ops['badframes']
            # time = np.arange(ops['yoff'].shape[0])

            # load pulse timestamps:
            tdms_filename = [fn for fn in glob.glob(os.path.join(raw_directory, '*.tdms')) if not 'h264' in fn][0]
            time = utils.get_image_timestamps_tdms(tdms_filename)
            if time.shape[0] > ops['nframes']:
                time = time.iloc[:ops['nframes'],:].reset_index(drop=True)
            df_pulse = utils.get_vns_timestamps_tdms(tdms_filename)
            df_pulse['width'] = df_pulse['width'].round(1)
            df_pulse['rate'] = df_pulse['rate'].astype(int)
            df_pulse['amplitude_b'] = pd.cut(df_pulse['amplitude'], bins=np.array([0,0.2,0.4,0.6,0.8,1]), labels=False,)
            
            # splits:
            df_pulse['group'] = 0
            df_pulse.loc[(df_pulse['amplitude_b']==4)&(df_pulse['width']==0.4), 'group'] = 3
            df_pulse.loc[(df_pulse['amplitude_b']==4)&(df_pulse['width']==0.2), 'group'] = 3
            df_pulse.loc[(df_pulse['amplitude_b']==3)&(df_pulse['width']==0.4), 'group'] = 3
            df_pulse.loc[(df_pulse['amplitude_b']==4)&(df_pulse['width']==0.1), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==3)&(df_pulse['width']==0.2), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==3)&(df_pulse['width']==0.1), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==2)&(df_pulse['width']==0.4), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==2)&(df_pulse['width']==0.2), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==1)&(df_pulse['width']==0.4), 'group'] = 2
            df_pulse.loc[(df_pulse['amplitude_b']==2)&(df_pulse['width']==0.1), 'group'] = 1
            df_pulse.loc[(df_pulse['amplitude_b']==1)&(df_pulse['width']==0.2), 'group'] = 1
            df_pulse.loc[(df_pulse['amplitude_b']==1)&(df_pulse['width']==0.1), 'group'] = 1
            df_pulse.loc[(df_pulse['amplitude_b']==0)&(df_pulse['width']==0.4), 'group'] = 1
            df_pulse.loc[(df_pulse['amplitude_b']==0)&(df_pulse['width']==0.2), 'group'] = 1
            df_pulse.loc[(df_pulse['amplitude_b']==0)&(df_pulse['width']==0.1), 'group'] = 1

            # incides of vns pulses:
            indices = time['time'].searchsorted(df_pulse['time'])

            # params:
            fs = 15
            n_jobs = 32
            thresh = 0.05

            # load raw data:
            ix  = np.linspace(0,ops['nframes']-1,ops['nframes']).astype('int')
            mov = sample_frames(ops, ix, ops['reg_file'])
            mean_img = np.mean(mov[~bad_frames,:,:], axis=0)

            ## FUNCTIONAL SNR:
            if not os.path.exists(os.path.join(temp_directory, 'snr.npy')):

                coords = [(i,j) for i in range(mean_img.shape[0]) for j in range(mean_img.shape[1])]
                res = Parallel(n_jobs=n_jobs, verbose=0, backend='loky')(delayed(compute_snr)(mov[:,coord[0],coord[1]], coord, fs) for coord in tqdm(coords))
                snr = np.array(res).reshape(mean_img.shape)
                np.save(os.path.join(temp_directory, 'snr.npy'), snr)

            else:
                snr = np.load(os.path.join(temp_directory, 'snr.npy'))

            ## VNS-EVOKED ACTIVITY:

            if not (os.path.exists(os.path.join(temp_directory, 'mask.npy'))&os.path.exists(os.path.join(temp_directory, 't_obs.npy'))):

                # create scalars:
                scalars = []
                for ind in indices:
                    print(ind)
                    scalars.append( mov[int(ind+(1*fs)):int(ind+(9*fs)),:,:].mean(axis=0) - mov[int(ind-(5*fs)):ind,:,:].mean(axis=0) )
                scalars = np.array(scalars)

                # statisitcs and save mask:
                n_samples = scalars.shape[0]
                alpha = 0.01
                threshold = sp.stats.distributions.t.ppf(1 - alpha, n_samples - 1)
                threshold_tfce = dict(start=0.1, step=0.5)
                sigma = 1e-3
                stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
                
                # t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(scalars[:,:,:], n_permutations=1024, n_jobs=4)
                t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                    scalars[:,:,:], threshold=threshold, stat_fun=stat_fun_hat, 
                    buffer_size=None, n_permutations=1024, n_jobs=n_jobs)

                # t_tfce_hat, _, p_tfce_hat, H0 = permutation_cluster_1samp_test(
                #     scalars[:,:,:], threshold=threshold_tfce, stat_fun=stat_fun_hat,
                #     buffer_size=None, n_permutations=1024, n_jobs=1)

                mask = np.zeros(t_obs.shape, dtype=bool)
                for i in range(len(clusters)):
                    if cluster_pv[i] < alpha:
                        mask[clusters[i]] = True
                mask[(t_obs < 0)] = False 

                # save mean activity trace:
                f = mov[:,mask].mean(axis=1)
                
                # save:
                np.save(os.path.join(temp_directory, 'mask.npy'), mask)
                np.save(os.path.join(temp_directory, 't_obs.npy'), t_obs)
                np.save(os.path.join(temp_directory, 'F.npy'), f)

            else:

                mask = np.load(os.path.join(temp_directory, 'mask.npy'))
                t_obs = np.load(os.path.join(temp_directory, 't_obs.npy'))
                f = np.load(os.path.join(temp_directory, 'F.npy'))

            # PLOT:
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(231)
            cf = ax.pcolormesh(np.arange(snr.shape[1]), np.arange(snr.shape[0]), snr, 
                vmin=np.percentile(snr.ravel(), 1), vmax=np.percentile(snr.ravel(), 99), cmap='Greys_r', rasterized=True)
            ax.set_aspect('equal')
            ax.set_title('signal-to-noise')
            fig.colorbar(cf, ax=ax)
            ax = fig.add_subplot(232)
            plt.hist(snr.ravel(), bins=100)
            plt.axvline(thresh, color='r')
            ax = fig.add_subplot(233)
            cf = ax.pcolormesh(np.arange(snr.shape[1]), np.arange(snr.shape[0]), snr>thresh, 
                vmin=0, vmax=1, cmap='Greys_r', rasterized=True)
            ax.set_aspect('equal')
            fig.colorbar(cf, ax=ax)
            ax.set_title('{} pixels'.format((snr>thresh).sum()))
            ax = fig.add_subplot(234)
            cf = ax.pcolormesh(np.arange(mean_img.shape[1]), np.arange(mean_img.shape[0]), mean_img, 
                vmin=np.percentile(mean_img.ravel(), 1), vmax=np.percentile(mean_img.ravel(), 99), cmap='Greys_r', rasterized=True)
            ax.set_title('mean image')
            fig.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            ax = fig.add_subplot(235)
            ax.pcolormesh(np.arange(mean_img.shape[1]), np.arange(mean_img.shape[0]), mean_img, 
                vmin=np.percentile(mean_img.ravel(), 1), vmax=np.percentile(mean_img.ravel(), 99), cmap='Greys_r', rasterized=True)
            cf = ax.pcolormesh(np.arange(mean_img.shape[1]), np.arange(mean_img.shape[0]), np.ma.masked_array(t_obs, ~mask),
            vmin=threshold, vmax=7, alpha=1, cmap="summer_r", rasterized=True)
            # vmin=0, vmax=0.01, cmap=cmap)
            ax.set_title('{} pixels'.format(mask.sum()))
            fig.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            plt.tight_layout()
            fig.savefig(os.path.join(temp_directory, 'snr.pdf'), dpi=300)

            # offsets:
            x = ops['xoff']
            y = ops['yoff']
            t = np.array(time['time'])
            if len(f) > len(t):
                f = f[:len(t)]
                x = x[:len(t)]
                y = y[:len(t)]

            df = pd.DataFrame({'time':t, 'F':f, 'x':x, 'y':y})
            epochs = utils.make_epochs(df=df, df_meta=df_pulse, locking='time', start=-60, dur=120, measure='F', fs=15,)
            epochs.to_hdf(os.path.join(temp_directory, 'epochs_f.hdf'), key='f')

            epochs_x = utils.make_epochs(df=df, df_meta=df_pulse, locking='time', start=-60, dur=120, measure='x', fs=15, baseline=True, b_start=-1, b_dur=1)
            epochs_y = utils.make_epochs(df=df, df_meta=df_pulse, locking='time', start=-60, dur=120, measure='y', fs=15, baseline=True, b_start=-1, b_dur=1)
            
            motion_cutoff = 2
            x = np.array(epochs_x.columns, dtype=float)
            clean = sum( (epochs_x.loc[:,(x>2.5)&(x<7.5)].mean(axis=1) <= motion_cutoff) & (epochs_y.loc[:,(x>2.5)&(x<7.5)].mean(axis=1) <= motion_cutoff) )
            
            # plot 1:
            fig = plt.figure(figsize=(12,2))
            ax = fig.add_subplot(111)
            ax.plot(df['time'], df['F'], lw=1)
            for p in df_pulse['time']:
                plt.axvline(p, color='r', lw=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('F (a.u.)')
            plt.tight_layout()
            fig.savefig(os.path.join(temp_directory, 'summary.pdf'))
            
            # plot 2:
            for data, title in zip([epochs, epochs_x, epochs_y], ['F', 'X', 'Y']):
                fig = plt.figure(figsize=(2,2))
                plt.fill_between(data.columns, data.median(axis=0)-data.sem(axis=0), data.median(axis=0)+data.sem(axis=0), alpha=0.2)
                plt.plot(np.array(data.columns), np.array(data.median(axis=0)), lw=1)
                plt.xlabel('Time (s)')
                plt.ylabel('F (a.u.)')
                plt.axvline(0, color='r', lw=0.5)
                if (title=='X') or (title=='Y'):
                    plt.axhline(5, ls='--', color='r')
                plt.title('{}/{} clean trials'.format(clean, epochs.shape[0]), size=7)
                plt.tight_layout()
                fig.savefig(os.path.join(temp_directory, 'evoked_{}.pdf'.format(title)))

        except:
            pass