import os, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import mne

from IPython import embed as shell

from tools_mcginley import utils


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
temp_dir = '/media/internal1/vns/'

run_stats = True

subjects = {
    'C7A2': ['1', '8'],
    'C7A6': ['1', '2', '3', '5', '6', '7', '8',],  
    'C1772': ['1', '5', '6', '7', '8',],
    'C1773': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    'D1767': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
}

subjects = {
    'C7A2': ['1',],
    'C7A6': ['1', '7', '8',],  
    'C1772': ['6', '7', '8',],
    'C1773': ['5', '6', '7', '8', '10',],
}

for subj in subjects.keys():
    for session in subjects[subj]:
        
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

        if run_stats:
            
            # load raw data:
            ix  = np.linspace(0,ops['nframes']-1,ops['nframes']).astype('int')
            mov = sample_frames(ops, ix, ops['reg_file'])
            mean_img = np.mean(mov[~bad_frames,:,:], axis=0)

            # create scalars:
            fs = 15
            scalars = []
            for ind in indices:
                print(ind)
                scalars.append( mov[int(ind+(1*fs)):int(ind+(9*fs)),:,:].mean(axis=0) - mov[int(ind-(5*fs)):ind,:,:].mean(axis=0) )
            scalars = np.array(scalars)

            # statisitcs and save mask:
            from functools import partial
            from mne.stats import (ttest_1samp_no_p, bonferroni_correction, fdr_correction,
                    permutation_t_test, permutation_cluster_1samp_test)

            n_samples = scalars.shape[0]
            alpha = 0.01
            threshold = sp.stats.distributions.t.ppf(1 - alpha, n_samples - 1)
            threshold_tfce = dict(start=0.1, step=0.5)
            sigma = 1e-3
            stat_fun_hat = partial(ttest_1samp_no_p, sigma=sigma)
            
            # t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(scalars[:,:,:], n_permutations=1024, n_jobs=4)
            t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(
                scalars[:,:,:], threshold=threshold, stat_fun=stat_fun_hat, 
                buffer_size=None, n_permutations=1024, n_jobs=8)

            # t_tfce_hat, _, p_tfce_hat, H0 = permutation_cluster_1samp_test(
            #     scalars[:,:,:], threshold=threshold_tfce, stat_fun=stat_fun_hat,
            #     buffer_size=None, n_permutations=1024, n_jobs=1)

            mask = np.zeros(t_obs.shape, dtype=bool)
            for i in range(len(clusters)):
                if cluster_pv[i] < alpha:
                    mask[clusters[i]] = True
            mask[(t_obs < 0)] = False 
            np.save(os.path.join(temp_directory, 'mask.npy'), mask)

            # save mean activity trace:
            f = mov[:,mask].mean(axis=1)
            np.save(os.path.join(temp_directory, 'F.npy'), f)

            # plot:
            for group in [0,1,2,3]:
                if group == 0:
                    ind = np.ones(scalars.shape[0], dtype=bool)
                else:
                    ind = np.array(df_pulse['group'] == group)
                print(sum(ind))
                # t_obs = sp.stats.ttest_1samp(scalars[ind,:,:], 0, axis=0)[0]
                for i in range(2):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    
                    from matplotlib.colors import LinearSegmentedColormap
                    # cmap = LinearSegmentedColormap.from_list('custom', ['blue', 'lightblue', 'lightgrey', 'yellow', 'red'], N=100)
                    cmap = LinearSegmentedColormap.from_list('custom', ['lightgrey', 'yellow', 'red'], N=100)

                    ax.pcolormesh(np.arange(mean_img.shape[1]), np.arange(mean_img.shape[0]), mean_img, 
                        vmin=np.percentile(mean_img.ravel(), 1), vmax=np.percentile(mean_img.ravel(), 99), cmap='Greys_r')
                    if i == 1:
                        ax.pcolormesh(np.arange(mean_img.shape[1]), np.arange(mean_img.shape[0]), np.ma.masked_array(t_obs, ~mask),
                        vmin=threshold, vmax=threshold*2, cmap=cmap)
                    ax.set_aspect('equal')
                    fig.savefig(os.path.join(temp_directory, 'meanImg_overlay_{}_{}.png'.format(group, i)), dpi=300)
                
                # gif:
                import imageio
                images = []
                for i in range(2):
                    images.append(imageio.imread(os.path.join(temp_directory, 'meanImg_overlay_{}_{}.png'.format(group, i))))
                imageio.mimsave(os.path.join(temp_directory, 'meanImg_overlay_{}.gif'.format(group)), images, duration=0.5)
        
        mask = np.load(os.path.join(temp_directory, 'mask.npy'))
        f = np.load(os.path.join(temp_directory, 'F.npy'))
        t = np.array(time['time'])
        if len(f) > len(t):
            f = f[:len(t)]
        df = pd.DataFrame({'time':t, 'F':f})
        epochs = utils.make_epochs(df=df, df_meta=df_pulse, locking='time', start=-60, dur=120, measure='F', fs=15,)
        epochs.to_hdf(os.path.join(temp_directory, 'epochs_f.hdf'), key='f')

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
        fig = plt.figure(figsize=(2,2))
        plt.fill_between(epochs.columns, epochs.mean(axis=0)-epochs.sem(axis=0), epochs.mean(axis=0)+epochs.sem(axis=0), alpha=0.2)
        plt.plot(np.array(epochs.columns), np.array(epochs.mean(axis=0)), lw=1)
        plt.xlabel('Time (s)')
        plt.ylabel('F (a.u.)')
        plt.axvline(0, color='r', lw=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(temp_directory, 'summary2.pdf'))