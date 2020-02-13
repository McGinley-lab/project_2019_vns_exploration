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
from vns_analyses import analyse_light_control_session

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

raw_dir = '/media/external1/raw/vns_light_control/'
project_dir = '/home/jwdegee/vns_exploration/'
fig_dir = os.path.join(project_dir, 'figures', 'light_control')
data_dir = os.path.join(project_dir, 'data', 'light_control')

df_stim = pd.DataFrame([])

tasks = []
subjects = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, '*'))]
for subj in subjects:
    sessions = [s.split('/')[-1] for s in glob.glob(os.path.join(raw_dir, subj, '*'))]
    for ses in sessions:
        file_tdms = glob.glob(os.path.join(raw_dir, subj, ses, '*.tdms'))
        file_pupil = glob.glob(os.path.join(data_dir, 'preprocess', subj, ses, '*.hdf'))
        if (len(file_pupil)==1)&(len(file_tdms)==1):
            tasks.append((raw_dir, data_dir, fig_dir, subj, ses))



preprocess = False
if preprocess:
    n_jobs = 12
    res = Parallel(n_jobs=n_jobs, verbose=1, backend='loky')(delayed(analyse_light_control_session)(*task) for task in tasks)

    # sort:
    df_meta = pd.concat([res[i][0] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_v = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_p = pd.concat([res[i][1] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_l = pd.concat([res[i][3] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_x = pd.concat([res[i][4] for i in range(len(res))], axis=0).reset_index(drop=True)
    # epochs_y = pd.concat([res[i][5] for i in range(len(res))], axis=0).reset_index(drop=True)
    epochs_b = pd.concat([res[i][2] for i in range(len(res))], axis=0).reset_index(drop=True)

    # save:
    df_meta.to_csv(os.path.join(data_dir, 'meta_data.csv'))
    # epochs_v.to_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
    epochs_p.to_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil')
    # epochs_l.to_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
    # epochs_x.to_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
    # epochs_y.to_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
    epochs_b.to_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
    

# load:
print('loading data')
df_meta = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
# epochs_v = pd.read_hdf(os.path.join(data_dir, 'epochs_v.hdf'), key='velocity')
epochs_p = pd.read_hdf(os.path.join(data_dir, 'epochs_p.hdf'), key='pupil') * 100
# epochs_l = pd.read_hdf(os.path.join(data_dir, 'epochs_l.hdf'), key='eyelid')
# epochs_x = pd.read_hdf(os.path.join(data_dir, 'epochs_x.hdf'), key='eye_x')
# epochs_y = pd.read_hdf(os.path.join(data_dir, 'epochs_y.hdf'), key='eye_y')
epochs_b = pd.read_hdf(os.path.join(data_dir, 'epochs_b.hdf'), key='blink')
print('finished loading data')

# baseline:
x = epochs_p.columns
epochs_p = epochs_p - np.atleast_2d(epochs_p.loc[:,(x>=-5)&(x<=-0)].mean(axis=1)).T

epochs_p = epochs_p.loc[:,(x>=-20)&(x<=40)]


# plot:
fig = plt.figure(figsize=(2,2))
x = np.array(epochs_p.columns, dtype=float)
plt.axvline(0, color='r', lw=0.5)
plt.fill_between(x, epochs_p.mean(axis=0)-epochs_p.sem(axis=0), epochs_p.mean(axis=0)+epochs_p.sem(axis=0), alpha=0.2)
plt.plot(x, epochs_p.mean(axis=0))
plt.xlabel('Time from pulse (s)')
plt.ylabel('Pupil')
sns.despine(trim=False, offset=2)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'pupil_responses.pdf'))

# fs = 30000
# a = np.random.normal(0,1,fs*100)
# a[(fs*10):int((fs*10)+fs*0.02)] = 10
# a = a + np.linspace(0,5,len(a))
# from tools_mcginley import utils
# b = utils._butter_highpass_filter(a, 1, fs, order=3)
# c = utils._butter_lowpass_filter(b, 20, fs, order=3)
# b_ = pd.Series(b)
# d = b_.rolling(int(0.02*fs)).mean(center=True)
# plt.plot(a)
# plt.plot(b)
# plt.plot(c)
# plt.plot(d)
# plt.show()