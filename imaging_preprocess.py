import sys, os, glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py 

from suite2p.run_s2p import run_s2p, default_ops

from tools_mcginley import utils
# from tools_mcginley.preprocess_calcium import merge

from IPython import embed as shell

subjects = {
    'C7A2':  ['1',      '3', '4', '5', '6', '7', '8'],
    'C7A6':  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'C1772': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'C1773': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
    
    
    
    'D1574': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'D1640': ['1', '2', '3', '4'], # all bad
    'D1641': ['1', '2',      '4'], 
    'D1766': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'D1767': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
}

subjects = {
    'C7A2': ['1', '8'],
    'C7A6': ['1', '2', '3', '5', '6', '7', '8',],  
    'C1772': ['1', '5', '6', '7', '8',],
    'C1773': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
}

subjects = {
    'D1766': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    # 'D1767': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
}

# subjects = {
#     'C1772': ['7'],
# }

raw_dir = '/media/external4/2p_imaging/vns/'
temp_dir = '/media/internal1/vns/'

# default zoom:
default_zoom = 4
default_size = 512 # in pixels
maxregshift = 20 # in micron
maxregshift_std = 4 # in micron

run_suite2p = True
plotting = True
merging = False

errors = []
for subj in subjects.keys():
    for session in subjects[subj]:
        
        print()
        print(subj)
        print(session)

        # directories:
        raw_directory = os.path.join(raw_dir, subj, session)
        temp_directory = os.path.join(temp_dir, subj, session)

        # create directories:
        os.makedirs(temp_directory, exist_ok=True)

        # files on server:
        files = glob.glob(os.path.join(raw_directory, '*.tif'))
        if len(files) == 0:
            print('NO TIFS FOR {}, session {}!!'.format(subj, session))
            continue

        # get zoom factor:
        zoom = utils.get_image_zoom_factor(files[0])
        px_per_micron = 0.0535+0.44*zoom

        # tiffs:
        if run_suite2p:

            matplotlib.use('Agg')
            
            print('Preprocessing {} {}'.format(subj, session))
           
            # copy to local:
            print('copying to local drive')
            for f in files:
                if not os.path.exists(os.path.join(temp_directory, f.split('/')[-1])):
                    os.system('cp {} {}'.format(f, temp_directory))

            # get default obs:
            ops = default_ops()

            # run analysis:
            print('run preprocessing') 
            db = {
                # main settings:
                'data_path': [temp_directory], 
                'save_path0': temp_directory, 
                'nchannels' : 2,
                'fs' : 15,
                'keep_movie_raw' : True,
                'frames_include' : -1,
                
                # motion correction:
                'do_registration': True,
                'two_step_registration': True,
                'nonrigid': False,
                'nimg_init': 2500,
                'maxregshift': maxregshift * px_per_micron / default_size,
                'magregshift_std': maxregshift_std * px_per_micron,
                # 'subpixel' : round(ops['subpixel'] / (zoom / default_zoom)),
                # 'smooth_sigma': ops['smooth_sigma'] * (zoom / default_zoom),
                'smooth_sigma_time' : 2,
                
                # roi detection:
                'roidetect': False,
                'sparse_mode': True,
                }

            try:
                opsEnd=run_s2p(ops=ops, db=db)
            except:
                errors.append('error for {} session {}'.format(subj, session))
            
            # # back to server:
            # print('pushing back to server')
            # os.system('cp -r {} {}'.format(os.path.join(temp_directory, 'suite2p'), processed_folder_server))

            # # clean-up:
            # print('delete from local')
            # os.system('rm -r {}'.format(temp_directory))

        if plotting: 
            
            # plots:
            ops = np.load(os.path.join(temp_directory, 'suite2p', 'ops1.npy'), allow_pickle=True)[0]
            time = np.arange(ops['yoff'].shape[0])
            for img in ['refImg', 'meanImg']:
                image = ops[img]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.pcolormesh(np.arange(image.shape[0]), np.arange(image.shape[1]), image, 
                    vmin=np.percentile(image.ravel(), 1), vmax=np.percentile(image.ravel(), 99), cmap='Greys_r')
                ax.set_aspect('equal')
                fig.savefig(os.path.join(temp_directory, 'suite2p', '{}.png'.format(img)))
            
            fig = plt.figure()
            ax = fig.add_subplot(411)
            ax.plot(time, ops['xoff'])
            ax.axhline(-maxregshift*px_per_micron*0.95, ls='--', lw=0.5, color='r')
            ax.axhline(maxregshift*px_per_micron*0.95, ls='--', lw=0.5, color='r')
            ax.axhline(0, lw=0.5, color='k')
            ax = fig.add_subplot(412)
            ax.plot(time, ops['yoff'])
            ax.axhline(-maxregshift*px_per_micron*0.95, ls='--', lw=0.5, color='r')
            ax.axhline(maxregshift*px_per_micron*0.95, ls='--', lw=0.5, color='r')
            ax.axhline(0, lw=0.5, color='k')
            ax = fig.add_subplot(413)
            ax.plot(time, ops['badframes'].astype(int))
            ax = fig.add_subplot(414)
            ax.plot(time, ops['corrXY'])
            plt.tight_layout()
            fig.savefig(os.path.join(temp_directory, 'suite2p', 'offsets.pdf'))

        if merging:
            
            merge.merge_rois(processed_folder_server, min_coh=0.1, max_px=15)

print()
print()
for error in errors:
    print(error)