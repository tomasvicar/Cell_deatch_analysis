import torch
import os
from skimage.io import imread
from tifffile import TiffWriter
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json


def fillmissing(signal_with_nans):
    
    y = signal_with_nans.copy()
    
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    
    return y


missing_pairs = dict()
missing_pairs['area'] = 'Compensated phase'
missing_pairs['mass'] = 'Compensated phase'
missing_pairs['density'] = 'Compensated phase'
missing_pairs['CDS'] = 'Compensated phase'
missing_pairs['mean_TRITC'] = 'Clipped:TRITC'
missing_pairs['mean_FITC'] = 'Clipped:FITC'
missing_pairs['mean_DAPI'] = 'Clipped:DAPI'
missing_pairs['mean_detection_DAPI'] = 'Clipped:DAPI'
missing_pairs['mean_TRITC_norm'] = 'Clipped:TRITC'
missing_pairs['mean_FITC_norm'] = 'Clipped:FITC'
missing_pairs['mean_DAPI_norm'] = 'Clipped:DAPI'
missing_pairs['mean_detection_DAPI_norm'] = 'Clipped:DAPI'




data_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\exported_tiff'
save_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\segmentation1'



filenames_qpi = glob(data_path + '/**/*Compensated phase.tiff')
filenames_dapi = [x.replace('Compensated phase.tiff','DAPI.tiff') for x in filenames_qpi]


for tiff_num, filename_qpi, in enumerate(filenames_qpi):
    
    if tiff_num < 166:
        continue
    
    
    print(str(tiff_num) + ' / ' + str(len(filenames_qpi)))
    print(filename_qpi)
    
    
    filename_dapi = filename_qpi.replace('Compensated phase.tiff','DAPI.tiff')
    filename_fitc = filename_qpi.replace('Compensated phase.tiff','FITC.tiff')
    filename_tritc = filename_qpi.replace('Compensated phase.tiff','TRITC.tiff')
    
    filename_fg = filename_qpi.replace('Compensated phase.tiff','fg.tiff').replace('exported_tiff','segmentation1')
    filename_detections = filename_qpi.replace('Compensated phase.tiff','detections.npy').replace('exported_tiff','detections')
    filename_missing =  filename_qpi.replace('Compensated phase.tiff','missing.json')
    
    filename_save = filename_qpi.replace('Compensated phase.tiff','').replace('exported_tiff','evaluate_whole') + 'results.json'
    
    
    with open(filename_missing) as json_file:
        missing =json.load(json_file)
        
    
    
    results = dict()
    
    
    fg = imread(filename_fg,key = slice(None))
    results['area'] = np.sum(fg.astype(np.float32), axis=(1,2))
    
    tmp = imread(filename_qpi,key = slice(None))
    results['mass'] = np.sum(fg.astype(np.float32) * tmp, axis=(1,2))
    results['density'] = results['mass'] / results['area']
    
    
    fg_merge = fg[1:,:,:] | fg[:-1,:,:]
    results['CDS'] =np.mean( np.diff(tmp, axis=0) ** 2 * fg_merge.astype(np.float32), axis=(1,2)) / np.sum(fg_merge.astype(np.float32), axis=(1,2))
    
    del fg_merge
    
    tmp = imread(filename_tritc,key = slice(None))
    results['mean_TRITC'] = np.sum(fg.astype(np.float32) * tmp, axis=(1,2)) / np.sum(fg.astype(np.float32), axis=(1,2))
    results['mean_TRITC_norm'] = results['mean_TRITC'] / (np.sum((fg == 0).astype(np.float32) * tmp, axis=(1,2)) / np.sum((fg == 0).astype(np.float32), axis=(1,2)))
    
    tmp = imread(filename_fitc,key = slice(None))
    results['mean_FITC'] = np.sum(fg.astype(np.float32) * tmp, axis=(1,2))/ np.sum(fg.astype(np.float32), axis=(1,2))
    results['mean_FITC_norm'] = results['mean_FITC'] / (np.sum((fg == 0).astype(np.float32) * tmp, axis=(1,2)) / np.sum((fg == 0).astype(np.float32), axis=(1,2)))
    
    tmp = imread(filename_dapi,key = slice(None))
    results['mean_DAPI'] = np.sum(fg.astype(np.float32) * tmp, axis=(1,2))/ np.sum(fg.astype(np.float32), axis=(1,2))
    results['mean_DAPI_norm'] = results['mean_DAPI'] / (np.sum((fg == 0).astype(np.float32) * tmp, axis=(1,2)) / np.sum((fg == 0).astype(np.float32), axis=(1,2)))
    
    
    tmp = imread(filename_dapi,key = slice(None))
    dapi_detection_mean = []
    detections = np.load(filename_detections, allow_pickle=True)
    for frame_num, det in enumerate(detections):
        
        if det.size == 0:
            dapi_detection_mean.append(np.nan)
        else:
            x = np.mean(tmp[frame_num, det[:,1], det[:,0]])
            dapi_detection_mean.append(x)
        
    results['mean_detection_DAPI']  = np.array(dapi_detection_mean)
    results['mean_detection_DAPI_norm'] = results['mean_detection_DAPI'] / (np.sum((fg == 0).astype(np.float32) * tmp, axis=(1,2)) / np.sum((fg == 0).astype(np.float32), axis=(1,2)))
    
        
        
    for key in results:
        signal = results[key]
        nans = np.array(missing[missing_pairs[key]])
        
            
        if key != 'CDS':
            if nans.size > 0:
                signal[nans] = np.nan
        else:
            nans = nans[nans < signal.size]
            if nans.size > 0:
                signal[nans] = np.nan
                signal[nans - 1] = np.nan
            signal = np.append(signal,np.nan)
        signal[0:2] = np.nan
             
        signal = fillmissing(signal)
        results[key] = signal.tolist()
        
     
     
     
    # for key in results:
    #     signal = np.array(results[key])
    #     plt.plot(signal)
    #     plt.title(key)
    #     plt.show() 
    
    
    
    
    folder = os.path.split(filename_save)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename_save, 'w') as json_file:
        json.dump(results, json_file)
    
    
    
    
    
    
    


