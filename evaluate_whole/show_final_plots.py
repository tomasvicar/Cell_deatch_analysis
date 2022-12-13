import torch
import os
from skimage.io import imread
from tifffile import TiffWriter
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'



data_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\exported_tiff'
results_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\evaluate_whole'


filenames_qpi = glob(data_path + '/**/*Compensated phase.tiff')
filenames_dapi = [x.replace('Compensated phase.tiff','DAPI.tiff') for x in filenames_qpi]


results_all = dict()
for tiff_num, filename_qpi, in enumerate(filenames_qpi):
    
    print(str(tiff_num) + ' / ' + str(len(filenames_qpi)))
    print(filename_qpi)
    
    
    filename_results = filename_qpi.replace('Compensated phase.tiff','').replace(data_path,results_path) + 'results.json'
        

        
    with open(filename_results) as json_file:
        results = json.load(json_file)
        
    results['mass_rel'] =  (np.array(results['mass']) / results['mass'][0]).tolist()
    results_all[os.path.split(filename_qpi)[-1].replace('_Compensated phase.tiff','')] = results
    
    features_keys = list(results.keys())
    
            

all_keys = list(results_all.keys())
all_keys_main = list(set([x[:-4] for x in all_keys]))
    
    

    
result_avg = dict()
for key_main in all_keys_main:
    
   
    
    result_avg[key_main] = dict()
    for features_key in features_keys:
        result_avg[key_main][features_key] = []
    
    for key in all_keys:
        
        if key_main not in key:
            continue
        
        for features_key in features_keys:
            tmp = np.array(results_all[key][features_key])
            tmp = medfilt(tmp,5)
            tmp = gaussian_filter1d(tmp,1)
            result_avg[key_main][features_key].append(tmp)
                
                
    
    

for used_main_keys, cell_type in zip( [['Matrine_4mM_PC3_1_DFT', 'RSL3_PC3_1_DFT', 'PC3_DOXO_400nM'], ['FaDu_DOXO_vyssi koncentrace', 'FaDu_RSL3_Nec1', 'FaDu_Doxo_fluo_pokracovani']], ['PC3', 'FaDu']):
    death_types = ['apo', 'fero', 'necro']
    used_features_keys = ['mass_rel','density','CDS']
        
    
    for features_key in used_features_keys:
        
        plt.figure(figsize=[12,4])
        used_key_main = []
        for key_main, death_type in zip(used_main_keys, death_types):
    
            
            used_key_main.append(key_main)
            
            tmp = result_avg[key_main][features_key]
            tmp = np.stack(tmp,axis=0)
            tmp = np.median(tmp,axis=0)
            
            tmp = tmp[:287]
            time = np.linspace(0,24,287)
            plt.plot(time,tmp, label=key_main+'_'+death_type)
                
            
        
        plt.title(cell_type + '   '  + features_key)
        # plt.legend()
        plt.savefig('results_final' + '/' + cell_type + '_' + features_key + '_plot.png',  bbox_inches='tight', pad_inches=0)
        plt.savefig('results_final' + '/' + cell_type + '_' + features_key + '_plot.svg',  bbox_inches='tight', pad_inches=0, transparent=True)
        plt.show()
    
        
        
    
    
        
    
for used_main_keys, cell_type in zip( [['Matrine_4mM_PC3_1_DFT', 'RSL3_PC3_1_DFT', 'PC3_DOXO_400nM'], ['FaDu_DOXO_vyssi koncentrace', 'FaDu_RSL3_Nec1', 'FaDu_Doxo_fluo_pokracovani']], ['PC3', 'FaDu']):
    death_types = ['apo', 'fero', 'necro']
    used_features_keys = ['mass_rel','density','CDS']
    colors = [[0, 0.4470, 0.7410],[0.8500, 0.3250, 0.0980],[0.4660, 0.6740, 0.1880]]
        

    plt.figure(figsize=[8,8])
    used_key_main = []
    for key_main, death_type, color in zip(used_main_keys, death_types, colors):

        
        used_key_main.append(key_main)
        
        tmp = result_avg[key_main]['density']
        tmp = np.stack(tmp,axis=0)
        if cell_type == 'PC3':
            density = tmp[:,int(6*60 /5)]
        else:
            density = tmp[:,int(12*60 /5)]
            
        tmp = result_avg[key_main]['CDS']
        tmp = np.stack(tmp,axis=0)
        if cell_type == 'PC3':
            cds = tmp[:,int(6*60 /5)]
        else:
            cds = tmp[:,int(12*60 /5)]

        

        plt.plot(cds,density,'.', label=key_main+'_'+death_type, color=color)
                
        
    plt.ylim([0.6,2.2])
    plt.xlim([0,1.2*1e-6])
    
    plt.title(cell_type + '   '  + features_key)
    # plt.legend()
    plt.savefig('results_final' + '/' + cell_type + '_' + '_points.png',  bbox_inches='tight', pad_inches=0)
    plt.savefig('results_final' + '/' + cell_type + '_' + '_points.svg',  bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    


    