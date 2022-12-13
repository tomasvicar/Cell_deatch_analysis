from glob import glob
import QDF
import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(42)
imgs_around = 2

path_qdfs = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction'
path_data_save = r'D:\nove_skluzavky_save\nuc_for_labeling'


fnames = glob(path_qdfs + '\*.qdf')

missing = dict()

for fname in fnames:
    
    print()
    print(os.path.split(fname)[1])
    # if not 'RSL3_PC3_1_DFT.qdf' in fname:
    #     continue
    

    reader = QDF.reader(fname)
    main_info = reader.main_info
    index = reader.index
    
    
    num_of_t = reader.ranges['Clipped:DAPI']['t']
    num_of_p = reader.ranges['Clipped:DAPI']['p']
    z = 0
    # print('ts DAPI='  + str(reader.ranges['Clipped:DAPI']['t'])  + ' ts QPI=' + str(reader.ranges['Compensated phase']['t']) )
    # print('ps DAPI='  + str(reader.ranges['Clipped:DAPI']['p'])  + ' ps QPI=' + str(reader.ranges['Compensated phase']['p']) )
    
    # print('imgs DAPI='  + str(reader.ranges['Clipped:DAPI']['t'] * reader.ranges['Clipped:DAPI']['p'])  + ' imgs QPI=' + str(reader.ranges['Compensated phase']['t'] * reader.ranges['Compensated phase']['p']) )
    # print('imgs DAPI='  + str(len(index['Clipped:DAPI'].keys()))  + ' imgs QPI=' + str(len(index['Compensated phase'].keys())) )
    
    
    
    for p in range(num_of_p):
    
        missing_dapi = []
        for t in range(num_of_t):
            
            if not '[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']' in index['Clipped:DAPI'].keys():
                missing_dapi.append(t)
            
        missing_qpi = []
        for t in range(num_of_t):
            
            if not '[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']' in index['Compensated phase'].keys():
                missing_qpi.append(t)
                
        print(missing_dapi)
        
        print(missing_qpi)     
        
        missing[os.path.split(fname)[1] + ' ' + str(p).zfill(2) + ' dapi'] = missing_dapi
        missing[os.path.split(fname)[1] + ' ' + str(p).zfill(2) + ' qpi'] = missing_qpi      

        
        
    


