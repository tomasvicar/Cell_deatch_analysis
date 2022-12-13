from glob import glob
import QDF
import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread, imwrite, TiffWriter



np.random.seed(42)
imgs_around = 0
margin = 50

path_qdfs = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction'
path_data_save = r'D:\nove_skluzavky_save\nuc_for_labeling'


fnames = glob(path_qdfs + '\*.qdf')

for fname in fnames:
    
    print()
    print(os.path.split(fname)[1])
    

    reader = QDF.reader(fname)
    main_info = reader.main_info
    index = reader.index
    
    
    num_of_t = reader.ranges['Clipped:DAPI']['t']
    num_of_p = reader.ranges['Clipped:DAPI']['p']
    z = 0


    
            
            
    
    for p in range(num_of_p):
        
        
        missing_dapi = []
        for t in range(num_of_t):
            if not '[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']' in index['Clipped:DAPI'].keys():
                missing_dapi.append(t)
            
        missing_qpi = []
        for t in range(num_of_t):
            if not '[' + str(t) + ', ' + str(p) + ', ' + str(z) + ']' in index['Compensated phase'].keys():
                missing_qpi.append(t)
                
                
        
        range_ = np.arange(imgs_around, num_of_t - imgs_around - 1)
        
        for miss in missing_dapi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])
                
                
        for miss in missing_qpi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])       
                

    
        t =  np.random.choice(range_)
        t1 = t.copy()
    
        c = 'Clipped:DAPI'
        dapi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.uint16)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            dapi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600)
            
        c = 'Compensated phase'
        qpi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.float32)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            qpi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600).astype(np.float32)
      
        
        fname_save = path_data_save + '/sample1/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_DAPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(dapi.shape[0]):
                tif.write(dapi[k,:,:] ,compression = 'zlib')
                
            
        
        fname_save = path_data_save + '/sample1/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_QPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(qpi.shape[0]):
                tif.write(qpi[k,:,:] ,compression = 'zlib')
                
        
        
        
        
        
        
        
        
        
        range_ = np.arange(imgs_around, num_of_t - imgs_around - 1)
        
        for miss in missing_dapi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])
                
                
        for miss in missing_qpi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])       
                

        remove = np.array([t - margin, t + margin])
        remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
        remove[remove < (0 + imgs_around)] = 0 + imgs_around
        for rem in range(remove[0],remove[1]):
            range_ = np.delete(range_, np.where(range_ == rem)[0])    
            

        t = np.random.choice(range_)
        t2 = t.copy()

        c = 'Clipped:DAPI'
        dapi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.uint16)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            dapi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600)
            
        c = 'Compensated phase'
        qpi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.float32)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            qpi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600).astype(np.float32)
      
        
        fname_save = path_data_save + '/sample2/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_DAPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(dapi.shape[0]):
                tif.write(dapi[k,:,:] ,compression = 'zlib')
                
            
        
        fname_save = path_data_save + '/sample2/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_QPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(qpi.shape[0]):
                tif.write(qpi[k,:,:] ,compression = 'zlib')
                






        range_ = np.arange(imgs_around, num_of_t - imgs_around - 1)
        
        for miss in missing_dapi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])
                
                
        for miss in missing_qpi:
            remove = np.array([miss - imgs_around , miss + imgs_around + 1])
            remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
            remove[remove < (0 + imgs_around)] = 0 + imgs_around
            for rem in range(remove[0],remove[1]):
                range_ = np.delete(range_, np.where(range_ == rem)[0])       
                

        remove = np.array([t1 - margin, t1 + margin])
        remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
        remove[remove < (0 + imgs_around)] = 0 + imgs_around
        for rem in range(remove[0],remove[1]):
            range_ = np.delete(range_, np.where(range_ == rem)[0])    
            
        remove = np.array([t2 - margin, t2 + margin])
        remove[remove > (num_of_t - 1 - imgs_around)] = num_of_t - 1 - imgs_around
        remove[remove < (0 + imgs_around)] = 0 + imgs_around
        for rem in range(remove[0],remove[1]):
            range_ = np.delete(range_, np.where(range_ == rem)[0])  
            

        t = np.random.choice(range_)
        t3 = t.copy()

        c = 'Clipped:DAPI'
        dapi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.uint16)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            dapi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600)
            
        c = 'Compensated phase'
        qpi = np.zeros((imgs_around * 2 +1,600,600),dtype = np.float32)
        for i, k in enumerate(range(t - imgs_around, t + imgs_around + 1)):
            qpi[i,:,:] = reader.get_image(c, k, p, z).reshape(600,600).astype(np.float32)
      
        
        fname_save = path_data_save + '/sample3/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_DAPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(dapi.shape[0]):
                tif.write(dapi[k,:,:] ,compression = 'zlib')
                
            
        
        fname_save = path_data_save + '/sample3/' + os.path.split(fname)[1].replace('.qdf','')  + '__p' + str(p).zfill(2) + '_t' + str(t).zfill(3) + '_QPI.tiff'
        with TiffWriter(fname_save,bigtiff=True) as tif:
            for k in range(qpi.shape[0]):
                tif.write(qpi[k,:,:] ,compression = 'zlib')
                
                