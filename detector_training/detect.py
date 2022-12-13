from skimage.feature import peak_local_max
import numpy as np
from skimage.morphology import h_maxima
from skimage.measure import regionprops

# def detect(predition, min_dist, min_val, min_h):
    
    
#     p1 = peak_local_max(predition, min_distance=min_dist, threshold_abs=min_val)
#     p2 = h_maxima(predition, min_h)
    
    
#     detections = []

#     for p in p1:
#         if p2[int(p[0]),int(p[1])] > 0:
            
#             detections.append([int(p[0]),int(p[1])])
        
#     return np.array(detections)


def detect(img, T, h, d):
    
    p1 = peak_local_max(img, min_distance=int(np.round(d)), threshold_abs=T)
    p2 = np.stack(np.nonzero(h_maxima(img, h)), axis=1)
    
    p1 = set([tuple(x) for x in p1.tolist()])
    p2 = set([tuple(x) for x in p2.tolist()])
    
    detections = list(p1.intersection(p2))

    detections = np.array(detections)
    
    if len(detections.shape) == 2:
        detections = detections[:,[1,0]]
    
    return detections