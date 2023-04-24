import numpy as np
from matplotlib import pyplot as plt 

# ==== process phase information === 

def smooth(f, width): 
    '''
    Average f over width. 
    The output is of length len(f)/width
    '''
    N = len(f)
    length = int(np.floor(N/width))
    f = f[:length*width]
    f_smooth = np.mean(f.reshape((length, width)), -1)
    return f_smooth 

def shift(phases, tol=1): 
    '''
    Shift the phases such that all data points are continuous 
    '''
    z = np.exp(phases*1j) 
    diff_z = np.abs(z[1:]-z[:-1])
    diff_ang = phases[1:]-phases[:-1]
    for (i, (dz, dtheta)) in enumerate(zip(diff_z, diff_ang)): 
        if np.abs(dtheta) - dz > tol:  
            if dtheta < 0:
                phases[i+1:] += np.pi*2
            else: 
                phases[i+1:] -= np.pi*2 