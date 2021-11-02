import numpy as np

def moving_average(y, N=3) :

    y_smooth = []
    y_smooth.append(y[0])
    for i in np.arange(1, N):
        y_smooth = np.mean(y[:i])



    return y_smooth
#

#

def moving_convolution(y, N=3) :

    #b = np.convolve(a, np.ones((n,))/n, mode='same')

    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones((N,))/N, mode='valid')

    return y_smooth

#

#

# fin
