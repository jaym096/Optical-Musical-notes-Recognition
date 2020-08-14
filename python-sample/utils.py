import numpy as np

def get_dist(i,k,a,b):
    return np.sqrt(np.square(i-a)+np.square(k-b))

def getfulld(arr):
    #finds distance of nearest edge
   resp = np.empty(arr.shape)
   for i in range(3):
       for j in range(3):
           leastd = float('inf')
           p,q = None, None
           for k in range(3):
               for l in range(3):
                   if arr[k][l] == 1:
                       d = get_dist(i,j,k,l)
                       if d<leastd:
                           leastd = d
                           p, q = k, l
           resp[i][j] = leastd
   return resp

def getgamma(val):
    if val:
        return 0
    return float('inf')

def getsimilaritymatrix(image,template):
    hshape = image.shape[0]-template.shape[0]+1
    vshape = image.shape[1]-template.shape[1]+1
    op = np.empty((
        hshape, vshape
    ))

    for i in range(image.shape[0]-2):
        for j in range(image.shape[1]-2):
            op[i][j] = image[
                i:i+template.shape[0], j:j+template.shape[1]
            ].dot(template)
