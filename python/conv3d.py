import numpy as np

def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def divup(x, m):
    return divide_roundup(x, m)

def pad_4d(d, dim, size):
    ''' pad d dim to multiple of size'''
    s = list(d.shape)
    assert len(s) == 4, 'd is not 4d'
    d = np.reshape(d,s)
    pad_shape = list(s)
    pad_shape[dim] = roundup_to_multiple(pad_shape[dim],size)
    d_pad = np.zeros(pad_shape, dtype=d.dtype)
    d_pad[:s[0], :s[1], :s[2] ,:s[3]]  = d
    return d_pad

def convolve4d(a, b, sh, sw, pad, debug=False):
    N,C,Y,X = a.shape
    K,C,R,S = b.shape
    H = (Y + 2 * pad - R) / sh + 1
    W = (X + 2 * pad - S) / sw + 1
    o = np.zeros((N,K,H,W))
    for n in range(N):
        for k in range(K):
            for h in range(H):
                for w in range(W):
                    for r in range(R):
                        for s in range(S):
                            for c in range(C):
                                y = (h * sh + r - pad) 
                                x = (w * sw + s - pad) 
                                if debug: print w,h,'=',x,y,'*',r,s,
                                if x in range(X) and y in range(Y):
                                    if debug: print 'valid',
                                    o[n][k][h][w] += a[n][c][y][x] * b[k][c][r][s]
                                if debug: print ''
    return o

def fold_channel(d, sh):
    N,C,H,W = d.shape
    d = d.swapaxes(1,3) # N W H C
    d = pad_4d(d, 2, sh) # pad H to multiple of stride
    d = d.reshape(N, W, divup(H, sh), C*sh)
    d = d.swapaxes(3,1) #N C' H' W

    return d

def fold_channel_2d(d, sh, sw):
    N,C,H,W = d.shape
    d = d.swapaxes(1,3) # N W H C
    d = pad_4d(d, 2, sh) # pad H to multiple of stride
    d = d.reshape(N, W, divup(H, sh), C*sh)
    d = d.swapaxes(1,2) #N H W C
    d = pad_4d(d, 2, sw) # pad W to multiple of stride
    d = d.reshape(N, divup(H, sh), divup(W, sw), C*sh*sw)
    d = d.swapaxes(1,2) #N W H C
    d = d.swapaxes(1,3) #N C H W

    return d

def create_idx_tensor(shape):
    return np.arange(np.product(shape)).reshape(shape)

def test_fold(N,C,H,W,K,R,S,sh,sw,pad=0,debug=False):
    if debug:
        print 'N,C,H,W,K,R,S,sh,sw,pad'
        print N,C,H,W,K,R,S,sh,sw,pad
    a = np.random.randint(0,10,(N,C,H,W))
    b = np.random.randint(0,10,(K,C,R,S))
    a = create_idx_tensor((N,C,H,W))
    b = create_idx_tensor((K,C,R,S))
    o = convolve4d(a,b,sh,sw,pad)
    if debug:
        print 'a'; print a
        print 'b'; print b
        print 'convolve with stride',sh,sw
        print 'o'; print o
    a2 = fold_channel_2d(a,sh,sw)
    b2 = fold_channel_2d(b,sh,sw)
    o2 = convolve4d(a2,b2,1,1,pad)
    if debug:
        print 'a2'; print a2
        print 'b2'; print b2
        print 'o2'; print o2
    if np.count_nonzero(o - o2) == 0:
        if debug:
            print 'PASS'
        return True
    else:
        if debug:
            print 'FAIL'
        return False

if __name__ == '__main__':

    a = np.random.randint(0,10,(1,1,5,5))
    b = np.random.randint(0,10,(1,1,3,3))
    o = convolve4d(a,b,2,0)
    a2 = fold_channel(a,2,1)
    b2 = fold_channel(b,2,1)
    o2 = convolve4d(a2,b2,1,0)

