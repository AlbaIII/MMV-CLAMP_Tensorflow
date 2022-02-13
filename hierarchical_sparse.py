#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow as tf
from tools import raputil as ru


# import raputil as ru

class Generator(object):
    #def __init__(self, A, **kwargs):
    def __init__(self, A, E, N, **kwargs):
        self.A = A
        [M1, N1] = A.shape
        self.E = E
        #vars(self).update(kwargs)
        #self.x_ = tf.placeholder(tf.float32, (None, N*2), name='x')
        #self.y_ = tf.placeholder(tf.float32, (None, M*2), name='y')
        #self.x_ = tf.placeholder(tf.float32, (None, N), name='x')
        #self.y_ = tf.placeholder(tf.float32, (None, M), name='y')
        #self.x_ = tf.placeholder(tf.float32, (None, N*E), name='x')
        #self.y_ = tf.placeholder(tf.float32, (None, M*E), name='y')
        #self.x_ = tf.placeholder(tf.float32, (N, None), name='x')
        #self.y_ = tf.placeholder(tf.float32, (M, None), name='y')
        # LAMP-C network
        self.x_ = tf.placeholder(tf.float32, (None,E*N1), name='x')
        self.y_ = tf.placeholder(tf.float32, (None,E*M1), name='y')
        # # LAMP-D network
        # self.x_ = tf.placeholder(tf.float32, (None,N1), name='x')
        # self.y_ = tf.placeholder(tf.float32, (None,M1), name='y')
        # # LAMP-H network
        # U = 2
        # E1 = int(E/U)
        # self.x_ = tf.placeholder(tf.float32, (None,E1*N1), name='x')
        # self.y_ = tf.placeholder(tf.float32, (None,E1*M1), name='y')


class TFGenerator(Generator):
    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)

    def __call__(self, sess):
        'generates y,x pair for training'
        return sess.run((self.ygen_, self.xgen_))


class NumpyGenerator(Generator):
    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)

    def __call__(self, sess):
        'generates y,x pair for training'
        return self.p.genYX(self.nbatches, self.nsubprocs)


def bernoulli_gaussian_hierarchical_sparse_trial(M=10, N=200, L=1000, Tg=3, pnz=0.05, kappa=0, SNR=0):
    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
    if kappa >= 1:
        # create a random operator with a specific condition number
        U, _, V = la.svd(A, full_matrices=False)
        s = np.logspace(0, np.log10(1 / kappa), M)
        A = np.dot(U * (s * np.sqrt(N) / la.norm(s)), V).astype(np.float32)
    A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A = A / A_col_norm
    Ao = A
    A_v = np.zeros([M + Tg, N * (Tg + 1)]).astype(np.float32)
    for i1 in range(N):
        for i2 in range(Tg + 1):
            A_v[i2:(i2 + M), i1 * (Tg + 1) + i2] = A[:, i1]
    A = A_v
    A_ = tf.constant(A, name='A')
    prob_hisps = TFGenerator(A=A, A_=A_, pnz=pnz, kappa=kappa, SNR=SNR)
    prob_hisps.name = 'Bernoulli-Gaussian-Hierarchical-Sparse, random A'

    test_batch = 5000
    bernoulli = np.random.uniform(0, 1, size=(N, L)).astype(np.float32)
    for col in range(L):
        for row in range(N):
            if bernoulli[row][col] < pnz:
                bernoulli[row][col] = 1
            else:
                bernoulli[row][col] = 0
        sum_ber = np.sum(bernoulli[:, col])
        if sum_ber == 0:
            bernoulli[0][col] = 1
    x_channel = np.random.normal(size=(N, L), scale=1).astype(np.float32)
    x_sync = np.multiply(x_channel, bernoulli)
    x_sync_test = x_sync[:, 0:test_batch]
    prob_hisps.bernoulli = bernoulli
    ind_test = bernoulli[:, 0:test_batch]
    prob_hisps.ind_test = ind_test
    prob_hisps.x_sync_test = x_sync_test
    noise_var = 1 / np.sqrt(M) * math.pow(10., -SNR / 10.)

    #    bernoulli_ = tf.to_float(tf.random_uniform((N, L)) < pnz)
    #    xgen_ = bernoulli_ * tf.random_normal((N, L))
    #    noise_var = pnz * N / M * math.pow(10., -SNR / 10.)
    #    ygen_ = tf.matmul(A_, xgen_) + tf.random_normal((M, L), stddev=math.sqrt(noise_var))

    user_delay = np.random.random_integers(0, Tg, size=(N, L))
    ud_test = user_delay[:, 0:test_batch]
    prob_hisps.ud_test = ud_test
    prob_hisps.ud = user_delay
    x_vir_channel = np.zeros([N * (Tg + 1), L]).astype(np.float32)
    for iL in range(L):
        for iu in range(N):
            if bernoulli[iu][iL] == 1:
                x_vir_channel[iu*(Tg+1) + user_delay[iu,iL],iL] = x_channel[iu,iL]
    #               print(x_vir_channel[iu][iL])
    x_test = x_vir_channel[:, 0:test_batch]
    noise = np.random.normal(size=(M+Tg, L), scale=noise_var).astype(np.float32)
    SNR_ex = np.array([3, 6, 9, 12])
    sigma_w = np.zeros([4]).astype(np.float32)
    noise_ex = np.zeros([M+Tg, 4*test_batch]).astype(np.float32)
    for iw in range(4):
        sigma_w[iw] = 1 / np.sqrt(M) * math.pow(10., -SNR_ex[iw] / 10.)
        noise_ex[:,iw*test_batch:(iw+1)*test_batch] = sigma_w[iw]/noise_var*noise[:,0:test_batch]


    y_signal_noiseless = np.dot(A_v, x_vir_channel)
    y_signal = y_signal_noiseless + noise
    y_test = y_signal[:, 0:test_batch]
    y_test_ex = np.zeros([M+Tg, 4*test_batch])
    x_test_ex = np.zeros([N*(Tg+1), 4*test_batch]).astype(np.float32)
    for iw in range(4):
        x_test_ex[:,iw*test_batch:(iw+1)*test_batch] = x_test
        y_test_ex[:,iw*test_batch:(iw+1)*test_batch] = y_signal_noiseless[:,0:test_batch] + noise_ex[:,iw*test_batch:(iw+1)*test_batch]
    prob_hisps.y_test_ex = y_test_ex
    prob_hisps.x_test_ex = x_test_ex

    y_sync = np.dot(Ao, x_sync) + noise[0:M, :]
    y_sync_test = y_sync[:, 0:test_batch]
    prob_hisps.y_sync_test = y_sync_test

    prob_hisps.Ao = Ao
    prob_hisps.x_sync = x_sync
    prob_hisps.y_sync = y_sync

    xgen_ = tf.constant(x_vir_channel, name='x_vir_channel')
    ygen_ = tf.constant(y_signal, name='y_signal')

    prob_hisps.x_test = x_test
    prob_hisps.y_test = y_test
    prob_hisps.xval = x_vir_channel
    prob_hisps.yval = y_signal
    prob_hisps.xinit = x_vir_channel
    prob_hisps.yinit = y_signal
    prob_hisps.xgen_ = xgen_
    prob_hisps.ygen_ = ygen_
    prob_hisps.noise_var = noise_var

    return prob_hisps

def bernoulli_gaussian_hierarchical_sparse_MMV_trial(M=10,N=200,E=2,L=200000,Tg=3,pnz=0.05,kappa=0,SNR=0):
    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
    if kappa >= 1:
        # create a random operator with a specific condition number
        U, _, V = la.svd(A, full_matrices=False)
        s = np.logspace(0, np.log10(1 / kappa), M)
        A = np.dot(U * (s * np.sqrt(N) / la.norm(s)), V).astype(np.float32)
    A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A = A / A_col_norm
    Ao = A
    #A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A_v = np.zeros([M+Tg, N*(Tg+1)]).astype(np.float32)
    for i1 in range(N):
        for i2 in range(Tg+1):
            A_v[i2:(i2 + M), i1 * (Tg + 1) + i2] = A[:, i1]
    A = A_v
    A_ = tf.constant(A, name='A')
    prob_hisps_MMV = TFGenerator(A=A, A_=A_, pnz=pnz, kappa=kappa, SNR=SNR, Ao=Ao, M=M, N=N, Tg=Tg, E=E)
    prob_hisps_MMV.name = 'Bernoulli-Gaussian-Hierarchical-Sparse-MMV, random A'

    test_batch = 20000
    bernoulli = np.random.uniform(0, 1, size=(L, N)).astype(np.float32)
    ind_channel = np.zeros(shape=(L, E, N)).astype(np.float32)
    for height in range(L):
        for row in range(N):
            if bernoulli[height,row] < pnz:
                bernoulli[height,row] = 1.0
                ind_channel[height,:,row] = np.ones(shape=(E)).astype(np.float32)
            else:
                bernoulli[height,row] = 0.0
                ind_channel[height,:,row] = np.zeros(shape=(E)).astype(np.float32)
        sum_ber = np.sum(bernoulli[height, :])
        if sum_ber == 0:
            bernoulli[height,0] = 1.0
            ind_channel[height,:,0] = np.ones(shape=(E)).astype(np.float32)
    prob_hisps_MMV.ind = np.transpose(bernoulli, (1,0))
    x_channel = np.random.normal(size=(L, E, N), scale=1.0).astype(np.float32)
    x_sync = np.multiply(x_channel, ind_channel)
    x_sync_test = x_sync[0:test_batch,:,:]
    x_sync_test_vec = x_sync_test.reshape([test_batch,N*E])
    prob_hisps_MMV.x_sync_test_mat = np.transpose(x_sync_test, (2,1,0))
    prob_hisps_MMV.x_sync_test = x_sync_test_vec
    ind_test = bernoulli[0:test_batch,:]
    prob_hisps_MMV.ind_test = np.transpose(ind_test, (1,0))

    user_delay = np.random.random_integers(0, Tg, size=(L, N))
    prob_hisps_MMV.ud = np.transpose(user_delay, (1,0))
    ud_test = user_delay[0:test_batch,:]
    prob_hisps_MMV.ud_test = np.transpose(ud_test, (1,0))

    x_vir_channel = np.zeros([L,E,N*(Tg+1)]).astype(np.float32)
    for iL in range(L):
        for iu in range(N):
            if bernoulli[iL,iu] == 1.0:
                x_vir_channel[iL,:,iu*(Tg+1)+user_delay[iL,iu]] = x_channel[iL,:,iu]
    x_vir_channel_vec = x_vir_channel.reshape([L,N*(Tg+1)*E])
    prob_hisps_MMV.x_mat = np.transpose(x_vir_channel, (2,1,0))
    x_test = x_vir_channel[0:test_batch,:,:]
    prob_hisps_MMV.x_test_mat = np.transpose(x_test, (2,1,0))
    x_test_vec = x_test.reshape([test_batch,N*(Tg+1)*E])
    prob_hisps_MMV.x_test = x_test_vec

    noise_var = 1 / np.sqrt(M) * math.pow(10., -SNR / 10.)
    noise = np.random.normal(size=(L,E,M+Tg), scale=noise_var).astype(np.float32)
    prob_hisps_MMV.noise_mat = np.transpose(noise, (2,1,0))
    SNR_ex = np.array([3, 6, 9, 12])
    sigma_w = np.zeros([4]).astype(np.float32)
    noise_ex = np.zeros([4*test_batch,E,M+Tg]).astype(np.float32)
    for iw in range(4):
        sigma_w[iw] = 1 / np.sqrt(M) * math.pow(10., -SNR_ex[iw] / 10.)
        noise_ex[iw*test_batch:(iw+1)*test_batch,:,:] = sigma_w[iw] / noise_var * noise[0:test_batch,:,:]

    y_signal_noiseless = np.dot(np.reshape(x_vir_channel, (E*L,N*(Tg+1))), np.transpose(A_v))
    y_signal_noiseless = y_signal_noiseless.reshape([L,E,M+Tg])
    y_signal = y_signal_noiseless + noise
    prob_hisps_MMV.y_mat = np.transpose(y_signal, (2,1,0))
    y_signal_vec = y_signal.reshape([L,(M+Tg)*E])
    y_test = y_signal[0:test_batch,:,:]
    y_test_vec = y_test.reshape([test_batch,(M+Tg)*E])
    prob_hisps_MMV.y_test_mat = np.transpose(y_test, (2,1,0))
    prob_hisps_MMV.y_test = y_test_vec
    y_test_ex = np.zeros([4*test_batch,E,M+Tg])
    x_test_ex = np.zeros([4*test_batch,E,N*(Tg+1)]).astype(np.float32)
    for iw in range(4):
        x_test_ex[iw*test_batch:(iw+1)*test_batch,:,:] = x_test
        y_test_ex[iw*test_batch:(iw+1)*test_batch,:,:] = y_signal_noiseless[0:test_batch,:,:] + noise_ex[iw*test_batch:(iw+1)*test_batch,:,:]
    prob_hisps_MMV.y_test_ex_mat = np.transpose(y_test_ex, (2,1,0))
    prob_hisps_MMV.x_test_ex_mat = np.transpose(x_test_ex, (2,1,0))
    y_test_ex_vec = y_test_ex.reshape([4*test_batch,(M+Tg)*E])
    x_test_ex_vec = x_test_ex.reshape([4*test_batch,N*(Tg+1)*E])
    prob_hisps_MMV.y_test_ex = y_test_ex_vec
    prob_hisps_MMV.x_test_ex = x_test_ex_vec


    y_sync_noiseless = np.dot(np.reshape(x_sync, (L*E,N)), np.transpose(Ao))
    y_sync_noiseless = y_sync_noiseless.reshape([L,E,M])
    y_sync = y_sync_noiseless + noise[:,:,0:M]
    y_sync_test = y_sync[0:test_batch,:,:]
    y_sync_test_vec = y_sync_test.reshape([test_batch,M*E])
    prob_hisps_MMV.y_sync_test = y_sync_test_vec
    prob_hisps_MMV.y_sync_test_mat = np.transpose(y_sync_test, (2,1,0))
    prob_hisps_MMV.Ao = Ao
    prob_hisps_MMV.x_sync_mat = np.transpose(x_sync, (2,1,0))
    prob_hisps_MMV.y_sync_mat = np.transpose(y_sync, (2,1,0))

    prob_hisps_MMV.x_sync_D = x_sync.reshape([L*E,N])
    prob_hisps_MMV.y_sync_D = y_sync.reshape([L*E,M])
    prob_hisps_MMV.x_sync_test_D = x_sync_test.reshape([test_batch*E,N])
    prob_hisps_MMV.y_sync_test_D = y_sync_test.reshape([test_batch*E,M])
    prob_hisps_MMV.y_signal_D = y_signal.reshape([L*E,M+Tg])
    prob_hisps_MMV.x_D = x_vir_channel.reshape([L*E,N*(Tg+1)])
    prob_hisps_MMV.y_test_D = y_test.reshape([test_batch*E,M+Tg])
    prob_hisps_MMV.x_test_D = x_test.reshape([test_batch*E,N*(Tg+1)])
    prob_hisps_MMV.y_test_ex_D = y_test_ex.reshape([4*test_batch*E,M+Tg])
    prob_hisps_MMV.x_test_ex_D = x_test_ex.reshape([4*test_batch*E,N*(Tg+1)])


    #xgen_ = tf.constant(x_vir_channel_vec, name='x_vir_channel')
    #ygen_ = tf.constant(y_signal_vec, name='y_signal')

    prob_hisps_MMV.xval = x_vir_channel_vec
    prob_hisps_MMV.yval = y_signal_vec
    prob_hisps_MMV.xinit = x_vir_channel_vec
    prob_hisps_MMV.yinit = y_signal_vec
    #prob_hisps_MMV.xgen_ = xgen_
    #prob_hisps_MMV.ygen_ = ygen_
    prob_hisps_MMV.noise_var = noise_var

    return prob_hisps_MMV

def bernoulli_gaussian_hierarchical_sparse_randomlocation_trial(M=10, N=200, L=1000, Tg=3, pnz=0.05, kappa=0, SNR=0):
    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
    if kappa >= 1:
        # create a random operator with a specific condition number
        U, _, V = la.svd(A, full_matrices=False)
        s = np.logspace(0, np.log10(1 / kappa), M)
        A = np.dot(U * (s * np.sqrt(N) / la.norm(s)), V).astype(np.float32)
    A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A = A / A_col_norm
    Ao = A
    A_v = np.zeros([M + Tg, N * (Tg + 1)]).astype(np.float32)
    for i1 in range(N):
        for i2 in range(Tg + 1):
            A_v[i2:(i2 + M), i1 * (Tg + 1) + i2] = A[:, i1]
    A = A_v
    A_ = tf.constant(A, name='A')
    prob_hisps = TFGenerator(A=A, A_=A_, pnz=pnz, kappa=kappa, SNR=SNR, iid=False)
    prob_hisps.name = 'Bernoulli-Gaussian-Hierarchical-Sparse, random A'

    alpha = 15.3
    beta  = 37.6
    lsf_sd = 0.1**(alpha+beta*np.log10(300))
    lsf    = np.random.uniform(low=100, high=1000,  size=N)
    lsf_t  = np.zeros([N*(Tg+1)]).astype(np.float32)
    for il in range(N):
        lsf_t[il] = 0.1**(alpha+beta*np.log10(lsf[il]))
        for iT in range(Tg+1):
            lsf_t[il*(Tg+1)+iT] = lsf[il]
    lsf = lsf/lsf_sd
    lsf_t = lsf_t/lsf_sd

    test_batch = 5000
    bernoulli = np.random.uniform(0, 1, size=(N, L)).astype(np.float32)
    for col in range(L):
        for row in range(N):
            if bernoulli[row][col] < pnz:
                bernoulli[row][col] = 1
            else:
                bernoulli[row][col] = 0
        sum_ber = np.sum(bernoulli[:, col])
        if sum_ber == 0:
            bernoulli[0][col] = 1
    x_channel = np.random.normal(size=(N, L), scale=1).astype(np.float32)
    x_channel = x_channel*lsf.reshape(lsf,[N,1])
    x_sync = np.multiply(x_channel, bernoulli)
    x_sync_test = x_sync[:, 0:test_batch]
    prob_hisps.bernoulli = bernoulli
    ind_test = bernoulli[:, 0:test_batch]
    prob_hisps.ind_test = ind_test
    prob_hisps.x_sync_test = x_sync_test
    noise_var = 1 / np.sqrt(M) * math.pow(10., -SNR / 10.)

    #    bernoulli_ = tf.to_float(tf.random_uniform((N, L)) < pnz)
    #    xgen_ = bernoulli_ * tf.random_normal((N, L))
    #    noise_var = pnz * N / M * math.pow(10., -SNR / 10.)
    #    ygen_ = tf.matmul(A_, xgen_) + tf.random_normal((M, L), stddev=math.sqrt(noise_var))

    user_delay = np.random.random_integers(0, Tg, size=(N, L))
    ud_test = user_delay[:, 0:test_batch]
    prob_hisps.ud_test = ud_test
    prob_hisps.ud = user_delay
    x_vir_channel = np.zeros([N * (Tg + 1), L]).astype(np.float32)
    for iL in range(L):
        for iu in range(N):
            if bernoulli[iu][iL] == 1:
                x_vir_channel[iu*(Tg+1) + user_delay[iu,iL],iL] = x_channel[iu,iL]
    #               print(x_vir_channel[iu][iL])
    x_test = x_vir_channel[:, 0:test_batch]
    noise = np.random.normal(size=(M+Tg, L), scale=noise_var).astype(np.float32)
    SNR_ex = np.array([3, 6, 9, 12])
    sigma_w = np.zeros([4]).astype(np.float32)
    noise_ex = np.zeros([M+Tg, 4*test_batch]).astype(np.float32)
    for iw in range(4):
        sigma_w[iw] = 1 / np.sqrt(M) * math.pow(10., -SNR_ex[iw] / 10.)
        noise_ex[:,iw*test_batch:(iw+1)*test_batch] = sigma_w[iw]/noise_var*noise[:,0:test_batch]


    y_signal_noiseless = np.dot(A_v, x_vir_channel)
    y_signal = y_signal_noiseless + noise
    y_test = y_signal[:, 0:test_batch]
    y_test_ex = np.zeros([M+Tg, 4*test_batch])
    x_test_ex = np.zeros([N*(Tg+1), 4*test_batch]).astype(np.float32)
    for iw in range(4):
        x_test_ex[:,iw*test_batch:(iw+1)*test_batch] = x_test
        y_test_ex[:,iw*test_batch:(iw+1)*test_batch] = y_signal_noiseless[:,0:test_batch] + noise_ex[:,iw*test_batch:(iw+1)*test_batch]
    prob_hisps.y_test_ex = y_test_ex
    prob_hisps.x_test_ex = x_test_ex

    y_sync = np.dot(Ao, x_sync) + noise[0:M, :]
    y_sync_test = y_sync[:, 0:test_batch]
    prob_hisps.y_sync_test = y_sync_test

    prob_hisps.Ao = Ao
    prob_hisps.x_sync = x_sync
    prob_hisps.y_sync = y_sync

    xgen_ = tf.constant(x_vir_channel, name='x_vir_channel')
    ygen_ = tf.constant(y_signal, name='y_signal')

    prob_hisps.x_test = x_test
    prob_hisps.y_test = y_test
    prob_hisps.xval = x_vir_channel
    prob_hisps.yval = y_signal
    prob_hisps.xinit = x_vir_channel
    prob_hisps.yinit = y_signal
    prob_hisps.xgen_ = xgen_
    prob_hisps.ygen_ = ygen_
    prob_hisps.noise_var = noise_var

    return prob_hisps

def bernoulli_gaussian_hierarchical_sparse_MMV_randomlocation_trial(M=10,N=200,E=2,L=200000,Tg=3,pnz=0.05,kappa=0,SNR=0):
    A = np.random.normal(size=(M, N), scale=1.0 / math.sqrt(M)).astype(np.float32)
    if kappa >= 1:
        # create a random operator with a specific condition number
        U, _, V = la.svd(A, full_matrices=False)
        s = np.logspace(0, np.log10(1 / kappa), M)
        A = np.dot(U * (s * np.sqrt(N) / la.norm(s)), V).astype(np.float32)
    A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A = A / A_col_norm
    Ao = A
    #A_col_norm = np.linalg.norm(A, ord=2, axis=0, keepdims=True)
    A_v = np.zeros([M+Tg, N*(Tg+1)]).astype(np.float32)
    for i1 in range(N):
        for i2 in range(Tg+1):
            A_v[i2:(i2 + M), i1 * (Tg + 1) + i2] = A[:, i1]
    A = A_v
    A_ = tf.constant(A, name='A')
    iid = False
    prob_hisps_MMV = TFGenerator(A=A, A_=A_, pnz=pnz, kappa=kappa, SNR=SNR, Ao=Ao, M=M, N=N, Tg=Tg, E=E, iid=iid)
    prob_hisps_MMV.name = 'Bernoulli-Gaussian-Hierarchical-Sparse-MMV, random A'
    prob_hisps_MMV.iid = iid

    alpha = 15.3
    beta  = 37.6
    lsf_sd = 0.1**((alpha+beta*np.log10(500))/10)
    distance = np.random.uniform(low=100, high=1000, size=N).astype(np.float32)
    lsf    = np.zeros([N]).astype(np.float32)
    lsf_t  = np.zeros([N*(Tg+1)]).astype(np.float32)
    for il in range(N):
        lsf[il] = 0.1**((alpha+beta*np.log10(distance[il]))/10)
        for iT in range(Tg+1):
            lsf_t[il*(Tg+1)+iT] = lsf[il]
    lsf = 1.0/lsf_sd*lsf
    lsf_t = 1.0/lsf_sd*lsf_t
    prob_hisps_MMV.lsf = lsf
    prob_hisps_MMV.lsf_t = lsf_t

    test_batch = 5000
    bernoulli = np.random.uniform(0, 1, size=(L, N)).astype(np.float32)
    ind_channel = np.zeros(shape=(L, E, N)).astype(np.float32)
    for height in range(L):
        for row in range(N):
            if bernoulli[height,row] < pnz:
                bernoulli[height,row] = 1.0
                ind_channel[height,:,row] = np.ones(shape=(E)).astype(np.float32)
            else:
                bernoulli[height,row] = 0.0
                ind_channel[height,:,row] = np.zeros(shape=(E)).astype(np.float32)
        sum_ber = np.sum(bernoulli[height, :])
        if sum_ber == 0:
            bernoulli[height,0] = 1.0
            ind_channel[height,:,0] = np.ones(shape=(E)).astype(np.float32)
    prob_hisps_MMV.ind = np.transpose(bernoulli, (1,0))
    x_channel = np.random.normal(size=(L, E, N), scale=1.0).astype(np.float32)
    x_channel = x_channel*lsf.reshape([1,1,N])
    x_sync = np.multiply(x_channel, ind_channel)
    x_sync_test = x_sync[0:test_batch,:,:]
    x_sync_test_vec = x_sync_test.reshape([test_batch,N*E])
    prob_hisps_MMV.x_sync_test_mat = np.transpose(x_sync_test, (2,1,0))
    prob_hisps_MMV.x_sync_test = x_sync_test_vec
    ind_test = bernoulli[0:test_batch,:]
    prob_hisps_MMV.ind_test = np.transpose(ind_test, (1,0))

    user_delay = np.random.random_integers(0, Tg, size=(L, N))
    prob_hisps_MMV.ud = np.transpose(user_delay, (1,0))
    ud_test = user_delay[0:test_batch,:]
    prob_hisps_MMV.ud_test = np.transpose(ud_test, (1,0))

    x_vir_channel = np.zeros([L,E,N*(Tg+1)]).astype(np.float32)
    for iL in range(L):
        for iu in range(N):
            if bernoulli[iL,iu] == 1.0:
                x_vir_channel[iL,:,iu*(Tg+1)+user_delay[iL,iu]] = x_channel[iL,:,iu]
    x_vir_channel_vec = x_vir_channel.reshape([L,N*(Tg+1)*E])
    prob_hisps_MMV.x_mat = np.transpose(x_vir_channel, (2,1,0))
    x_test = x_vir_channel[0:test_batch,:,:]
    prob_hisps_MMV.x_test_mat = np.transpose(x_test, (2,1,0))
    x_test_vec = x_test.reshape([test_batch,N*(Tg+1)*E])
    prob_hisps_MMV.x_test = x_test_vec

    noise_var = 1 / np.sqrt(M) * math.pow(10., -SNR / 10.)
    noise = np.random.normal(size=(L,E,M+Tg), scale=noise_var).astype(np.float32)
    prob_hisps_MMV.noise_mat = np.transpose(noise, (2,1,0))
    SNR_ex = np.array([3, 6, 9, 12])
    sigma_w = np.zeros([4]).astype(np.float32)
    noise_ex = np.zeros([4*test_batch,E,M+Tg]).astype(np.float32)
    for iw in range(4):
        sigma_w[iw] = 1 / np.sqrt(M) * math.pow(10., -SNR_ex[iw] / 10.)
        noise_ex[iw*test_batch:(iw+1)*test_batch,:,:] = sigma_w[iw] / noise_var * noise[0:test_batch,:,:]

    y_signal_noiseless = np.dot(np.reshape(x_vir_channel, (E*L,N*(Tg+1))), np.transpose(A_v))
    y_signal_noiseless = y_signal_noiseless.reshape([L,E,M+Tg])
    y_signal = y_signal_noiseless + noise
    prob_hisps_MMV.y_mat = np.transpose(y_signal, (2,1,0))
    y_signal_vec = y_signal.reshape([L,(M+Tg)*E])
    y_test = y_signal[0:test_batch,:,:]
    y_test_vec = y_test.reshape([test_batch,(M+Tg)*E])
    prob_hisps_MMV.y_test_mat = np.transpose(y_test, (2,1,0))
    prob_hisps_MMV.y_test = y_test_vec
    y_test_ex = np.zeros([4*test_batch,E,M+Tg])
    x_test_ex = np.zeros([4*test_batch,E,N*(Tg+1)]).astype(np.float32)
    for iw in range(4):
        x_test_ex[iw*test_batch:(iw+1)*test_batch,:,:] = x_test
        y_test_ex[iw*test_batch:(iw+1)*test_batch,:,:] = y_signal_noiseless[0:test_batch,:,:] + noise_ex[iw*test_batch:(iw+1)*test_batch,:,:]
    prob_hisps_MMV.y_test_ex_mat = np.transpose(y_test_ex, (2,1,0))
    prob_hisps_MMV.x_test_ex_mat = np.transpose(x_test_ex, (2,1,0))
    y_test_ex_vec = y_test_ex.reshape([4*test_batch,(M+Tg)*E])
    x_test_ex_vec = x_test_ex.reshape([4*test_batch,N*(Tg+1)*E])
    prob_hisps_MMV.y_test_ex = y_test_ex_vec
    prob_hisps_MMV.x_test_ex = x_test_ex_vec

    y_sync_noiseless = np.dot(np.reshape(x_sync, (L*E,N)), np.transpose(Ao))
    y_sync_noiseless = y_sync_noiseless.reshape([L,E,M])
    y_sync = y_sync_noiseless + noise[:,:,0:M]
    y_sync_test = y_sync[0:test_batch,:,:]
    y_sync_test_vec = y_sync_test.reshape([test_batch,M*E])
    prob_hisps_MMV.y_sync_test = y_sync_test_vec
    prob_hisps_MMV.y_sync_test_mat = np.transpose(y_sync_test, (2,1,0))
    prob_hisps_MMV.Ao = Ao
    prob_hisps_MMV.x_sync_mat = np.transpose(x_sync, (2,1,0))
    prob_hisps_MMV.y_sync_mat = np.transpose(y_sync, (2,1,0))

    prob_hisps_MMV.x_sync_D = x_sync.reshape([L*E,N])
    prob_hisps_MMV.y_sync_D = y_sync.reshape([L*E,M])
    prob_hisps_MMV.x_sync_test_D = x_sync_test.reshape([test_batch*E,N])
    prob_hisps_MMV.y_sync_test_D = y_sync_test.reshape([test_batch*E,M])
    prob_hisps_MMV.y_signal_D = y_signal.reshape([L*E,M+Tg])
    prob_hisps_MMV.x_D = x_vir_channel.reshape([L*E,N*(Tg+1)])
    prob_hisps_MMV.y_test_D = y_test.reshape([test_batch*E,M+Tg])
    prob_hisps_MMV.x_test_D = x_test.reshape([test_batch*E,N*(Tg+1)])
    prob_hisps_MMV.y_test_ex_D = y_test_ex.reshape([4*test_batch*E,M+Tg])
    prob_hisps_MMV.x_test_ex_D = x_test_ex.reshape([4*test_batch*E,N*(Tg+1)])


    #xgen_ = tf.constant(x_vir_channel_vec, name='x_vir_channel')
    #ygen_ = tf.constant(y_signal_vec, name='y_signal')

    prob_hisps_MMV.xval = x_vir_channel_vec
    prob_hisps_MMV.yval = y_signal_vec
    prob_hisps_MMV.xinit = x_vir_channel_vec
    prob_hisps_MMV.yinit = y_signal_vec
    #prob_hisps_MMV.xgen_ = xgen_
    #prob_hisps_MMV.ygen_ = ygen_
    prob_hisps_MMV.noise_var = noise_var

    return prob_hisps_MMV

def cbg_hisps_MMV_trial(A0r,A0i,M=40,N=200,E=1,L=105000,Tg=3,pnz=0.05,SNR=0,iid=True):
    # complex-valued system
    Ar = np.zeros([M+Tg,N*(Tg+1)],dtype=np.float32)
    Ai = np.zeros([M+Tg,N*(Tg+1)],dtype=np.float32)
    for i1 in range(N):
        for i2 in range(Tg+1):
            Ar[i2:(i2+M),i1*(Tg+1)+i2] = A0r[:,i1]
            Ai[i2:(i2+M),i1*(Tg+1)+i2] = A0i[:,i1]
    A = np.zeros([2*(M+Tg),2*N*(Tg+1)]).astype(np.float32)
    A[0:(M+Tg),0:(N*(Tg+1))] = Ar
    A[0:(M+Tg),(N*(Tg+1)):(2*N*(Tg+1))] = -Ai
    A[(M+Tg):(2*(M+Tg)),0:(N*(Tg+1))] = Ai
    A[(M+Tg):(2*(M+Tg)),(N*(Tg+1)):(2*N*(Tg+1))] = Ar
    AT = np.transpose(A)
    A0 = np.zeros([2*M,2*N]).astype(np.float32)
    A0[0:M,0:N] = A0r
    A0[0:M,N:(2*N)] = -A0i
    A0[M:(2*M),0:N] = A0i
    A0[M:(2*M),N:(2*N)] = A0r
    A_ = tf.constant(A, name='A')
    prob_hisps_MMV = TFGenerator(A=A, A_=A_, pnz=pnz, SNR=SNR, A0=A0, M=M, N=N, Tg=Tg, E=E, iid=iid)
    prob_hisps_MMV.name = 'Complex-Bernoulli-Gaussian-Hierarchical-Sparse-MMV, random A'
    prob_hisps_MMV.A = A
    prob_hisps_MMV.iid = iid
    prob_hisps_MMV.M = M
    prob_hisps_MMV.N = N
    prob_hisps_MMV.E = E
    prob_hisps_MMV.Tg = Tg
    prob_hisps_MMV.pnz = pnz
    prob_hisps_MMV.Mt = M+Tg
    prob_hisps_MMV.Nt = N*(Tg+1)

    # large-scale fading
    if iid == False:
        alpha = 15.3
        beta = 37.6
        lsf_sd = 0.1 ** ((alpha + beta * np.log10(150)) / 10)
        distance = np.random.uniform(low=50, high=250, size=N).astype(np.float32)
        lsf = np.zeros([N]).astype(np.float32)
        lsf_t = np.zeros([N*(Tg+1)]).astype(np.float32)
        for il in range(N):
            lsf[il] = 0.1**( (alpha + beta*np.log10(distance[il]))/10 )
            for iT in range(Tg+1):
                lsf_t[il*(Tg+1)+iT] = lsf[il]
        lsf = np.sqrt(1.0/lsf_sd*lsf)
        lsf_t = np.sqrt(1.0/lsf_sd*lsf_t)
    else:
        lsf = np.ones([N]).astype(np.float32)
        lsf_t = np.ones([N*(Tg+1)]).astype(np.float32)
    prob_hisps_MMV.lsf = lsf
    prob_hisps_MMV.lsf_t = lsf_t

    # channel
    naep = np.floor(pnz*N).astype(np.int32)
    test_batch = 5000
    bernoulli = (np.random.uniform(0,1,size=(L,N))<pnz).astype(np.float32)
    nau = np.sum(bernoulli,axis=1)
    for iL in range(test_batch):
        if nau[iL] == 0:
            bernoulli[iL,0:naep] = np.ones([naep]).astype(np.float32)
    oneE = np.ones([E],dtype=np.float32)
    ind_channel = bernoulli.reshape([L,1,N]) * np.reshape(oneE,(1,E,1))
    prob_hisps_MMV.ind = np.transpose(bernoulli, (1,0))
    x_channel_r = np.sqrt(0.5)*np.random.normal(size=(L,E,N),scale=1.0).astype(np.float32)
    x_channel_i = np.sqrt(0.5)*np.random.normal(size=(L,E,N),scale=1.0).astype(np.float32)
    x_channel_r = x_channel_r*lsf.reshape([1,1,N])
    x_channel_i = x_channel_i*lsf.reshape([1,1,N])
    x_sync_r = np.multiply(x_channel_r, ind_channel)
    x_sync_i = np.multiply(x_channel_i, ind_channel)
    x_sync = np.zeros([L,E,2*N]).astype(np.float32)
    x_sync[:,:,0:N] = x_sync_r
    x_sync[:,:,N:(2*N)] = x_sync_i
    x_sync_test = x_sync[0:test_batch,:,:]
    #x_sync_test_vec = x_sync_test.reshape([test_batch,2*N*E])
    prob_hisps_MMV.x_sync_test_mat = np.transpose(x_sync_test, (2,1,0))
    prob_hisps_MMV.x_sync_test = x_sync_test
    prob_hisps_MMV.x_sync_train = x_sync[test_batch:L,:,:]
    ind_test = bernoulli[0:test_batch,:]
    prob_hisps_MMV.ind_test = np.transpose(ind_test,(1,0))

    user_delay = np.random.random_integers(0,Tg,size=(L,N))
    prob_hisps_MMV.ud = np.transpose(user_delay,(1,0))
    ud_test = user_delay[0:test_batch,:]
    prob_hisps_MMV.ud_test = np.transpose(ud_test,(1,0))

    x_vir_channel = np.zeros([L,E,2*N*(Tg+1)]).astype(np.float32)
    x_vir_channel_r = np.zeros([L,E,N*(Tg+1)],dtype=np.float32)
    x_vir_channel_i = np.zeros([L,E,N*(Tg+1)],dtype=np.float32)
    for iL in range(L):
        for iu in range(N):
            if bernoulli[iL,iu] == 1.0:
                index1 = iu*(Tg+1)+user_delay[iL,iu]
                index2 = N*(Tg+1)+iu*(Tg+1)+user_delay[iL,iu]
                x_vir_channel[iL,:,index1] = x_channel_r[iL,:,iu]
                x_vir_channel[iL,:,index2] = x_channel_i[iL,:,iu]
                x_vir_channel_r[iL,:,index1] = x_channel_r[iL,:,iu]
                x_vir_channel_i[iL,:,index1] = x_channel_i[iL,:,iu]
    #x_vir_channel_vec = x_vir_channel.reshape([L,N*(Tg+1)*E])
    #prob_hisps_MMV.x_mat = np.transpose(x_vir_channel, (2,1,0))
    x_test = x_vir_channel[0:test_batch,:,:]
    prob_hisps_MMV.x_test_mat = np.transpose(x_test, (2,1,0))
    #x_test_vec = x_test.reshape([test_batch,N*(Tg+1)*E])
    prob_hisps_MMV.x_test = x_test.reshape([test_batch,E*2*N*(Tg+1)])
    x_train = x_vir_channel[test_batch:L,:,:]
    prob_hisps_MMV.x_train = x_train.reshape([-1,E*2*N*(Tg+1)])

    # noise and ex signal with various SNR
    noise_var = 1/np.sqrt(M)*np.sqrt(math.pow(10., -SNR/10.))
    noise = noise_var*np.sqrt(0.5)*np.random.normal(size=(L,E,2*(M+Tg)), scale=1.0).astype(np.float32)
    prob_hisps_MMV.noise_mat = np.transpose(noise, (2,1,0))
    SNR_ex = np.array([2,4,6,8])
    sigma_w = np.zeros([4]).astype(np.float32)
    noise_ex = np.zeros([4*test_batch,E,2*(M+Tg)]).astype(np.float32)
    for iw in range(4):
        sigma_w[iw] = 1/np.sqrt(M)*math.pow(10., -SNR_ex[iw]/10.)
        noise_ex[(iw*test_batch):((iw+1)*test_batch),:,:] = sigma_w[iw]/noise_var*noise[0:test_batch,:,:]

    # signal y in sync system (seems useless)
    y_sync_noiseless = np.dot(np.reshape(x_sync,(L*E,2*N)), np.transpose(A0))
    y_sync_noiseless = y_sync_noiseless.reshape([L,E,2*M])
    noise_sync = np.zeros([L,E,2*M])
    noise_sync[:,:,0:M] = noise[:,:,0:M]
    noise_sync[:,:,M:(2*M)] = noise[:,:,(M+Tg):(2*M+Tg)]
    y_sync = y_sync_noiseless + noise_sync
    y_sync_test = y_sync[0:test_batch,:,:]
    y_sync_train = y_sync[test_batch:L,:,:]
    #y_sync_test_vec = y_sync_test.reshape([test_batch,M*E])
    prob_hisps_MMV.y_sync_test = y_sync_test.reshape([test_batch,E*2*M])
    prob_hisps_MMV.y_sync_train = y_sync_train.reshape([-1,E*2*M])
    prob_hisps_MMV.y_sync_test_mat = np.transpose(y_sync_test, (2,1,0))
    prob_hisps_MMV.A0 = A0
    #prob_hisps_MMV.x_sync_mat = np.transpose(x_sync, (2,1,0))
    #prob_hisps_MMV.y_sync_mat = np.transpose(y_sync, (2,1,0))

    # signal y in async system
    y_signal_noiseless = np.dot(np.reshape(x_vir_channel, (E*L,2*N*(Tg+1))), AT)
    y_signal_noiseless = y_signal_noiseless.reshape([L,E,2*(M+Tg)])
    y_signal = y_signal_noiseless + noise
    #prob_hisps_MMV.y_mat = np.transpose(y_signal, (2,1,0))
    #y_signal_vec = y_signal.reshape([L,(M+Tg)*E])
    y_test = y_signal[0:test_batch,:,:]
    #y_test_vec = y_test.reshape([test_batch, (M+Tg)*E])
    prob_hisps_MMV.y_test_mat = np.transpose(y_test, (2,1,0))
    prob_hisps_MMV.y_test = y_test.reshape(test_batch,E*2*(M+Tg))
    y_train = y_signal[test_batch:L,:,:]
    prob_hisps_MMV.y_train = y_train.reshape([-1,E*2*(M+Tg)])
    y_test_ex = np.zeros([4*test_batch,E,2*(M+Tg)])
    x_test_ex = np.zeros([4*test_batch,E,2*N*(Tg+1)]).astype(np.float32)
    for iw in range(4):
        x_test_ex[(iw*test_batch):((iw+1)*test_batch),:,:] = x_test
        y_test_ex[(iw*test_batch):((iw+1)*test_batch),:,:] = y_signal_noiseless[0:test_batch,:,:] + noise_ex[(iw*test_batch):((iw+1)*test_batch),:,:]
    prob_hisps_MMV.y_test_ex_mat = np.transpose(y_test_ex, (2,1,0))
    prob_hisps_MMV.x_test_ex_mat = np.transpose(x_test_ex, (2,1,0))
    #y_test_ex_vec = y_test_ex.reshape([5*test_batch, (M+Tg)*E])
    #x_test_ex_vec = x_test_ex.reshape([5*test_batch, N*(Tg+1)*E])
    prob_hisps_MMV.y_test_ex = y_test_ex.reshape([-1,E*2*(M+Tg)])
    prob_hisps_MMV.x_test_ex = x_test_ex.reshape([-1,E*2*N*(Tg+1)])

    Ac = Ar+1j*Ai
    xc = np.reshape(x_vir_channel_r+1j*x_vir_channel_i, (-1,N*(Tg+1)))
    AcH = np.transpose(Ac)
    y1 = np.transpose(np.matmul(xc,AcH))
    y1r = y1.real
    y1i = y1.imag
    y2r = y_signal_noiseless[:,:,0:(M+Tg)]
    y2i = y_signal_noiseless[:,:,(M+Tg):2*(M+Tg)]
    # aa1 = y1r-y2r
    # aa2 = y1i-y2i

    # LAMP-D is considered
    prob_hisps_MMV.y_train_D = np.reshape(y_signal[test_batch:L,:,:], (-1,2*(M+Tg)))
    prob_hisps_MMV.x_train_D = np.reshape(x_vir_channel[test_batch:L,:,:], (-1,2*N*(Tg+1)))
    prob_hisps_MMV.y_test_D = y_test.reshape([test_batch*E,2*(M+Tg)])
    prob_hisps_MMV.x_test_D = x_test.reshape([test_batch*E,2*N*(Tg+1)])
    prob_hisps_MMV.y_test_ex_D = y_test_ex.reshape([4*test_batch*E,2*(M+Tg)])
    prob_hisps_MMV.x_test_ex_D = x_test_ex.reshape([4*test_batch*E,2*N*(Tg+1)])

    # LAMP-H is considered
    prob_hisps_MMV.y_train_H = np.reshape(y_signal[test_batch:L,:,:], (-1,2*2*(M+Tg)))
    prob_hisps_MMV.x_train_H = np.reshape(x_vir_channel[test_batch:L,:,:], (-1,2*2*N*(Tg+1)))
    prob_hisps_MMV.y_test_H = y_test.reshape([-1,2*2*(M+Tg)])
    prob_hisps_MMV.x_test_H = x_test.reshape([-1,2*2*N*(Tg+1)])
    prob_hisps_MMV.y_test_ex_H = y_test_ex.reshape([-1,2*2*(M+Tg)])
    prob_hisps_MMV.x_test_ex_H = x_test_ex.reshape([-1,2*2*N*(Tg+1)])

    prob_hisps_MMV.xval = x_vir_channel.reshape([L*E,-1])
    prob_hisps_MMV.yval = y_signal.reshape([L*E,-1])
    prob_hisps_MMV.xinit = x_vir_channel.reshape([L*E,-1])
    prob_hisps_MMV.yinit = y_signal.reshape([L*E,-1])
    #prob_hisps_MMV.xgen_ = xgen_
    #prob_hisps_MMV.ygen_ = ygen_
    prob_hisps_MMV.noise_var = noise_var

    return prob_hisps_MMV


def random_access_problem(which=1):
    #    import raputil as ru
    if which == 1:
        opts = ru.Problem.scenario1()
    else:
        opts = ru.Problem.scenario2()

    p = ru.Problem(**opts)
    x1 = p.genX(1)
    y1 = p.fwd(x1)
    A = p.S
    M, N = A.shape
    nbatches = int(math.ceil(1000 / x1.shape[1]))
    prob = NumpyGenerator(p=p, nbatches=nbatches, A=A, opts=opts, iid=(which == 1))
    if which == 2:
        prob.maskX_ = tf.expand_dims(tf.constant((np.arange(N) % (N // 2) < opts['Nu']).astype(np.float32)), 1)

    _, prob.noise_var = p.add_noise(y1)

    unused = p.genYX(nbatches)  # for legacy reasons -- want to compare against a previous run
    (prob.yval, prob.xval) = p.genYX(nbatches)
    (prob.yinit, prob.xinit) = p.genYX(nbatches)
    import multiprocessing as mp
    prob.nsubprocs = mp.cpu_count()
    return prob
