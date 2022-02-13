#!/usr/bin/python
from tools import problems
import hierarchical_sparse

import numpy as np
import numpy.linalg as la
import os
from tools import raputil as ru

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
import tensorflow as tf

# import our problems, networks and training modules
from tools import problems,networks,train

from scipy.io import savemat

def save_problem_MMV(base,prob):
    print('saving {b}.mat,{b}.npz norm(x)={x:.7f} norm(y)={y:.7f}'.format(b=base,x=la.norm(prob.xval), y=la.norm(prob.yval) ) )
    D = dict(A=prob.A, Ao=prob.Ao, x_test=prob.x_test, y_test=prob.y_test,
             x_sync_test=prob.x_sync_test, y_sync_test=prob.y_sync_test, ud_test=prob.ud_test,
             ind_test=prob.ind_test, y_test_ex=prob.y_test_ex, x_test_mat=prob.x_test_mat,
             y_test_mat=prob.y_test_mat, x_sync_test_mat=prob.x_sync_test_mat, y_sync_test_mat=prob.y_sync_test_mat,
             y_test_ex_mat=prob.y_test_ex_mat, lsf=prob.lsf, lsf_t=prob.lsf_t)
#    D = dict(A=prob.A, x=prob.xval, y=prob.yval, ind=prob.ind, ud=prob.ud,
#             Ao=prob.Ao, x_test=prob.x_test, y_test=prob.y_test,
#             x_sync_test=prob.x_sync_test, y_sync_test=prob.y_sync_test, ud_test=prob.ud_test,
#             ind_test=prob.ind_test, y_test_ex=prob.y_test_ex, noise_mat=prob.noise_mat,
#             x_mat=prob.x_mat, y_mat=prob.y_mat, x_sync_mat=prob.x_sync_mat, y_sync_mat=prob.y_sync_mat,
#             x_test_mat=prob.x_test_mat, y_test_mat=prob.y_test_mat, x_sync_test_mat=prob.x_sync_test_mat,
#             y_sync_test_mat=prob.y_sync_test_mat,
#             y_test_ex_mat=prob.y_test_ex_mat)
    np.savez( base + '.npz', **D)
    #np.savez(base + '.npz', prob)
    savemat(base + '.mat',D,oned_as='column')



def save_cproblem_MMV(base1,base2,prob):
    print('saving {b}.mat, {c}.mat norm(x)={x:.7f} norm(y)={y:.7f}'.format(b=base1,c=base2,x=la.norm(prob.xval), y=la.norm(prob.yval) ) )
    D1 = dict(A=prob.A, A0=prob.A0, ind_test=prob.ind_test, ud_test=prob.ud_test,
             y_train=prob.y_train, x_train=prob.x_train, y_test=prob.y_test, x_test=prob.x_test,
             y_test_ex=prob.y_test_ex, x_test_ex=prob.x_test_ex,
             y_test_ex_mat=prob.y_test_ex_mat, x_test_ex_mat=prob.x_test_ex_mat,
             y_test_mat=prob.y_test_mat, x_test_mat=prob.x_test_mat, lsf=prob.lsf, lsf_t=prob.lsf_t,
             y_train_D=prob.y_train_D, x_train_D=prob.x_train_D,
             y_test_D=prob.y_test_D, x_test_D=prob.x_test_D,
             y_test_ex_D=prob.y_test_ex_D, x_test_ex_D=prob.x_test_ex_D,
             y_train_H=prob.y_train_H, x_train_H=prob.x_train_H,
             y_test_H=prob.y_test_H, x_test_H = prob.x_test_H,
             y_test_ex_H=prob.y_test_ex_H, x_test_ex_H = prob.x_test_ex_H
             )
    D2 = dict(A=prob.A, A0=prob.A0, ind_test=prob.ind_test, ud_test=prob.ud_test,
             y_test=prob.y_test, x_test=prob.x_test,
             y_test_ex=prob.y_test_ex, x_test_ex=prob.x_test_ex,
             y_test_ex_mat=prob.y_test_ex_mat, x_test_ex_mat=prob.x_test_ex_mat,
             y_test_mat=prob.y_test_mat, x_test_mat=prob.x_test_mat, lsf=prob.lsf, lsf_t=prob.lsf_t)
    savemat(base1+'.mat',D1,oned_as='column')
    savemat(base2+'.mat',D2,oned_as='column')

np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)
# system setting
M = 9
N = 200
Tg = 0
E = 1
L = 105000
pnz = 0.005
SNR = 0
iid = True
A1 = np.random.normal(size=(2*M,N),scale=1.0).astype(np.float32)
A_col_norm = np.linalg.norm(A1, ord=2, axis=0, keepdims=True)
A1 = A1/A_col_norm
A1r = A1[0:M,:]
A1i = A1[M:(2*M),:]



#save_problem('problem_hierarchical_sparse', hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_trial(M=10,N=200,L=1000,pnz=0.05,kappa=0,SNR=0))
##save_problem_MMV('Data_train/1_Test_MAT_problem_hisps_MMV_training_M80_N200_E4_Tg3_SNR0_2e4train_2e4test',
##                 hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_MMV_trial(M=80,N=200,E=4,L=20000,Tg=3,pnz=0.05,kappa=0,SNR=0))

# real system generation
##save_problem_MMV('Data_train/1_Test_MAT_problem_random_hisps_MMV_training_M80_N200_E1_Tg3_SNR0_1e5train_5e3test',
##                 hierarchical_sparse.bernoulli_gaussian_hierarchical_sparse_MMV_randomlocation_trial(M=80,N=200,E=1,L=105000,Tg=3,pnz=0.05,kappa=0,SNR=3.9))

# complex system generation
save_cproblem_MMV('CMMV_Data/2MAT_cproblem_ESS_M{M}_N{N}_E{E}_Tg{T}_SNR{S}_train1e5'.format(M=M,N=N,E=E,T=Tg,S=SNR),
                  'CMMV_Data/2MAT_cproblem_ESS_M{M}_N{N}_E{E}_Tg{T}_SNR{S}_test5e3'.format(M=M,N=N,E=E,T=Tg,S=SNR),
                  hierarchical_sparse.cbg_hisps_MMV_trial(A0r=A1r,A0i=A1i,M=M,N=N,E=E,L=L,Tg=Tg,pnz=pnz,SNR=SNR,iid=iid))
print('done!')