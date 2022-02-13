#!/usr/bin/python
import numpy as np
import numpy.linalg as la
import re
import math
import tensorflow as tf

__doc__ = """
This file contains various separable shrinkage functions for use in TensorFlow.
All functions perform shrinkage toward zero on each elements of an input vector
    r = x + w, where x is sparse and w is iid Gaussian noise of a known variance rvar

All shrink_* functions are called with signature

    xhat,dxdr = func(r,rvar,theta)

Hyperparameters are supplied via theta (which has length ranging from 1 to 5)
    shrink_soft_threshold : 1 or 2 parameters
    shrink_bgest : 2 parameters
    shrink_expo : 3 parameters
    shrink_spline : 3 parameters
    shrink_piecwise_linear : 5 parameters

A note about dxdr:
    dxdr is the per-column average derivative of xhat with respect to r.
    So if r is in Real^(NxL),
    then xhat is in Real^(NxL)
    and dxdr is in Real^L
"""

def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def auto_gradients(xhat , r ):
    """Return the per-column average gradient of xhat xhat with respect to r.
    """
    dxdr = tf.gradients(xhat,r)[0]
    dxdr = tf.reduce_mean(dxdr,0)
    minVal=.5/int(r.get_shape()[0])
    dxdr = tf.maximum( dxdr, minVal)
    return dxdr

def shrink_soft_threshold(r,rvar,theta):
    """
    soft threshold function
        y=sign(x)*max(0,abs(x)-theta[0]*sqrt(rvar) )*scaling
    where scaling is theta[1] (default=1)
    in other words, if theta is len(1), then the standard
    """
    if len(theta.get_shape())>0 and theta.get_shape() != (1,):
        lam = theta[0] * tf.sqrt(rvar)
        scale=theta[1]
    else:
        lam  = theta * tf.sqrt(rvar)
        scale = None
    lam = tf.maximum(lam,0)
    arml = tf.abs(r) - lam
    xhat = tf.sign(r) * tf.maximum(arml,0)
    dxdr = tf.reduce_mean(tf.to_float(arml>0),0)
    if scale is not None:
        xhat = xhat*scale
        dxdr = dxdr*scale
    return (xhat,dxdr)

def shrink_soft_threshold_V2(r,rvar,theta):
    """
    soft threshold function
        y=sign(x)*max(0,abs(x)-theta[0]*sqrt(rvar) )*scaling
    where scaling is theta[1] (default=1)
    in other words, if theta is len(1), then the standard
    """
    lam  = theta * tf.sqrt(rvar)
    lam = tf.maximum(lam,0)
    arml = tf.abs(r) - lam
    xhat = tf.sign(r) * tf.maximum(arml,0)
    dxdr = tf.reduce_mean(tf.to_float(arml>0),1)

    return (xhat,dxdr)

def shrink_soft_threshold_MMV(r,rvar,theta,E,N):
    """
    soft threshold function
        y=sign(x)*max(0,abs(x)-theta[0]*sqrt(rvar) )*scaling
    where scaling is theta[1] (default=1)
    in other words, if theta is len(1), then the standard
    """
    lam  = theta * tf.sqrt(tf.to_float(E)) * tf.sqrt(rvar)
    lam = tf.maximum(lam,0)
    rp  = tf.sqrt(tf.reduce_sum(r**2,1))
    arml = rp - lam
    xhat1 = r / tf.reshape(rp,(-1,1,N)) * tf.reshape(tf.maximum(arml,0),(-1,1,N))
    xhat = tf.reshape(xhat1, (-1,E*N))
    # calculation of the derivative of the shrinkage function
    dxdr = tf.reduce_mean(tf.reshape(tf.to_float(arml>0),(-1,N,1,1))*tf.reshape(tf.eye(E),(1,1,E,E)),1)

    return (xhat,dxdr)

def shrink_stSSest(r,rvar,theta,pt, N, Tg):

    abs_r = tf.abs(r)
    thres = tf.contrib.distributions.percentile(abs_r, float(100.0-pt), axis=0, keep_dims=True)

    #batch_size = r.get_shape().as_list()[1]
    #r_c = np.zeros([N, ])

    index_r = tf.logical_and(abs_r > 0, abs_r > thres)
    index_r = tf.to_float(index_r)
    index_r = tf.stop_gradient(index_r)
    cindex_r = 1.0 - index_r

    xhat1 = tf.multiply(index_r, r)
    dxdr1 = tf.reduce_mean(index_r, 0)

    (xhat2, dxdr2) = shrink_soft_threshold(tf.multiply(cindex_r, r), rvar, theta)
    xhat = xhat1 + xhat2
    dxdr = dxdr1 + dxdr2

    return (xhat, dxdr)

def shrink_bgest(r,rvar,theta):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    xvar1 = abs(theta[...,0])
    loglam = theta[...,1] # log(1/lambda - 1)
    beta = 1/(1+rvar/xvar1)
    r2scale = r*r*beta/rvar
    rho = tf.exp(loglam - .5*r2scale ) * tf.sqrt(1 +xvar1/rvar)
    rho1 = rho+1
    xhat = beta*r/rho1

    # computation of the derivative of the shrinkage function
    dxdr = beta*((1+rho*(1+r2scale) ) / tf.square( rho1 ))
    dxdr = tf.reduce_mean(dxdr,0)
    return (xhat,dxdr)

def shrink_bgest_V2(r,rvar,theta):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    xvar1 = abs(theta[0,...])
    loglam = theta[1,...] # log(1/lambda - 1)
    beta = 1/(1+rvar/xvar1)
    r2scale = r*r*beta/rvar
    rho = tf.exp(loglam - .5*r2scale ) * tf.sqrt(1+xvar1/rvar)
    rho1 = rho+1
    xhat = beta*r/rho1


    dxdr = beta*( (rho1+rho*r2scale) / tf.square( rho1 ))
    dxdr = tf.reduce_mean(dxdr,1)
    return (xhat,dxdr)

def shrink_bgSSest(r,rvar,theta,pt):

    abs_r = tf.abs(r)
    thres = tf.contrib.distributions.percentile(abs_r, float(100.0-pt), axis=0, keep_dims=True)

    index_r = tf.logical_and(abs_r > 0, abs_r > thres)
    index_r = tf.to_float(index_r)
    index_r = tf.stop_gradient(index_r)
    cindex_r = 1.0 - index_r

    xhat1 = tf.multiply(index_r, r)
    dxdr1 = tf.reduce_mean(index_r, 0)

    (xhat2, dxdr2) = shrink_bgest(tf.multiply(cindex_r, r), rvar, theta)
    xhat = xhat1 + xhat2
    dxdr = dxdr1 + dxdr2

    return (xhat, dxdr)

def shrink_bgest_MMV(r,rvar,theta,E,N):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    #rvar = tf.reshape(rvar, (-1, 1))
    rvar = tf.reshape(rvar, (-1,1,1))
    xvar1 = tf.reshape(abs(theta[0,...]), (1,1,-1))
    loglam = tf.reshape(theta[1,...], (1,1,-1)) # log(1/lambda - 1)
    beta = 1/(1+rvar/xvar1)

    # computation of xhat
    r2scale = tf.reduce_sum(r*r, axis=1, keep_dims=True)*beta/rvar
    rho = tf.exp(loglam - .5*r2scale ) * (tf.sqrt(1+xvar1/rvar)**E)
    rho1 = rho+1
    rho2 = 1.0/rho1*beta
    xhat1 = r * tf.reshape(rho2, (-1,1,N))
    xhat = tf.reshape(xhat1, (-1,N*E))

    # computation of derivative of shrinkage function to calculate the Onsager term
    r_t = tf.transpose(r, (0,2,1))
    r_re1 = tf.reshape(r_t, (-1,N,E,1))
    r_re2 = tf.reshape(r_t, (-1,N,1,E))
    ueye = tf.eye(E)
    nmrt = tf.reshape(rho1, (-1,N,1,1))*tf.reshape(ueye, (1,1,E,E)) + tf.reshape(rho*(beta/rvar), (-1,N,1,1))*(r_re1*r_re2)
    dxdr_s = nmrt / tf.reshape(tf.square(rho1), (-1,N,1,1))
    dxdr1 = tf.reduce_mean(dxdr_s, axis=1)
    dxdr2 = dxdr1 * tf.reshape(beta, (-1,1,1))
    dxdr = dxdr2
    return (xhat,dxdr)

def shrink_csoft_threshold(r,rvar,theta,thetaex,Nt,Nu,E,Tg):
    tex0 = thetaex[0]
    tex1 = thetaex[1]
    tex2 = thetaex[2]

    r = tf.reshape(r, (-1,2*E,Nt))
    rvar = tf.reshape(rvar, (-1,1,1))
    ts = tf.to_float(E)*rvar
    # same large-scale fading
    ts = theta*tf.sqrt(ts)
    # various large-scale fading
    # theta1 = tf.reshape(tf.reshape(theta, (-1,1))*tf.ones([1,Tg+1],dtype=tf.float32), (1,1,-1))
    # ts = theta*ts
    ts = tf.maximum(ts,0)

    rp = tf.sqrt(tf.reduce_sum(r*r,axis=1,keepdims=True))
    rmts = rp-ts
    rmtsI = tf.maximum(rmts,0)
    xhat1 = r*rmtsI/rp
    xhat = tf.reshape(xhat1, (-1,2*E*Nt))

    teyec = tf.reshape(tf.concat([tf.eye(E),tf.zeros([E,E],dtype=tf.float32)],0), (1,-1,E))
    dxdr1 = tf.reduce_mean(tf.to_float(rmts>0),axis=2,keepdims=True)
    dxdr  = tf.reshape(dxdr1,(-1,1,1))*teyec

    return  (xhat,dxdr)

def shrink_cbg_MMV(r,rvar,theta,thetaex,Nt,Nu,E,Tg):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    # various large-scale fading
    #rvar = tf.reshape(rvar, (-1, 1))
    #Nu = (N/(Tg+1)/2).astype(np.int32)
    #Nt = (N/2).astype(np.int32)

    # # same large-scale fading
    #     # xvar1 = tf.reshape(abs(theta[0,...]), (1,1,-1))
    #     # loglam = tf.reshape(theta[1,...], (1,1,-1)) # (1-pa)/pa*(Tg+1)
    #     # tex0 = thetaex[0]
    #     # tex1 = thetaex[1]
    #     # tex2 = thetaex[2]
    #     # rvar = tf.reshape(rvar, (-1,1,1))
    #     # beta = xvar1/(xvar1+rvar)
    #     # r = tf.reshape(r,(-1,2*E,Nt))
    #     #
    #     # # computation of xhat
    #     # # c11 = 1+xvar1/rvar
    #     # # c12 = rvar/(xvar1+rvar)
    #     # # q1 = (c11**E)*tf.exp(-tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    #     # # q1_inv = ((c12)**E)*tf.exp(tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    #     # # q1_inv1 = tf.reshape(q1_inv, (-1,1,Nu,Tg+1))
    #     # # t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    #     # # q2 = tf.reshape( (tf.reduce_sum(q1_inv1,axis=3,keepdims=True)*t1), (-1,1,Nt) ) + loglam
    #     # # Q = q1*q2
    #     # # Q_inv = tf.maximum(Q,(1e-10))
    #     # # xhat1 = r/Q*beta
    #     # c11 = (1+xvar1/rvar)**E
    #     # q1e = tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar
    #     # q1e = tf.maximum(q1e,float(-40))
    #     # q1e = tf.minimum(q1e,float(40))
    #     # q1  = tf.exp(q1e)
    #     # q11 = tf.reshape(q1, (-1,1,Nu,Tg+1))
    #     # t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    #     # Q = tf.reshape( (tf.reduce_sum(q11, axis=3, keepdims=True)*t1),(-1,1,Nt) ) + c11*loglam
    #     # q1Q = q1/Q
    #     # #q1Q = tf.maximum(q1Q,1e-10)
    #     # xhat1 = r*q1Q*beta
    #     # xhat1 = tex0*xhat1 - tex1*r
    #     # xhat = tf.reshape(xhat1, (-1,2*E*Nt))
    #     #
    #     # # computation of the derivate of the shrinkage function
    #     # rt = tf.reshape(r, (-1,E,2,Nt))
    #     # rt11 = tf.transpose(rt, (0,3,1,2))
    #     # rt11_rc = tf.reshape(rt11[:,:,:,0], (-1,Nt,E,1))
    #     # rt11_rr = tf.reshape(rt11_rc, (-1,Nt,1,E))
    #     # rt11_ic = tf.reshape(rt11[:,:,:,1], (-1,Nt,E,1))
    #     # rt11_ir = tf.reshape(rt11_ic, (-1,Nt,1,E))
    #     # # rhr: derivative of the eta function (B in the paper)
    #     # # B = [[rhr_r], [rhr_i]]
    #     # rhr_r = rt11_rc*rt11_rr + rt11_ic*rt11_ir
    #     # rhr_i = rt11_rc*rt11_ir - rt11_ic*rt11_rr
    #     # rhr = tf.concat([rhr_r,rhr_i],2)
    #     # rhr = rhr*tf.reshape(beta/rvar,(-1,1,1,1))
    #     # teyec = tf.reshape(tf.concat([tf.eye(E),tf.zeros([E,E],dtype=tf.float32)],0), (1,1,-1,E))
    #     # q1Q4 = tf.reshape(q1Q,(-1,Nt,1,1))
    #     # dxdr1 = (teyec*q1Q4 + rhr*q1Q4 - rhr*(q1Q4**2))*tf.reshape(beta,(-1,1,1,1))
    #     # dxdr2 = tex2*(tex0*dxdr1 - tex1*teyec)
    #     # # dxdr1 = ( teyec/tf.reshape(Q,(-1,Nt,1,1))+tf.reshape((1/Q-1/(Q)**2)*beta/rvar,(-1,Nt,1,1))*rhr )*tf.reshape(beta,(-1,Nt,1,1))
    #     # dxdr = tf.reduce_mean(dxdr2,axis=1,keepdims=False)


    # #various large-scale fading
    xvar1 = tf.reshape(tf.reshape(abs(theta[0,...]), (-1,1))*tf.ones([1,Tg+1],dtype=np.float32), (1,1,-1))
    loglam = tf.reshape(tf.reshape(abs(theta[1,...]), (-1,1))*tf.ones([1,Tg+1],dtype=np.float32), (1,1,-1))
    tex0 = thetaex[0]
    tex1 = thetaex[1]
    tex2 = thetaex[2]
    rvar = tf.reshape(rvar,(-1,1,1))
    beta = xvar1/(xvar1+rvar)
    r = tf.reshape(r, (-1,2*E,Nt))

    # computation of xhat
    r2scale = tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar
    rho = tf.exp(loglam-r2scale)*((1+xvar1/rvar)**E)
    rho1 = rho+1
    rho2 = 1.0/rho1*beta
    xhat1 = tex0*r*rho2 - tex1*r
    xhat = tf.reshape(xhat1, (-1,2*E*Nt))

    # computation of derivative of shrinkage function to calculate the Onsager term
    rt = tf.reshape(r, (-1,E,2,Nt))
    rt11 = tf.transpose(rt, (0,3,1,2))
    rt11_rc = tf.reshape(rt11[:,:,:,0], (-1,Nt,E,1))
    rt11_rr = tf.reshape(rt11_rc, (-1,Nt,1,E))
    rt11_ic = tf.reshape(rt11[:,:,:,1], (-1,Nt,E,1))
    rt11_ir = tf.reshape(rt11_ic, (-1,Nt,1,E))
    rhr_r = rt11_rc*rt11_rr + rt11_ic*rt11_ir
    rhr_i = rt11_rc*rt11_ir - rt11_ic*rt11_rr
    rhr = tf.concat([rhr_r,rhr_i], 2)
    rhr = rhr*tf.reshape(beta/rvar, (-1,Nt,1,1))
    teyec = tf.reshape(tf.concat([tf.eye(E),tf.zeros([E,E], dtype=tf.float32)], 0), (1,1,-1,E))
    dxdr1 = tf.reshape(1/rho1,(-1,Nt,1,1))*teyec + tf.reshape(rho/(rho1**2),(-1,Nt,1,1))*rhr
    dxdr1 = tf.reshape(beta,(-1,Nt,1,1))*dxdr1
    dxdr2 = tex2*(tex0*dxdr1 - tex1*teyec)
    dxdr = tf.reduce_mean(dxdr2, axis=1, keepdims=False)

    return (xhat, dxdr)

def shrink_sbgest_MMV(r,rvar,theta,thetaex,Nt,Nu,E,Tg):
    """Bernoulli-Gaussian MMSE estimator
    Perform MMSE estimation E[x|r]
    for x ~ BernoulliGaussian(lambda,xvar1)
        r|x ~ Normal(x,rvar)

    The parameters theta[0],theta[1] represent
        The variance of non-zero x[i]
            xvar1 = abs(theta[0])
        The probability of nonzero x[i]
            lamba = 1/(exp(theta[1])+1)
    """
    # various large-scale fading
    #rvar = tf.reshape(rvar, (-1, 1))
    #Nu = (N/(Tg+1)/2).astype(np.int32)
    #Nt = (N/2).astype(np.int32)

    # same large-scale fading
    xvar1 = tf.reshape(abs(theta[0,...]), (1,1,-1))
    loglam = tf.reshape(theta[1,...], (1,1,-1)) # (1-pa)/pa*(Tg+1)
    tex0 = thetaex[0]
    tex1 = thetaex[1]
    tex2 = thetaex[2]
    rvar = tf.reshape(rvar, (-1,1,1))
    beta = xvar1/(xvar1+rvar)
    r = tf.reshape(r,(-1,2*E,Nt))

    # computation of xhat
    # c11 = 1+xvar1/rvar
    # c12 = rvar/(xvar1+rvar)
    # q1 = (c11**E)*tf.exp(-tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    # q1_inv = ((c12)**E)*tf.exp(tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    # q1_inv1 = tf.reshape(q1_inv, (-1,1,Nu,Tg+1))
    # t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    # q2 = tf.reshape( (tf.reduce_sum(q1_inv1,axis=3,keepdims=True)*t1), (-1,1,Nt) ) + loglam
    # Q = q1*q2
    # Q_inv = tf.maximum(Q,(1e-10))
    # xhat1 = r/Q*beta
    c11 = (1+xvar1/rvar)**E
    q1e = tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar
    q1e = tf.maximum(q1e,float(-40))
    q1e = tf.minimum(q1e,float(40))
    q1  = tf.exp(q1e)
    q11 = tf.reshape(q1, (-1,1,Nu,Tg+1))
    t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    Q = tf.reshape( (tf.reduce_sum(q11, axis=3, keepdims=True)*t1),(-1,1,Nt) ) + c11*loglam
    q1Q = q1/Q
    #q1Q = tf.maximum(q1Q,1e-10)
    xhat1 = r*q1Q*beta
    xhat1 = tex0*xhat1 - tex1*r
    xhat = tf.reshape(xhat1, (-1,2*E*Nt))

    # computation of the derivate of the shrinkage function
    rt = tf.reshape(r, (-1,E,2,Nt))
    rt11 = tf.transpose(rt, (0,3,1,2))
    rt11_rc = tf.reshape(rt11[:,:,:,0], (-1,Nt,E,1))
    rt11_rr = tf.reshape(rt11_rc, (-1,Nt,1,E))
    rt11_ic = tf.reshape(rt11[:,:,:,1], (-1,Nt,E,1))
    rt11_ir = tf.reshape(rt11_ic, (-1,Nt,1,E))
    # rhr: derivative of the eta function (B in the paper)
    # B = [[rhr_r], [rhr_i]]
    rhr_r = rt11_rc*rt11_rr + rt11_ic*rt11_ir
    rhr_i = rt11_rc*rt11_ir - rt11_ic*rt11_rr
    rhr = tf.concat([rhr_r,rhr_i],2)
    rhr = rhr*tf.reshape(beta/rvar,(-1,1,1,1))
    teyec = tf.reshape(tf.concat([tf.eye(E),tf.zeros([E,E],dtype=tf.float32)],0), (1,1,-1,E))
    q1Q4 = tf.reshape(q1Q,(-1,Nt,1,1))
    dxdr1 = (teyec*q1Q4 + rhr*q1Q4 - rhr*(q1Q4**2))*tf.reshape(beta,(-1,1,1,1))
    dxdr2 = tex2*(tex0*dxdr1 - tex1*teyec)
    # dxdr1 = ( teyec/tf.reshape(Q,(-1,Nt,1,1))+tf.reshape((1/Q-1/(Q)**2)*beta/rvar,(-1,Nt,1,1))*rhr )*tf.reshape(beta,(-1,Nt,1,1))
    dxdr = tf.reduce_mean(dxdr2,axis=1,keepdims=False)


    # # #various large-scale fading
    # xvar1 = tf.reshape(tf.reshape(abs(theta[0,...]), (-1,1))*tf.ones([1,Tg+1],dtype=np.float32), (1,1,-1))
    # loglam = tf.reshape(tf.reshape(abs(theta[1,...]), (-1,1))*tf.ones([1,Tg+1],dtype=np.float32), (1,1,-1))
    # tex0 = thetaex[0]
    # tex1 = thetaex[1]
    # tex2 = thetaex[2]
    # rvar = tf.reshape(rvar,(-1,1,1))
    # beta = xvar1/(xvar1+rvar)
    # r = tf.reshape(r, (-1,2*E,Nt))
    #
    # # computation of xhat
    # # c11 = 1+xvar1/rvar
    # # c12 = rvar/(xvar1+rvar)
    # # q1 = (c11**E)*tf.exp(-tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    # # q1_inv = ((c12)**E)*tf.exp(tf.reduce_sum(r*r,axis=1,keepdims=True)*beta/rvar)
    # # q1_inv1 = tf.reshape(q1_inv, (-1,1,Nu,Tg+1))
    # # t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    # # q2 = tf.reshape( (tf.reduce_sum(q1_inv1,axis=3,keepdims=True)*t1), (-1,1,Nt) ) + loglam
    # # Q = q1*q2
    # # Q_inv = tf.maximum(Q,(1e-10))
    # # xhat1 = r/Q*beta
    # c11 = (1+xvar1/rvar)**E
    # q1e = tf.reduce_sum(r*r, axis=1, keepdims=True)*beta/rvar
    # q1e = tf.maximum(q1e, float(-40))
    # q1e = tf.minimum(q1e, float(40))
    # q1 = tf.exp(q1e)
    # q11 = tf.reshape(q1, (-1,1,Nu,Tg+1))
    # t1 = tf.ones([1,1,1,Tg+1], dtype=tf.float32)
    # Q = tf.reshape((tf.reduce_sum(q11,axis=3,keepdims=True)*t1), (-1,1,Nt)) + c11*loglam
    # q1Q = q1/Q
    # # q1Q = tf.maximum(q1Q,1e-10)
    # xhat1 = r*q1Q*beta
    # xhat1 = tex0*xhat1 - tex1*r
    # xhat = tf.reshape(xhat1, (-1,2*E*Nt))
    #
    # # computation of the derivate of the shrinkage function
    # rt = tf.reshape(r, (-1,E,2,Nt))
    # rt11 = tf.transpose(rt, (0,3,1,2))
    # rt11_rc = tf.reshape(rt11[:,:,:,0], (-1,Nt,E,1))
    # rt11_rr = tf.reshape(rt11_rc, (-1,Nt,1,E))
    # rt11_ic = tf.reshape(rt11[:,:,:,1], (-1,Nt,E,1))
    # rt11_ir = tf.reshape(rt11_ic, (-1,Nt,1,E))
    # # rhr: derivative of the eta function (B in the paper)
    # # B = [[rhr_r], [rhr_i]]
    # rhr_r = rt11_rc*rt11_rr + rt11_ic*rt11_ir
    # rhr_i = rt11_rc*rt11_ir - rt11_ic*rt11_rr
    # rhr = tf.concat([rhr_r,rhr_i],2)
    # rhr = rhr*tf.reshape(beta/rvar,(-1,Nt,1,1))
    # teyec = tf.reshape(tf.concat([tf.eye(E),tf.zeros([E,E],dtype=tf.float32)],0), (1,1,-1,E))
    # q1Q4 = tf.reshape(q1Q, (-1,Nt,1,1))
    # dxdr1 = (teyec*q1Q4 + rhr*q1Q4 - rhr*(q1Q4**2)) * tf.reshape(beta,(-1,Nt,1,1))
    # dxdr2 = tex2*(tex0*dxdr1 - tex1*teyec)
    # # dxdr1 = ( teyec/tf.reshape(Q,(-1,Nt,1,1))+tf.reshape((1/Q-1/(Q)**2)*beta/rvar,(-1,Nt,1,1))*rhr )*tf.reshape(beta,(-1,Nt,1,1))
    # dxdr = tf.reduce_mean(dxdr2, axis=1, keepdims=False)

    return (xhat, dxdr)

def shrink_piecwise_linear(r,rvar,theta):
    """Implement the piecewise linear shrinkage function.
        With minor modifications and variance normalization.
        theta[...,0] : abscissa of first vertex, scaled by sqrt(rvar)
        theta[...,1] : abscissa of second vertex, scaled by sqrt(rvar)
        theta[...,2] : slope from origin to first vertex
        theta[''',3] : slope from first vertex to second vertex
        theta[...,4] : slope after second vertex
    """
    ab0 = theta[...,0]
    ab1 = theta[...,1]
    sl0 = theta[...,2]
    sl1 = theta[...,3]
    sl2 = theta[...,4]

    # scale each column by sqrt(rvar)
    scale_out = tf.sqrt(rvar)
    scale_in = 1/scale_out
    rs = tf.sign(r*scale_in)
    ra = tf.abs(r*scale_in)

    # split the piecewise linear function into regions
    rgn0 = tf.to_float( ra<ab0)
    rgn1 = tf.to_float( ra<ab1) - rgn0
    rgn2 = tf.to_float( ra>=ab1)
    xhat = scale_out * rs*(
            rgn0*sl0*ra +
            rgn1*(sl1*(ra - ab0) + sl0*ab0 ) +
            rgn2*(sl2*(ra - ab1) +  sl0*ab0 + sl1*(ab1-ab0) )
            )
    dxdr =  sl0*rgn0 + sl1*rgn1 + sl2*rgn2
    dxdr = tf.reduce_mean(dxdr,0)
    return (xhat,dxdr)

def pwlin_grid(r_,rvar_,theta_,dtheta = .75):
    """piecewise linear with noise-adaptive grid spacing.
    returns xhat,dxdr
    where
        q = r/dtheta/sqrt(rvar)
        xhat = r * interp(q,theta)

    all but the  last dimensions of theta must broadcast to r_
    e.g. r.shape = (500,1000) is compatible with theta.shape=(500,1,7)
    """
    ntheta = int(theta_.get_shape()[-1])
    scale_ = dtheta / tf.sqrt(rvar_)
    ars_ = tf.clip_by_value( tf.expand_dims( tf.abs(r_)*scale_,-1),0.0, ntheta-1.0 )
    centers_ = tf.constant( np.arange(ntheta),dtype=tf.float32 )
    outer_distance_ = tf.maximum(0., 1.0-tf.abs(ars_ - centers_) ) # new dimension for distance to closest bin centers (or center)
    gain_ = tf.reduce_sum( theta_ * outer_distance_,axis=-1) # apply the gain (learnable)
    xhat_ = gain_ * r_
    dxdr_ = tf.gradients(xhat_,r_)[0]
    return (xhat_,dxdr_)

def shrink_expo(r,rvar,theta):
    """ Exponential shrinkage function
        xhat = r*(theta[1] + theta[2]*exp( - r^2/(2*theta[0]^2*rvar ) ) )
    """
    r2 = tf.square(r)
    den = -1/(2*tf.square(theta[0])*rvar)
    rho = tf.exp( r2 * den)
    xhat = r*( theta[1] + theta[2] * rho )
    return (xhat,auto_gradients(xhat,r) )

def shrink_spline(r,rvar,theta):
    """ Spline-based shrinkage function
    """
    scale = theta[0]*tf.sqrt(rvar)
    rs = tf.sign(r)
    ar = tf.abs(r/scale)
    ar2 = tf.square(ar)
    ar3 = ar*ar2
    reg1 = tf.to_float(ar<1)
    reg2 = tf.to_float(ar<2)-reg1
    ar_m2 = 2-ar
    ar_m2_p2 = tf.square(ar_m2)
    ar_m2_p3 = ar_m2*ar_m2_p2
    beta3 = ( (2./3 - ar2  + .5*ar3)*reg1 + (1./6*(ar_m2_p3))*reg2 )
    xhat = r*(theta[1] + theta[2]*beta3)
    return (xhat,auto_gradients(xhat,r))

def get_shrinkage_function(name):
	"retrieve a shrinkage function and some (probably awful) default parameter values"
	try:
		return {
			'soft':(shrink_soft_threshold,(1.3) ),
            'soft_V2':(shrink_soft_threshold_V2, (.5,)),
            'soft_MMV':(shrink_soft_threshold_MMV, (.5)),
			'bg':(shrink_bgest, (1.,math.log(1/.05-1.)) ),
            'bg_V2':(shrink_bgest_V2, ([[1.], [math.log(1/.05*4.0-1)]])),
			'pwlin':(shrink_piecwise_linear, (2,4,0.1,1.5,.95) ),
			'pwgrid':(pwlin_grid, np.linspace(.1,1,15).astype(np.float32)  ),
			'expo':(shrink_expo, (2.5,.9,-1) ),
			'spline':(shrink_spline, (3.7,.9,-1.5)),
            'bgSS':(shrink_bgSSest, (1., math.log(1/.05-1))),
            'stSS':(shrink_stSSest, (1.,1.)),
            'bgMMV': (shrink_bgest_MMV, np.array([[1.], [math.log(1/(.05/4.)-1)]])),
            'sbgMMV':(shrink_sbgest_MMV, np.array([[1.], [(1.0-0.05)/0.0125]])),
            'cstMMV':(shrink_csoft_threshold, np.array([1.0])),
            'cbgMMV':(shrink_cbg_MMV, np.array([[1.0],[math.log(1/0.025-1)]]))
		}[name]
	except KeyError as ke:
		raise ValueError('unrecognized shrink function %s' % name)
		sys.exit(1)

def tfcf(v):
    " return a tensorflow constant float version of v"
    return tf.constant(v,dtype=tf.float32)

def tfvar(v):
    " return a tensorflow variable float version of v"
    return tf.Variable(v,dtype=tf.float32)

def nmse(x1,x2):
    "return the normalized mean squared error between 2 numpy arrays"
    xdif=x1-x2
    return 2*(xdif*xdif).sum() / ( (x1*x1).sum() + (x2*x2).sum())

def test_func(shrink_func,theta,**kwargs):
    # repeat the same experiment
    tf.reset_default_graph()
    tf.set_random_seed(kwargs.get('seed',1) )

    N = kwargs.get('N',200)
    L = kwargs.get('L',400)
    tol = kwargs.get('tol',1e-6)
    step = kwargs.get('step',1e-4)
    shape = (N,L)
    xvar_ = tfcf(kwargs.get('xvar1',1))
    pnz_ = tfcf(kwargs.get('pnz',.1))
    rvar = np.ones(L)*kwargs.get('rvar',.1)
    rvar_ = tfcf(rvar)
    gx = tf.to_float(tf.random_uniform(shape ) < pnz_) * tf.random_normal(shape, stddev=tf.sqrt(xvar_), dtype=tf.float32)
    gr = gx + tf.random_normal(shape,stddev=tf.sqrt(rvar_), dtype=tf.float32)
    x_ = tf.placeholder(gx.dtype,gx.get_shape())
    r_ = tf.placeholder(gr.dtype,gr.get_shape())

    theta_ = tfvar(theta)

    xhat_,dxdr_ = shrink_func(r_,rvar_ ,theta_)
    loss = tf.nn.l2_loss(xhat_-x_)
    optimize_theta = tf.train.AdamOptimizer(step).minimize(loss,var_list=[theta_])

    # calculate an empirical gradient for comparison
    dr_ = tfcf(1e-4)
    dxdre_ = tf.reduce_mean( (shrink_func(r_+.5*dr_,rvar_ ,theta_)[0] - shrink_func(r_-.5*dr_,rvar_ ,theta_)[0]) / dr_ ,0)

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        (x,r) = sess.run((gx,gr))
        fd = {x_:x,r_:r}
        loss_prev = float('inf')
        for i in range(500):
            for j in range(50):
                sess.run(optimize_theta,fd)
            loss_cur,theta_cur = sess.run((loss,theta_),fd)
            #print 'loss=%s, theta=%s' % (str(loss_cur),str(theta_cur))
            if (1-loss_cur/loss_prev) < tol:
                break
            loss_prev = loss_cur
        xhat,dxdr,theta,dxdre = sess.run( (xhat_,dxdr_,theta_,dxdre_),fd)

    assert xhat.shape==(N,L)
    assert dxdr.shape==(L,) # MMV-specific -- we assume one average gradient per column
    assert nmse(dxdr,dxdre) < tol

    tf.reset_default_graph()
    estname = re.sub('.*shrink_([^ ]*).*','\\1', repr(shrink_func) )
    print('####   %s loss=%g \ttheta=%s' % (estname,loss_cur,repr(theta)))
    if False:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(r.reshape(-1),xhat.reshape(-1),'b.')
        plt.plot(r,xhat,'.')
        plt.show()
    return (x,r,xhat,rvar)


def show_shrinkage(shrink_func,theta,**kwargs):
    tf.reset_default_graph()
    tf.set_random_seed(kwargs.get('seed',1) )

    N = kwargs.get('N',500)
    L = kwargs.get('L',4)
    nsigmas = kwargs.get('sigmas',10)
    shape = (N,L)
    rvar = 1e-4
    r = np.reshape( np.linspace(0,nsigmas,N*L)*math.sqrt(rvar),shape)
    r_ = tfcf(r)
    rvar_ = tfcf(np.ones(L)*rvar)

    xhat_,dxdr_ = shrink_func(r_,rvar_ ,tfcf(theta))

    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
        xhat = sess.run(xhat_)
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(r.reshape(-1),r.reshape(-1),'y')
    plt.plot(r.reshape(-1),xhat.reshape(-1),'b')
    if kwargs.has_key('title'):
        plt.suptitle(kwargs['title'])
    plt.show()


if __name__ == "__main__":
    import sys
    import getopt
    usage = """
    -h : help
    -p file : load problem definition parameters from npz file
    -f function : use the named shrinkage function, one of {soft,bg,pwlin,expo,spline}
    """
    try:
        opts,args = getopt.getopt(sys.argv[1:] , 'hp:s:f:')
        opts = dict(opts)
    except getopt.GetoptError as e:
        opts={'-h':True}
    if opts.has_key('-h'):
        sys.stderr.write(usage)
        sys.exit()

    shrinkage_name = opts.get('-f','soft')
    f,theta = get_shrinkage_function( shrinkage_name )
    if opts.has_key('-s'):
        D=dict(np.load(opts['-s']).items())
        t=0
        while D.has_key('theta_%d'% t):
            theta_t = D['theta_%d' % t]
            show_shrinkage(f,theta_t,title='shrinkage=%s, theta_%d=%s' % (shrinkage_name,t, theta_t))
            t += 1
    else:
        show_shrinkage(f,theta)

    """
    test_func(shrink_bgest, (1,math.log(1/.1-1)) ,**parms)
    test_func(shrink_soft_threshold,(1.7,1.2) ,**parms)
    test_func(shrink_piecwise_linear, (2,4,0.1,1.5,.95) ,**parms)
    test_func(shrink_expo, (2.5,.9,-1) ,**parms)
    test_func(shrink_spline, (3.7,.9,-1.5) ,**parms)
    """
