#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math

import tensorflow as tf
import tools.shrinkage as shrinkage

def build_LISTA(prob,T,initial_lambda=.1,untied=False):
    """
    Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_)
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T):
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ )
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers

def build_LAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    #B = A.T / (1.01 * la.norm(A,2)**2)
    B = np.transpose(A)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('LAMP-{0} Linear T=1'.format(shrink),By_,(B_,)) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LAMP_V2(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    #B = A.T / (1.01 * la.norm(A,2)**2)
    B = A
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( prob.y_, B_ )
    layers.append( ('LAMP-{0} Linear T=1'.format(shrink),By_,(B_,)) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reshape(tf.reduce_sum(tf.square(prob.y_),1),(-1,1)) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( xhat_ , np.transpose(prob.A) ) + tf.reshape(bt_, (-1,1)) * vt_
        rvar_ = tf.reshape(tf.reduce_sum(tf.square(vt_),1),(-1,1)) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(vt_, B_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(vt_, B_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LAMP_E2E(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    B = A.T
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    #layers.append( ('LAMP-{0} Linear T=1'.format(shrink),By_,(B_,)) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    #layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            #layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        #layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    layers.append( ('LAMP-{0} T={1}'.format(shrink,T), xhat_, None))

    return layers

def build_LAMP_E2E_V2(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    #B = A.T / (1.01 * la.norm(A,2)**2)
    B = A
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( prob.y_, B_ )
    #layers.append( ('LAMP-{0} Linear T=1'.format(shrink),By_,(B_,)) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reshape(tf.reduce_sum(tf.square(prob.y_),1),(-1,1)) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} non-linear T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        bt_ = tf.reshape(dxdr_, (-1,1)) * NOverM
        vt_ = prob.y_ - tf.matmul( xhat_ , np.transpose(prob.A) ) + bt_ * vt_
        rvar_ = tf.reshape(tf.reduce_sum(tf.square(vt_),1),(-1,1)) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(vt_, B_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(vt_, B_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    #layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,T),xhat_,None ) )


    return layers

def build_LAMPSS(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """

    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    #p_max = 60.0 / 800.0
    #p_ini = 60.0 / 800.0
    p_max = 5
    p_ini = 0.5
    pt = p_ini * np.ones((T, 1), dtype=np.float32)

    for i in range(T):
        pt[i] = (i+1) * pt[i]
        if pt[i] > p_max:
            pt[i] = p_max


    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    percent = pt[0]
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ , percent )
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        percent = pt[t]
        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_, percent )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LAMPMMV(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    E = prob.E
    Tg = prob.Tg
    M,N = A.shape
    B = A
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,M))
    Y_Bt = tf.matmul( tf.reshape(Y_ini,(-1,M)), B_ )
    Y_Bt = tf.reshape(Y_Bt, (-1,E,N))
    Y_Bt1 = tf.reshape(Y_Bt, (-1,N*E))
    layers.append( ( 'LAMPMMV-{0} Linear'.format(shrink) , Y_Bt1 , (B_,) ) )

    if getattr(prob,'iid',True) == False:        # debug
        # set up individual parameters for every coordinate
        ##theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
        theta_init = np.ones((2,N),dtype=np.float32)
        theta_init[0,:] = prob.lsf_t
        theta_init[1,:] = 0.05/4*np.ones((N),dtype=np.float32)
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reshape(tf.reduce_sum(tf.square(prob.y_), 1), (-1,1)) * OneOverM * OneOverE
    (xhat_,dxdr_) = eta( Y_Bt , rvar_ , theta_ , E , N, Tg)
    layers.append( ('LAMPMMV-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    Vt_ = prob.y_
    Vt_ = tf.reshape(Vt_, (-1,E,M))
    for t in range(1,T):
        if len(dxdr_.get_shape())==4:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        Bt_Vt = tf.matmul(dxdr_, Vt_) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,N))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_, (-1,N)), tf.transpose(prob.A)), (-1,E,M)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,M*E))
        rvar_ = tf.reshape(tf.reduce_sum(tf.square(Vt_vec_),1), (-1,1)) * OneOverM * OneOverE
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1,N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMPMMV-{0} linear T={1}'.format(shrink,t+1),rhat1_ ,(B_,) ) )
        else:
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1, M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1, N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat2_, rvar_, theta_, E, N, Tg )
        layers.append( ('LAMPMMV-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LAMPMMV_E2E(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    E = prob.E
    M,N = A.shape
    B = A
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,M))
    Y_Bt = tf.matmul( tf.reshape(Y_ini,(-1,M)), B_ )
    Y_Bt = tf.reshape(Y_Bt, (-1,E,N))
    Y_Bt1 = tf.reshape(Y_Bt, (-1,N*E))
    #layers.append( ( 'Linear' , Y_Bt1 , None ) )

    if getattr(prob,'iid',True) == False:        # debug
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reshape(tf.reduce_sum(tf.square(prob.y_), 1), (-1,1)) * OneOverM * OneOverE
    (xhat_,dxdr_) = eta( Y_Bt , rvar_ , theta_ , E , N)
    #dxdr_1 = tf.reshape(dxdr_,(-1,1))
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    Vt_ = prob.y_
    Vt_ = tf.reshape(Vt_, (-1,E,M))
    for t in range(1,T):
        if len(dxdr_.get_shape())==4:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        Bt_Vt = tf.matmul(dxdr_, Vt_) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,N))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_, (-1,N)), tf.transpose(prob.A)), (-1,E,M)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,M*E))
        rvar_ = tf.reshape(tf.reduce_sum(tf.square(Vt_vec_),1), (-1,1)) * OneOverM * OneOverE
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1,N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)
            #layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat1_ ,(B_,) ) )
        else:
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1, M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1, N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat2_ ,rvar_ , theta_, E , N )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    #layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink, T), xhat_, None) )

    return layers

def build_LAMPMMV_H42(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    E = 2
    M,N = A.shape
    B = A
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,M))
    Y_Bt = tf.matmul( tf.reshape(Y_ini,(-1,M)), B_ )
    Y_Bt = tf.reshape(Y_Bt, (-1,E,N))
    Y_Bt1 = tf.reshape(Y_Bt, (-1,N*E))
    layers.append( ( 'LAMPMMV-{0} Linear'.format(shrink) , Y_Bt1 , None ) )

    if getattr(prob,'iid',True) == False:        # debug
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reshape(tf.reduce_sum(tf.square(prob.y_), 1), (-1,1)) * OneOverM * OneOverE
    (xhat_,dxdr_) = eta( Y_Bt , rvar_ , theta_ , E , N)
    layers.append( ('LAMPMMV-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    Vt_ = prob.y_
    Vt_ = tf.reshape(Vt_, (-1,E,M))
    for t in range(1,T):
        if len(dxdr_.get_shape())==4:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        Bt_Vt = tf.matmul(dxdr_, Vt_) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,N))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_, (-1,N)), tf.transpose(prob.A)), (-1,E,M)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,M*E))
        rvar_ = tf.reshape(tf.reduce_sum(tf.square(Vt_vec_),1), (-1,1)) * OneOverM * OneOverE
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1,N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMPMMV-{0} linear T={1}'.format(shrink,t+1),rhat1_ ,(B_,) ) )
        else:
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1, M)), B_), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1, N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat2_ ,rvar_ , theta_, E , N )
        layers.append( ('LAMPMMV-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LAMPMMV_D(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    E = 1
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,M))
    Y_Bt = tf.matmul( tf.reshape(Y_ini,(-1,M)), tf.transpose(B_) )
    Y_Bt = tf.reshape(Y_Bt, (-1,E,N))
    Y_Bt1 = tf.reshape(Y_Bt, (-1,N*E))
    layers.append( ( 'Linear' , Y_Bt1 , None ) )

    if getattr(prob,'iid',True) == False:        # debug
        # set up individual parameters for every coordinate
        theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_), 1) * OneOverM * OneOverE
    (xhat_,dxdr_) = eta( Y_Bt , rvar_ , theta_ , E , N)
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    Vt_ = prob.y_
    Vt_ = tf.reshape(Vt_, (-1,E,M))
    for t in range(1,T):
        if len(dxdr_.get_shape())==4:
            dxdr_ = tf.reduce_mean(dxdr_,axis=1)
        Bt_Vt = tf.matmul(dxdr_, Vt_) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,N))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_, (-1,N)), tf.transpose(prob.A)), (-1,E,M)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,M*E))
        rvar_ = tf.reduce_sum(tf.square(Vt_vec_),1) * OneOverM * OneOverE
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,M)), tf.transpose(B_)), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1,N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat1_ ,(B_,) ) )
        else:
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_, (-1, M)), tf.transpose(B_)), (-1,E,N))
            rhat1_ = tf.reshape(rhat2_, (-1, N*E))
            #rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat2_ ,rvar_ , theta_, E , N )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_CLAMPMMV_C(prob,T,shrink,untied):
    # complex LAMP network for the MMV problem
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    layers = []
    A = prob.A
    E = prob.E
    Tg = prob.Tg
    # M, N = A.shape
    M = prob.M
    N = prob.N
    Nt = N*(Tg+1)
    Mt = M+Tg
    B = A
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    if getattr(prob, 'iid', True) == False:  # debug
        # set up individual parameters for every coordinate
        ##theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
        # bg-est
        theta_init1 = np.ones((2,N), dtype=np.float32)
        theta_init1[0,:] = prob.lsf**2
        theta_init1[1,:] = (1-prob.pnz)/prob.pnz*(Tg+1)*np.ones((N),dtype=np.float32)
        # theta_init1[1,:] = math.log(1/prob.pnz*(Tg+1)-1)*np.ones((N),dtype=np.float32)
        # st-est
        # theta_init1 = 1.3*np.ones((1,N), dtype=np.float32)
    else:
        theta_init1 = np.reshape(np.ones((2),dtype=np.float32),(2,1))
        theta_init1[0] = 1.0
        theta_init1[1] = (1-prob.pnz)/prob.pnz*(Tg+1)
        # theta_init1[1] = math.log(1/prob.pnz*(Tg+1)-1)
        # st-est
        # theta_init1 = 1.3*np.ones((1), dtype=np.float32)
    thetaex = np.zeros([3]).astype(np.float32)
    thetaex[0] = 1.0
    thetaex[1] = 0.0
    thetaex[2] = 1.0
    # print('theta_init=' + repr(theta_init1))
    theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_0')
    thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_0')
    thetaex1 = np.ones((1), dtype=np.float32)
    thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_0')

    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,2*Mt))
    # Y_ini = prob.y_
    # y_ini = tf.reshape(Y_ini, (-1,2*Mt*E))
    Y_Bt = thetaex1_*tf.matmul(tf.reshape(Y_ini, (-1,2*Mt)), B_)
    # Y_Bt = tf.matmul(tf.reshape(Y_ini, (-1, 2*Mt)), B)
    Y_Bt = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # Y_Bt1 = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, (B_,)))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, ()))
    T_list = [theta_, thetaex_, thetaex1_]
    T_tuple = tuple(T_list)

    OneOverMt = tf.constant(float(1)/Mt, dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E, dtype=tf.float32)
    NOverM = tf.constant(float(Nt)/Mt, dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(y_ini),axis=1,keepdims=True) * OneOverMt*OneOverE
    (xhat_, dxdr_) = eta(Y_Bt, rvar_, theta_, thetaex_, Nt, N, E, Tg)
    # layers.append(('LAMPMMV-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,)))
    layers.append(('LAMPMMV-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,thetaex_,),T_tuple))

    Vt_ = Y_ini
    # Vt_ = tf.reshape(Vt_, (-1,E,2*Mt))
    for t in range(1, T):
        if len(dxdr_.get_shape()) == 4:
            dxdr_ = tf.reduce_mean(dxdr_, axis=1)
        Vt_r = Vt_[:,:,0:Mt]
        Vt_i = Vt_[:,:,Mt:(2*Mt)]
        # Vt_0 = [Vt_r,  Vt_i
        #         -Vt_i, Vt_r]
        Vt_1 = tf.concat([Vt_r,Vt_i],2)
        Vt_2 = tf.concat([-Vt_i,Vt_r],2)
        Vt_0 = tf.concat([Vt_1,Vt_2],1)
        Bt_Vt = tf.matmul(tf.transpose(dxdr_,(0,2,1)), Vt_0) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,2*Nt))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_,(-1,2*Nt)), tf.transpose(prob.A)), (-1,E,2*Mt)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,2*Mt*E))
        rvar_ = tf.reduce_sum(tf.square(Vt_vec_),1,keepdims=True) * OneOverMt*OneOverE
        theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_' + str(t))
        thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_' + str(t))
        thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_' + str(t))
        T_list.append(theta_)
        T_list.append(thetaex_)
        T_list.append(thetaex1_)
        T_tuple = tuple(T_list)
        if untied:
            B_ = tf.Variable(B, dtype=tf.float32, name='B_' + str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            rhat1_ = tf.reshape(rhat2_, (-1, 2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append(('LAMPMMV-{0}, linear T={1}'.format(shrink, t+1), rhat1_, (B_,)))
        else:
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,2*Mt)), B), (-1,E,2*Nt))
            rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))

            # rhat1_ = tf.reshape(rhat2_, (-1,2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_, dxdr_) = eta(rhat2_, rvar_, theta_, thetaex_, Nt, N, E, Tg)
        # layers.append(('LAMPMMV-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,)))
        layers.append(('LAMPMMV-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,thetaex_,),T_tuple))

    # layers.append(('LAMPMMV-{0} non-linear T={1}'.format(shrink, T), xhat_, (theta_,)))

    return layers

def build_CLAMPMMV_D(prob,T,shrink,untied):
    # complex LAMP network for the MMV problem
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    layers = []
    A = prob.A
    # E = prob.E
    E = 1
    Tg = prob.Tg
    # M, N = A.shape
    M = prob.M
    N = prob.N
    Nt = N*(Tg+1)
    Mt = M+Tg
    B = A
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    if getattr(prob, 'iid', True) == False:  # debug
        # set up individual parameters for every coordinate
        ##theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
        # bg-est
        theta_init1 = np.ones((2,N), dtype=np.float32)
        theta_init1[0,:] = prob.lsf
        theta_init1[1,:] = (1-prob.pnz)/prob.pnz*(Tg+1)*np.ones((N),dtype=np.float32)
        # st-est
        # theta_init1 = 1.3*np.ones((1,N), dtype=np.float32)
    else:
        theta_init1 = np.reshape(np.ones((2),dtype=np.float32),(2,1))
        theta_init1[0] = 1.0
        theta_init1[1] = (1-prob.pnz)/prob.pnz*(Tg+1)
        # st-est
        # theta_init1 = 1.3*np.ones((1), dtype=np.float32)
    thetaex = np.zeros([3]).astype(np.float32)
    thetaex[0] = 1.0
    thetaex[1] = 0.0
    thetaex[2] = 1.0
    thetaex1 = np.ones([1],dtype=np.float32)
    # print('theta_init=' + repr(theta_init1))
    theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_0')
    thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_0')
    thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_0')

    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,2*Mt))
    # Y_ini = prob.y_
    # y_ini = tf.reshape(Y_ini, (-1,2*Mt*E))
    # Y_Bt = tf.matmul(tf.reshape(Y_ini, (-1,2*Mt)), B_)
    Y_Bt = thetaex1_* tf.matmul(tf.reshape(Y_ini, (-1,2*Mt)), B_)
    # Y_Bt = thetaex1_*tf.matmul(tf.reshape(Y_ini, (-1, 2*Mt)), B)
    Y_Bt = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # Y_Bt1 = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, (B_,)))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, ()))
    T_list = [theta_, thetaex_, thetaex1_]
    T_tuple = tuple(T_list)

    OneOverMt = tf.constant(float(1)/Mt, dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E, dtype=tf.float32)
    NOverM = tf.constant(float(Nt)/Mt, dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(y_ini),axis=1,keepdims=True) * OneOverMt*OneOverE
    (xhat_, dxdr_) = eta(Y_Bt, rvar_, theta_, thetaex_, Nt, N, E, Tg)
    # layers.append(('LAMPMMV-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,)))
    layers.append(('LAMPMMVD-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,thetaex_,),T_tuple))

    Vt_ = Y_ini
    # Vt_ = tf.reshape(Vt_, (-1,E,2*Mt))
    for t in range(1, T):
        if len(dxdr_.get_shape()) == 4:
            dxdr_ = tf.reduce_mean(dxdr_, axis=1)
        Vt_r = Vt_[:,:,0:Mt]
        Vt_i = Vt_[:,:,Mt:(2*Mt)]
        # Vt_0 = [Vt_r,  Vt_i
        #         -Vt_i, Vt_r]
        Vt_1 = tf.concat([Vt_r,Vt_i],2)
        Vt_2 = tf.concat([-Vt_i,Vt_r],2)
        Vt_0 = tf.concat([Vt_1,Vt_2],1)
        Bt_Vt = tf.matmul(tf.transpose(dxdr_,(0,2,1)), Vt_0) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,2*Nt))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_,(-1,2*Nt)), tf.transpose(prob.A)), (-1,E,2*Mt)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,2*Mt*E))
        rvar_ = tf.reduce_sum(tf.square(Vt_vec_),1,keepdims=True) * OneOverMt*OneOverE
        theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_' + str(t))
        thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_' + str(t))
        thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_' + str(t))
        T_list.append(theta_)
        T_list.append(thetaex_)
        T_list.append(thetaex1_)
        T_tuple = tuple(T_list)
        if untied:
            B_ = tf.Variable(B, dtype=tf.float32, name='B_' + str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            rhat1_ = tf.reshape(rhat2_, (-1, 2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append(('LAMPMMVD-{0}, linear T={1}'.format(shrink, t+1), rhat1_, (B_,)))
        else:
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,2*Mt)), B), (-1,E,2*Nt))
            rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            # rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))

            # rhat1_ = tf.reshape(rhat2_, (-1,2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_, dxdr_) = eta(rhat2_, rvar_, theta_, thetaex_, Nt, N, E, Tg)
        # layers.append(('LAMPMMV-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,)))
        layers.append(('LAMPMMVD-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,thetaex_,),T_tuple))

    # layers.append(('LAMPMMV-{0} non-linear T={1}'.format(shrink, T), xhat_, (theta_,)))

    return layers

def build_CLAMPMMV_H(prob,T,shrink,untied):
    # complex LAMP network for the MMV problem
    eta, theta_init = shrinkage.get_shrinkage_function(shrink)
    layers = []
    A = prob.A
    # E = prob.E
    E = 2
    Tg = prob.Tg
    # M, N = A.shape
    M = prob.M
    N = prob.N
    Nt = N*(Tg+1)
    Mt = M+Tg
    B = A
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    if getattr(prob, 'iid', True) == False:  # debug
        # set up individual parameters for every coordinate
        ##theta_init = theta_init*np.ones( (1,N),dtype=np.float32 )
        # bg-est
        theta_init1 = np.ones((2,N), dtype=np.float32)
        theta_init1[0,:] = prob.lsf
        theta_init1[1,:] = (1-prob.pnz)/prob.pnz*(Tg+1)*np.ones((N),dtype=np.float32)
        # st-est
        # theta_init1 = 1.3*np.ones((1,N), dtype=np.float32)
    else:
        theta_init1 = np.reshape(np.ones((2),dtype=np.float32),(2,1))
        theta_init1[0] = 1.0
        theta_init1[1] = (1-prob.pnz)/prob.pnz*(Tg+1)
        # st-est
        # theta_init1 = 1.3*np.ones((1), dtype=np.float32)
    thetaex = np.zeros([3]).astype(np.float32)
    thetaex[0] = 1.0
    thetaex[1] = 0.0
    thetaex[2] = 1.0
    thetaex1 = np.ones([1],dtype=np.float32)
    # print('theta_init=' + repr(theta_init1))
    theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_0')
    thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_0')
    # thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_0')

    y_ini = prob.y_
    Y_ini = tf.reshape(y_ini, (-1,E,2*Mt))
    # Y_ini = prob.y_
    # y_ini = tf.reshape(Y_ini, (-1,2*Mt*E))
    Y_Bt = tf.matmul(tf.reshape(Y_ini, (-1,2*Mt)), B_)
    # Y_Bt = thetaex1_*tf.matmul(tf.reshape(Y_ini, (-1, 2*Mt)), B)
    Y_Bt = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # Y_Bt1 = tf.reshape(Y_Bt, (-1,E,2*Nt))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, (B_,)))
    # layers.append(('CLAMPMMV-{0}, T=1, Linear'.format(shrink), Y_Bt, ()))
    T_list = [theta_, thetaex_]
    T_tuple = tuple(T_list)

    OneOverMt = tf.constant(float(1)/Mt, dtype=tf.float32)
    OneOverE = tf.constant(float(1)/E, dtype=tf.float32)
    NOverM = tf.constant(float(Nt)/Mt, dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(y_ini),axis=1,keepdims=True) * OneOverMt*OneOverE
    (xhat_, dxdr_) = eta(Y_Bt, rvar_, theta_, thetaex_, Nt, N, E, Tg)
    # layers.append(('LAMPMMV-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,)))
    layers.append(('LAMPMMVH-{0}, non-linear T=1'.format(shrink), xhat_, (theta_,thetaex_,),T_tuple))

    Vt_ = Y_ini
    # Vt_ = tf.reshape(Vt_, (-1,E,2*Mt))
    for t in range(1, T):
        if len(dxdr_.get_shape()) == 4:
            dxdr_ = tf.reduce_mean(dxdr_, axis=1)
        Vt_r = Vt_[:,:,0:Mt]
        Vt_i = Vt_[:,:,Mt:(2*Mt)]
        # Vt_0 = [Vt_r,  Vt_i
        #         -Vt_i, Vt_r]
        Vt_1 = tf.concat([Vt_r,Vt_i],2)
        Vt_2 = tf.concat([-Vt_i,Vt_r],2)
        Vt_0 = tf.concat([Vt_1,Vt_2],1)
        Bt_Vt = tf.matmul(tf.transpose(dxdr_,(0,2,1)), Vt_0) * NOverM
        xhat1_ = tf.reshape(xhat_, (-1,E,2*Nt))
        Vt_ = Y_ini - tf.reshape(tf.matmul(tf.reshape(xhat1_,(-1,2*Nt)), tf.transpose(prob.A)), (-1,E,2*Mt)) + Bt_Vt
        Vt_vec_ = tf.reshape(Vt_, (-1,2*Mt*E))
        rvar_ = tf.reduce_sum(tf.square(Vt_vec_),1,keepdims=True) * OneOverMt*OneOverE
        theta_ = tf.Variable(theta_init1, dtype=tf.float32, name='theta_' + str(t))
        thetaex_ = tf.Variable(thetaex, dtype=tf.float32, name='thetaex_' + str(t))
        # thetaex1_ = tf.Variable(thetaex1, dtype=tf.float32, name='thetaex1_' + str(t))
        T_list.append(theta_)
        T_list.append(thetaex_)
        # T_list.append(thetaex1_)
        T_tuple = tuple(T_list)
        if untied:
            B_ = tf.Variable(B, dtype=tf.float32, name='B_' + str(t))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            rhat1_ = tf.reshape(rhat2_, (-1, 2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append(('LAMPMMVH-{0}, linear T={1}'.format(shrink, t+1), rhat1_, (B_,)))
        else:
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_, (-1,2*Mt)), B), (-1,E,2*Nt))
            rhat2_ = xhat1_ + tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))
            # rhat2_ = xhat1_ + thetaex1_*tf.reshape(tf.matmul(tf.reshape(Vt_,(-1,2*Mt)), B_), (-1,E,2*Nt))

            # rhat1_ = tf.reshape(rhat2_, (-1,2*Nt*E))
            # rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_, dxdr_) = eta(rhat2_, rvar_, theta_, thetaex_, Nt, N, E, Tg)
        # layers.append(('LAMPMMV-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,)))
        layers.append(('LAMPMMVH-{0}, non-linear T={1}'.format(shrink,t+1), xhat_, (theta_,thetaex_,),T_tuple))

    # layers.append(('LAMPMMV-{0} non-linear T={1}'.format(shrink, T), xhat_, (theta_,)))

    return layers

def build_LMAMP(prob,T,shrink,untied):
    """
    Builds a LAMP network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = tf.matmul( B_ , prob.y_ )
    layers.append( ('Linear',By_,None) )

    if getattr(prob,'iid',True) == False:
        # set up individual parameters for every coordinate
        theta_init = theta_init*0.5*np.ones( (N,1),dtype=np.float32 )
    theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_0')
    OneOverM = tf.constant(float(1)/M,dtype=tf.float32)
    NOverM = tf.constant(float(N)/M,dtype=tf.float32)
    rvar_ = tf.reduce_sum(tf.square(prob.y_),0) * OneOverM
    (xhat_,dxdr_) = eta( By_,rvar_ , theta_ )
    layers.append( ('LAMP-{0} T=1'.format(shrink),xhat_,(theta_,) ) )

    vt_ = prob.y_
    for t in range(1,T):
        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        bt_ = dxdr_ * NOverM
        vt_ = prob.y_ - tf.matmul( prob.A_ , xhat_ ) + bt_ * vt_
        rvar_ = tf.reduce_sum(tf.square(vt_),0) * OneOverM
        theta_ = tf.Variable(theta_init,name='theta_'+str(t))
        if untied:
            B_ =  tf.Variable(B,dtype=tf.float32,name='B_'+str(t))
            rhat_ = xhat_ + tf.matmul(B_,vt_)
            layers.append( ('LAMP-{0} linear T={1}'.format(shrink,t+1),rhat_ ,(B_,) ) )
        else:
            rhat_ = xhat_ + tf.matmul(B_,vt_)

        (xhat_,dxdr_) = eta( rhat_ ,rvar_ , theta_ )
        layers.append( ('LAMP-{0} non-linear T={1}'.format(shrink,t+1),xhat_,(theta_,) ) )

    return layers

def build_LVAMP(prob,T,shrink):
    """
    Build the LVMAP network with an SVD parameterization.
    Learns the measurement noise variance and nonlinearity parameters
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    print('theta_init='+repr(theta_init))
    layers=[]
    A = prob.A
    M,N = A.shape
    AA = np.matmul(A,A.T)
    s2,U = la.eigh(AA)  # this is faster than svd, but less precise if ill-conditioned
    s = np.sqrt(s2)
    V = np.matmul( A.T,U) / s
    print('svd reconstruction error={nmse:.3f}dB'.format(nmse=20*np.log10(la.norm(A-np.matmul(U*s,V.T))/la.norm(A) ) ) )
    assert np.allclose( A, np.matmul(U*s,V.T),rtol=1e-4,atol=1e-4)
    V_ = tf.constant(V,dtype=tf.float32,name='V')

    # precompute some tensorflow constants
    rS2_ = tf.constant( np.reshape( 1/(s*s),(-1,1) ).astype(np.float32) )  # reshape to (M,1) to allow broadcasting
    #rj_ = tf.zeros( (N,L) ,dtype=tf.float32)
    rj_ = tf.zeros_like( prob.x_)
    taurj_ =  tf.reduce_sum(prob.y_*prob.y_,0)/(N)
    logyvar_ = tf.Variable( 0.0,name='logyvar',dtype=tf.float32)
    yvar_ = tf.exp( logyvar_)
    ytilde_ = tf.matmul( tf.constant( ((U/s).T).astype(np.float32) ) ,prob.y_)  # inv(S)*U*y
    Vt_ = tf.transpose(V_)

    xhat_ = tf.constant(0,dtype=tf.float32)
    for t in range(T):  # layers 0 thru T-1
        # linear step (LMMSE estimation and Onsager correction)
        varRat_ = tf.reshape(yvar_/taurj_,(1,-1) ) # one per column
        scale_each_ = 1/( 1 + rS2_*varRat_ ) # LMMSE scaling individualized per element {singular dimension,column}
        zetai_ = N/tf.reduce_sum(scale_each_,0) # one per column  (zetai_ is 1/(1-alphai) from Phil's derivation )
        adjust_ = ( scale_each_*(ytilde_ - tf.matmul(Vt_,rj_))) * zetai_ #  adjustment in the s space
        ri_ = rj_ + tf.matmul(V_, adjust_ )  # bring the adjustment back into the x space and apply it
        tauri_ = taurj_*(zetai_-1) # adjust the variance

        # non-linear step
        theta_ = tf.Variable(theta_init,dtype=tf.float32,name='theta_'+str(t))
        xhat_,dxdr_ = eta(ri_,tauri_,theta_)
        if t==0:
            learnvars = None # really means "all"
        else:
            learnvars=(theta_,)
        layers.append( ('LVAMP-{0} T={1}'.format(shrink,t+1),xhat_, learnvars ) )

        if len(dxdr_.get_shape())==2:
            dxdr_ = tf.reduce_mean(dxdr_,axis=0)
        zetaj_ = 1/(1-dxdr_)
        rj_ = (xhat_ - dxdr_*ri_)*zetaj_ # apply Onsager correction
        taurj_ = tauri_*(zetaj_-1) # adjust the variance

    return layers

def build_LVAMP_dense(prob,T,shrink,iid=False):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    layers=[]
    A = prob.A
    M,N = A.shape

    Hinit = np.matmul(prob.xinit,la.pinv(prob.yinit) )
    H_ = tf.Variable(Hinit,dtype=tf.float32,name='H0')
    xhat_lin_ = tf.matmul(H_,prob.y_)
    layers.append( ('Linear',xhat_lin_,None) )

    if shrink=='pwgrid':
        theta_init = np.linspace(.01,.99,15).astype(np.float32)
    vs_def = np.array(1,dtype=np.float32)
    if not iid:
        theta_init = np.tile( theta_init ,(N,1,1))
        vs_def = np.tile( vs_def ,(N,1))

    theta_ = tf.Variable(theta_init,name='theta0',dtype=tf.float32)
    vs_ = tf.Variable(vs_def,name='vs0',dtype=tf.float32)
    rhat_nl_ = xhat_lin_
    rvar_nl_ = vs_ * tf.reduce_sum(prob.y_*prob.y_,0)/N

    xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_nl_, None ) )
    for t in range(1,T):
        alpha_nl_ = tf.reduce_mean( alpha_nl_,axis=0) # each col average dxdr

        gain_nl_ = 1.0 /(1.0 - alpha_nl_)
        rhat_lin_ = gain_nl_ * (xhat_nl_ - alpha_nl_ * rhat_nl_)
        rvar_lin_ = rvar_nl_ * alpha_nl_ * gain_nl_

        H_ = tf.Variable(Hinit,dtype=tf.float32,name='H'+str(t))
        G_ = tf.Variable(.9*np.identity(N),dtype=tf.float32,name='G'+str(t))
        xhat_lin_ = tf.matmul(H_,prob.y_) + tf.matmul(G_,rhat_lin_)

        layers.append( ('LVAMP-{0} lin T={1}'.format(shrink,1+t),xhat_lin_, (H_,G_) ) )

        alpha_lin_ = tf.expand_dims(tf.diag_part(G_),1)

        eps = .5/N
        alpha_lin_ = tf.maximum(eps,tf.minimum(1-eps, alpha_lin_ ) )

        vs_ = tf.Variable(vs_def,name='vs'+str(t),dtype=tf.float32)

        gain_lin_ = vs_ * 1.0/(1.0 - alpha_lin_)
        rhat_nl_ = gain_lin_ * (xhat_lin_ - alpha_lin_ * rhat_lin_)
        rvar_nl_ = rvar_lin_ * alpha_lin_ * gain_lin_

        theta_ = tf.Variable(theta_init,name='theta'+str(t),dtype=tf.float32)

        xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
        alpha_nl_ = tf.maximum(eps,tf.minimum(1-eps, alpha_nl_ ) )
        layers.append( ('LVAMP-{0}  nl T={1}'.format(shrink,1+t),xhat_nl_, (vs_,theta_,) ) )

    return layers
