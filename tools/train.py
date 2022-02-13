#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import sys
import tensorflow as tf
from scipy.io import savemat, loadmat
import scipy.io as scio

def save_trainable_vars(sess,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in tf.trainable_variables():
        save[str(v.name)] = sess.run(v)
    save.update(kwargs)
    np.savez(filename,**save)

def load_trainable_vars(sess,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in tf.trainable_variables() ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                sess.run(tf.assign( tv[k], d) )
            else:
                other[k] = d
    except IOError:
        pass
    return other

def setup_training(layer_info,prob, trinit=1e-3,refinements=(.5,.1,.01),final_refine=None ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of training operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    maskX_ = getattr(prob,'maskX_',1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ *maskX_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    training_stages=[]
    #print(layer_info)
    for name,xhat_,var_list,var_list2 in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        nmse_  = tf.nn.l2_loss( (xhat_ - prob.x_)*maskX_) / nmse_denom_
        #x_est_ = prob.x_
        # mse_ = tf.nn.l2_loss(xhat_ - prob.x_)*2
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            #training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
            training_stages.append((name+' trainrate=1', xhat_, loss_, nmse_, train_, var_list))
            # train_ = tf.train.AdamOptimizer(tr_).minimize(loss_)
            # training_stages.append((name + ' trainrate=1', xhat_, loss_, nmse_, train_, ()))
            print(name+' trainrate=1 training stages is appended.')
        for fm in refinements:
            if fm != 1:
                train2_ = tf.train.AdamOptimizer(tr_*fm).minimize(loss_, var_list=var_list2)
                training_stages.append( (name+' refine trainrate = '+str(fm),xhat_,loss_,nmse_,train2_,var_list2) )
                print(name + ' refine trainrate = '+str(fm)+' training stages is appended.')
    if final_refine:
        train2_ = tf.train.AdamOptimizer(tr_*final_refine*0.003).minimize(loss_)
        training_stages.append( (name+' final refine '+str(final_refine),xhat_,loss_,nmse_,train2_,()) )
        print(name + ' final refine training stages is appended.')

    return training_stages

def setup_testing( layer_info,prob,trinit=1e-3 ):
    """ Given a list of layer info (name,xhat_,newvars),
    create an output list of testing operations (name,xhat_,loss_,nmse_,trainop_ ).
    Each layer_info element will be split into one or more output training operations
    based on the presence of newvars and len(refinements)
    """
    losses_=[]
    nmse_=[]
    trainers_=[]
    #assert np.array(refinements).min()>0,'all refinements must be in (0,1]'
    #assert np.array(refinements).max()<=1,'all refinements must be in (0,1]'

    maskX_ = getattr(prob,'maskX_',1)
    if maskX_ != 1:
        print('masking out inconsequential parts of signal x for nmse reporting')

    nmse_denom_ = tf.nn.l2_loss(prob.x_ *maskX_)

    tr_ = tf.Variable(trinit,name='tr',trainable=False)
    testing_stages=[]
    #print(layer_info)
    for name,xhat_,var_list,var_list2 in layer_info:
        loss_  = tf.nn.l2_loss( xhat_ - prob.x_)
        nmse_  = tf.nn.l2_loss( (xhat_ - prob.x_)*maskX_) / nmse_denom_
        nmse_1 = tf.reduce_mean(tf.reduce_sum((xhat_-prob.x_)**2,1)/tf.reduce_sum((prob.x_)**2,1), 0, keepdims=False)
        #x_est_ = prob.x_
        if var_list is not None:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_, var_list=var_list)
            # train_ = []
            #training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
            testing_stages.append((name+' refine trainrate=1', xhat_, loss_, nmse_, nmse_1, train_, var_list))
            print(name + ' trainrate=1 training stages is appended.')
        else:
            train_ = tf.train.AdamOptimizer(tr_).minimize(loss_)
            # train_ = []
            # training_stages.append( (name,xhat_,loss_,nmse_,train_,var_list) )
            testing_stages.append((name + ' refine trainrate=1', xhat_, loss_, nmse_, nmse_1, train_, ()))

    return testing_stages


def do_training_CMMVC(training_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T=LT

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train']
    x_train = DT['x_train']
    y_test = DT['y_test']
    x_test = DT['x_test']
    y_test_ex = DT['y_test_ex']
    x_test_ex = DT['x_test_ex']
    y_test_iw2 = y_test_ex[0:5000,:]
    x_test_iw2 = x_test_ex[0:5000,:]
    y_test_iw4 = y_test_ex[5000:10000, :]
    x_test_iw4 = x_test_ex[5000:10000, :]
    y_test_iw6 = y_test_ex[10000:15000, :]
    x_test_iw6 = x_test_ex[10000:15000, :]
    y_test_iw8 = y_test_ex[15000:20000, :]
    x_test_iw8 = x_test_ex[15000:20000, :]


    batch_size = 100

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            tbi = i % 1000      # debug
            tbi_f = tbi*batch_size
            tbi_l = (tbi+1)*batch_size
            y_train_batch = y_train[tbi_f:tbi_l,:]
            x_train_batch = x_train[tbi_f:tbi_l,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_test, prob.x_:x_test})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*Nt))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*Nt*E])
    for i in range(4):
        i_f = i*5000
        i_l = (i+1)*5000
        x_est_ex[i_f:i_l,:] = sess.run(xhat_,feed_dict={prob.y_:y_test_ex[i_f:i_l,:],prob.x_:x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_est_ex,(-1,E,2*Nt))
    x_est_ex = np.transpose(x_est_ex,(2,1,0))
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')

    return sess

def do_testing_CMMVC(testing_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    # state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    # done = state.get('done', [])
    # log = str(state.get('log', ''))
    # saver = tf.train.Saver()
    # saver.restore(sess, savemodel)

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T = LT

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train']
    x_train = DT['x_train']
    y_test = DT['y_test']
    x_test = DT['x_test']
    y_test_ex = DT['y_test_ex']
    x_test_ex = DT['x_test_ex']
    y_test_iw2 = y_test_ex[0:5000,:]
    x_test_iw2 = x_test_ex[0:5000,:]
    y_test_iw4 = y_test_ex[5000:10000, :]
    x_test_iw4 = x_test_ex[5000:10000, :]
    y_test_iw6 = y_test_ex[10000:15000, :]
    x_test_iw6 = x_test_ex[10000:15000, :]
    y_test_iw8 = y_test_ex[15000:20000, :]
    x_test_iw8 = x_test_ex[15000:20000, :]


    batch_size = 200

    nmse1 = np.zeros([5000,T])
    nmse  = np.zeros([1,T])
    nmse0_dB = np.zeros([T], dtype=np.float32)
    i = -1
    for name, xhat_, loss_, nmse_, nmse_1, train_, var_list in testing_stages:
        i = i+1
        # nmse1 = np.zeros([5000])
        nmse0 = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
        # nmse0 = sess.run(nmse_, feed_dict={prob.y_: y_test_ex[15000:20000,:], prob.x_: x_test_ex[15000:20000,:]})
        nmse0_dB[i] = 10*np.log10(nmse0)
        print('In iteration '+str(i+1)+' , nmse = {nmse:.6f}dB.'.format(nmse=nmse0_dB[i]))
        x_t = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
        nmse1[:,i] = 10*np.log10(np.sum((x_t-x_test)**2, axis=1)/np.sum(x_test**2, axis=1))
        nmse[0, i] = np.mean(nmse1[:, i])
        print('In iteration '+str(i+1)+' , nmse(average samples) = {nmse:.6f}dB.'.format(nmse=nmse[0,i]))

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*N*(Tg+1)))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*N*(Tg+1)*E])
    for i in range(4):
        i_f = i*5000
        i_l = (i+1)*5000
        x_est_ex[i_f:i_l, :] = sess.run(xhat_, feed_dict={prob.y_: y_test_ex[i_f:i_l,:],
                                                          prob.x_: x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_est_ex, (-1,E,2*N*(Tg+1)))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    D = dict(x_AMP=x_est, x_AMP_ex=x_est_ex)
    # D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')


    return sess

def do_training_CMMVD(training_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T=LT

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train_D']
    x_train = DT['x_train_D']
    y_test = DT['y_test_D']
    x_test = DT['x_test_D']
    y_test_ex = DT['y_test_ex_D']
    x_test_ex = DT['x_test_ex_D']
    y_test_iw2 = y_test_ex[0:5000*E,:]
    x_test_iw2 = x_test_ex[0:5000*E,:]
    y_test_iw4 = y_test_ex[5000*E:10000*E, :]
    x_test_iw4 = x_test_ex[5000*E:10000*E, :]
    y_test_iw6 = y_test_ex[10000*E:15000*E, :]
    x_test_iw6 = x_test_ex[10000*E:15000*E, :]
    y_test_iw8 = y_test_ex[15000*E:20000*E, :]
    x_test_iw8 = x_test_ex[15000*E:20000*E, :]


    batch_size = 100

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            tbi = i % (1000*E)      # debug
            tbi_f = tbi*batch_size
            tbi_l = (tbi+1)*batch_size
            y_train_batch = y_train[tbi_f:tbi_l,:]
            x_train_batch = x_train[tbi_f:tbi_l,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_test, prob.x_:x_test})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along

            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*Nt))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*Nt*E])
    x_temp = np.zeros([20000*E,Nt*2],dtype=np.float32)
    for i in range(4):
        i_f = i*5000*E
        i_l = (i+1)*5000*E
        x_temp[i_f:i_l,:] = sess.run(xhat_,feed_dict={prob.y_:y_test_ex[i_f:i_l,:],prob.x_:x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_temp,(-1,E,2*Nt))
    x_est_ex = np.transpose(x_est_ex,(2,1,0))
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')

    return sess

def do_testing_CMMVD(testing_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    # state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    # done = state.get('done', [])
    # log = str(state.get('log', ''))
    saver = tf.train.Saver()
    saver.restore(sess, savemodel)

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T = LT

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train_D']
    x_train = DT['x_train_D']
    y_test = DT['y_test_D']
    x_test = DT['x_test_D']
    y_test_ex = DT['y_test_ex_D']
    x_test_ex = DT['x_test_ex_D']
    y_test_iw2 = y_test_ex[0:5000*E, :]
    x_test_iw2 = x_test_ex[0:5000*E, :]
    y_test_iw4 = y_test_ex[5000*E:10000*E, :]
    x_test_iw4 = x_test_ex[5000*E:10000*E, :]
    y_test_iw6 = y_test_ex[10000*E:15000*E, :]
    x_test_iw6 = x_test_ex[10000*E:15000*E, :]
    y_test_iw8 = y_test_ex[15000*E:20000*E, :]
    x_test_iw8 = x_test_ex[15000*E:20000*E, :]


    batch_size = 200

    nmse1 = np.zeros([5000*E,T])
    nmse  = np.zeros([1,T])
    nmse0_dB = np.zeros([T], dtype=np.float32)
    i = -1
    for name, xhat_, loss_, nmse_, nmse_1, train_, var_list in testing_stages:
        i = i+1
        # nmse1 = np.zeros([5000])
        nmse0 = sess.run(nmse_, feed_dict={prob.y_:y_test,prob.x_:x_test})
        # nmse0 = sess.run(nmse_, feed_dict={prob.y_: y_test_ex[15000:20000,:], prob.x_: x_test_ex[15000:20000,:]})
        nmse0_dB[i] = 10*np.log10(nmse0)
        print('In iteration '+str(i+1)+' , nmse = {nmse:.6f}dB.'.format(nmse=nmse0_dB[i]))
        x_t = sess.run(xhat_,feed_dict={prob.y_:y_test,prob.x_:x_test})
        nmse1[:,i] = 10*np.log10(np.sum((x_t-x_test)**2,axis=1)/np.sum(x_test**2,axis=1))
        nmse[0,i] = np.mean(nmse1[:,i])
        print('In iteration '+str(i+1)+' , nmse(average samples) = {nmse:.6f}dB.'.format(nmse=nmse[0,i]))

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*N*(Tg+1)))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*Nt*E])
    x_temp = np.zeros([20000*E,Nt*2], dtype=np.float32)
    for i in range(4):
        i_f = i*5000*E
        i_l = (i+1)*5000*E
        x_temp[i_f:i_l, :] = sess.run(xhat_, feed_dict={prob.y_: y_test_ex[i_f:i_l,:],
                                                          prob.x_: x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_temp, (-1,E,2*N*(Tg+1)))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    # D = dict(x_AMP=x_est, x_AMP_ex=x_est_ex)
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')


    return sess

def do_training_CMMVH(training_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T = LT
    U = 2
    E1 = int(E/U)

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train_H']
    x_train = DT['x_train_H']
    y_test = DT['y_test_H']
    x_test = DT['x_test_H']
    y_test_ex = DT['y_test_ex_H']
    x_test_ex = DT['x_test_ex_H']
    y_test_iw2 = y_test_ex[0:5000*E,:]
    x_test_iw2 = x_test_ex[0:5000*E,:]
    y_test_iw4 = y_test_ex[5000*E:10000*E, :]
    x_test_iw4 = x_test_ex[5000*E:10000*E, :]
    y_test_iw6 = y_test_ex[10000*E:15000*E, :]
    x_test_iw6 = x_test_ex[10000*E:15000*E, :]
    y_test_iw8 = y_test_ex[15000*E:20000*E, :]
    x_test_iw8 = x_test_ex[15000*E:20000*E, :]


    batch_size = 100

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            tbi = i % (1000*U)      # debug
            tbi_f = tbi*batch_size
            tbi_l = (tbi+1)*batch_size
            y_train_batch = y_train[tbi_f:tbi_l,:]
            x_train_batch = x_train[tbi_f:tbi_l,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_test, prob.x_:x_test})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*Nt))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*Nt*E])
    x_temp = np.zeros([20000*U,E1*Nt*2],dtype=np.float32)
    for i in range(4):
        i_f = i*5000*E1
        i_l = (i+1)*5000*E1
        x_temp[i_f:i_l,:] = sess.run(xhat_,feed_dict={prob.y_:y_test_ex[i_f:i_l,:],prob.x_:x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_temp,(-1,E,2*Nt))
    x_est_ex = np.transpose(x_est_ex,(2,1,0))
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')

    return sess

def do_testing_CMMVH(testing_stages,prob,LT,loaddata,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    # state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    # done = state.get('done', [])
    # log = str(state.get('log', ''))
    saver = tf.train.Saver()
    saver.restore(sess, savemodel)

    N = prob.N
    M = prob.M
    Nt = prob.Nt
    Mt = prob.Mt
    E = prob.E
    Tg = prob.Tg
    T = LT
    U = 2
    E1 = int(E/U)

    DT = loadmat(loaddata)
    A = DT['A']
    y_train = DT['y_train_H']
    x_train = DT['x_train_H']
    y_test = DT['y_test_H']
    x_test = DT['x_test_H']
    y_test_ex = DT['y_test_ex_H']
    x_test_ex = DT['x_test_ex_H']
    y_test_iw2 = y_test_ex[0:5000*U, :]
    x_test_iw2 = x_test_ex[0:5000*U, :]
    y_test_iw4 = y_test_ex[5000*U:10000*U, :]
    x_test_iw4 = x_test_ex[5000*U:10000*U, :]
    y_test_iw6 = y_test_ex[10000*U:15000*U, :]
    x_test_iw6 = x_test_ex[10000*U:15000*U, :]
    y_test_iw8 = y_test_ex[15000*U:20000*U, :]
    x_test_iw8 = x_test_ex[15000*U:20000*U, :]


    batch_size = 200

    nmse1 = np.zeros([5000*U,T])
    nmse  = np.zeros([1,T])
    nmse0_dB = np.zeros([T], dtype=np.float32)
    i = -1
    for name, xhat_, loss_, nmse_, nmse_1, train_, var_list in testing_stages:
        i = i+1
        # nmse1 = np.zeros([5000])
        nmse0 = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
        # nmse0 = sess.run(nmse_, feed_dict={prob.y_: y_test_ex[15000:20000,:], prob.x_: x_test_ex[15000:20000,:]})
        nmse0_dB[i] = 10*np.log10(nmse0)
        print('In iteration '+str(i+1)+' , nmse = {nmse:.6f}dB.'.format(nmse=nmse0_dB[i]))
        for i1 in range(5000*U):
            ii1 = i1
            ii2 = (i1+1)
            y_input = np.reshape(y_test[ii1:ii2, :], (-1,E1*2*Mt))
            x_input = np.reshape(x_test[ii1:ii2, :], (-1,E1*2*Nt))
            nmse1[i1,i] = sess.run(nmse_, feed_dict={prob.y_: y_input, prob.x_: x_input})
            nmse1[i1,i] = 10*np.log10(nmse1[i1,i])
        nmse[0,i] = np.mean(nmse1[:,i])
        print('In iteration '+str(i)+' , nmse(average samples) = {nmse:.6f}dB.'.format(nmse=nmse[0,i]))

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1,E,2*N*(Tg+1)))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,2*Nt*E])
    x_temp = np.zeros([20000*U,E1*Nt*2], dtype=np.float32)
    for i in range(4):
        i_f = i*5000*U
        i_l = (i+1)*5000*U
        x_temp[i_f:i_l, :] = sess.run(xhat_, feed_dict={prob.y_: y_test_ex[i_f:i_l,:],
                                                          prob.x_: x_test_ex[i_f:i_l,:]})
    # x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_temp, (-1,E,2*N*(Tg+1)))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    # D = dict(x_AMP=x_est, x_AMP_ex=x_est_ex)
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')


    return sess


def do_training(training_stages,prob,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))
    saver = tf.train.Saver()

    #x_np = np.zeros([800,1000])

    batch_size = 200
    y_all = prob.yval
    x_all = prob.xval
    #y_train = y_all
    #x_train = x_all
    y_train = y_all[:,5000:105000]
    x_train = x_all[:,5000:105000]
    y_test = y_all[:,0:5000]
    x_test = x_all[:,0:5000]
    y_test_ex_all = prob.y_test_ex
    x_test_ex_all = prob.x_test_ex
    y_test_iw3 = y_test_ex_all[:,0:5000]
    x_test_iw3 = x_test_ex_all[:,0:5000]
    y_test_iw6 = y_test_ex_all[:, 5000:10000]
    x_test_iw6 = x_test_ex_all[:, 5000:10000]
    y_test_iw9 = y_test_ex_all[:, 10000:15000]
    x_test_iw9 = x_test_ex_all[:, 10000:15000]
    y_test_iw12 = y_test_ex_all[:, 15000:20000]
    x_test_iw12 = x_test_ex_all[:, 15000:20000]

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            training_batch_index = i % 500
            y_train_batch = y_train[:,training_batch_index*batch_size:(training_batch_index+1)*batch_size]
            x_train_batch = x_train[:,training_batch_index*batch_size:(training_batch_index+1)*batch_size]
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                #x_est = sess.run(xhat_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            if i%1000 == 0:
                nmse_test = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
                nmse_test_dB = 10 * np.log10(nmse_test)
                print('\rSNR=0dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_dB))
                #nmse_test_ex = np.zeros([4]).astype(np.float32)
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw3, prob.x_: x_test_iw3})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=3dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw6, prob.x_: x_test_iw6})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=6dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw9, prob.x_: x_test_iw9})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=9dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw12, prob.x_: x_test_iw12})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=12dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))

            #y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)

        #nmse_test = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})

        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

        nmse_test = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
        nmse_test_dB = 10 * np.log10(nmse_test)
        print('\rSNR=0dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_dB))
        # nmse_test_ex = np.zeros([4]).astype(np.float32)
        nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw3, prob.x_: x_test_iw3})
        nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
        print('\rSNR=3dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
        nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw6, prob.x_: x_test_iw6})
        nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
        print('\rSNR=6dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
        nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw9, prob.x_: x_test_iw9})
        nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
        print('\rSNR=9dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
        nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw12, prob.x_: x_test_iw12})
        nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
        print('\rSNR=12dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))


    my_model = saver.save(sess, savemodel)

    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')


    #nmse = sess.run(nmse_, feed_dict={prob.yval:y, prob.xval:x})
    #y, x = prob(sess)
    #x_np = x.eval(Session=sess)
    #savemat(savefilemat, {'x_est':x_np})

    return sess

def do_training_MMV(training_stages,prob,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))
    saver = tf.train.Saver()

    #x_np = np.zeros([800,1000])

    A = prob.A
    M,N = A.shape
    E = prob.E

    batch_size = 200
    y_all = prob.yval
    x_all = prob.xval
    #y_train = y_all
    #x_train = x_all
    y_train = y_all[5000:105000,:]   # debug
    x_train = x_all[5000:105000,:]   # debug
    y_test = y_all[0:5000,:]
    x_test = x_all[0:5000,:]
    y_test_ex_all = prob.y_test_ex
    x_test_ex_all = prob.x_test_ex
    y_test_iw3 = y_test_ex_all[0:5000,:]
    x_test_iw3 = x_test_ex_all[0:5000,:]
    y_test_iw6 = y_test_ex_all[5000:10000,:]
    x_test_iw6 = x_test_ex_all[5000:10000,:]
    y_test_iw9 = y_test_ex_all[10000:15000,:]
    x_test_iw9 = x_test_ex_all[10000:15000,:]
    y_test_iw12 = y_test_ex_all[15000:20000,:]
    x_test_iw12 = x_test_ex_all[15000:20000,:]

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            training_batch_index = i % 500      # debug
            y_train_batch = y_train[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            x_train_batch = x_train[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_test, prob.x_:x_test})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                #x_est = sess.run(xhat_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin() - 1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            if i%1000 == 0:
                nmse_test = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
                nmse_test_dB = 10 * np.log10(nmse_test)
                print('\rSNR=0dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_dB))
                #nmse_test_ex = np.zeros([4]).astype(np.float32)
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw3, prob.x_: x_test_iw3})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=3dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw6, prob.x_: x_test_iw6})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=6dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw9, prob.x_: x_test_iw9})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=9dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw12, prob.x_: x_test_iw12})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=12dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))

            #y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    # nmse_test = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test, prob.x_: x_test})
    x_est = np.reshape(x_est, (-1, E, N))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros([20000,N*E])
    for i in range(4):
        x_est_ex[i*5000:(i+1)*5000,:] = sess.run(xhat_,
                                                 feed_dict={prob.y_:prob.y_test_ex[i*5000:(i+1)*5000,:],prob.x_:prob.x_test_ex[i*5000:(i+1)*5000,:]})
    #x_est_ex = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex, prob.x_: prob.x_test_ex})
    x_est_ex = np.reshape(x_est_ex, (-1, E, N))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    D = dict(x_LAMP=x_est, x_LAMP_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')

    #nmse = sess.run(nmse_, feed_dict={prob.yval:y, prob.xval:x})
    #y, x = prob(sess)
    #x_np = x.eval(Session=sess)
    #savemat(savefilemat, {'x_est':x_np})

    return sess

def do_training_MMV_D(training_stages,prob,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))
    saver = tf.train.Saver()

    #x_np = np.zeros([800,1000])

    A = prob.A
    M,N = A.shape
    E = prob.E
    b_size = 5000*E

    batch_size = 200
    y_all = prob.y_signal_D
    x_all = prob.x_D
    #y_train = y_all
    #x_train = x_all
    y_train = y_all[5000*E:105000*E,:]   # debug
    x_train = x_all[5000*E:105000*E,:]   # debug

    #index_shuffle = np.random.permutation(100000*E)
    #y_train_shuffle = y_train[index_shuffle,:]
    #x_train_shuffle = x_train[index_shuffle,:]


    y_test = y_all[0:5000*E,:]
    x_test = x_all[0:5000*E,:]
    y_test_ex_all = prob.y_test_ex_D
    x_test_ex_all = prob.x_test_ex_D
    y_test_iw3 = y_test_ex_all[0:5000*E,:]
    x_test_iw3 = x_test_ex_all[0:5000*E,:]
    y_test_iw6 = y_test_ex_all[5000*E:10000*E,:]
    x_test_iw6 = x_test_ex_all[5000*E:10000*E,:]
    y_test_iw9 = y_test_ex_all[10000*E:15000*E,:]
    x_test_iw9 = x_test_ex_all[10000*E:15000*E,:]
    y_test_iw12 = y_test_ex_all[15000*E:20000*E,:]
    x_test_iw12 = x_test_ex_all[15000*E:20000*E,:]

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            training_batch_index = i % (500*E)      # debug
            y_train_batch = y_train[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            x_train_batch = x_train[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_train_batch, prob.x_:x_train_batch})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
                #x_est = sess.run(xhat_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            if i%1000 == 0:
                nmse_test = sess.run(nmse_, feed_dict={prob.y_: y_test, prob.x_: x_test})
                nmse_test_dB = 10 * np.log10(nmse_test)
                print('\rSNR=0dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_dB))
                #nmse_test_ex = np.zeros([4]).astype(np.float32)
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw3, prob.x_: x_test_iw3})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=3dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw6, prob.x_: x_test_iw6})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=6dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw9, prob.x_: x_test_iw9})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=9dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_iw12, prob.x_: x_test_iw12})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=12dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))

            #y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    # nmse_test = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
    x_est = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_D, prob.x_: prob.x_test_D})
    x_est = np.reshape(x_est, (-1, E, N))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros(shape=(20000*E,N)).astype(np.float32)
    for i in range(4):
        x_est_ex[i*b_size:(i+1)*b_size,:] = sess.run(xhat_, feed_dict={prob.y_: prob.y_test_ex_D[i*b_size:(i+1)*b_size,:], prob.x_: prob.x_test_ex_D[i*b_size:(i+1)*b_size,:]})
    x_est_ex = np.reshape(x_est_ex, (-1, E, N))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    D = dict(x_LAMP_D=x_est, x_LAMP_D_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')




    #nmse = sess.run(nmse_, feed_dict={prob.yval:y, prob.xval:x})
    #y, x = prob(sess)
    #x_np = x.eval(Session=sess)
    #savemat(savefilemat, {'x_est':x_np})

    return sess

def do_training_MMV_H(training_stages,prob,savefile,savefilemat,savemodel,ivl=10,maxit=1000000,better_wait=500):
    # ivl=10,maxit=1000000,better_wait=5000
    """
    ivl:how often should we compute the nmse of the validation set?
    maxit: max number of training iterations
    better_wait:wait this many iterations for an nmse that is better than the prevoius best of the current training session
    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval) ) )

    state = load_trainable_vars(sess,savefile) # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done=state.get('done',[])
    log=str(state.get('log',''))
    saver = tf.train.Saver()

    #x_np = np.zeros([800,1000])

    A = prob.A
    M,N = A.shape
    E = prob.E
    E_net = 2
    g = 2
    b_size = 5000*E

    batch_size = 200
    y_all = prob.y_signal_D
    x_all = prob.x_D
    #y_train = y_all
    #x_train = x_all
    y_train = y_all[5000*E:105000*E,:]   # debug
    x_train = x_all[5000*E:105000*E,:]   # debug
    y_train_H = np.reshape(y_train, (100000*E_net,-1))
    x_train_H = np.reshape(x_train, (100000*E_net,-1))

    #index_shuffle = np.random.permutation(100000*E)
    #y_train_shuffle = y_train[index_shuffle,:]
    #x_train_shuffle = x_train[index_shuffle,:]


    y_test = y_all[0:5000*E,:]
    x_test = x_all[0:5000*E,:]
    y_test_H = np.reshape(y_test, (5000*E_net,-1))
    x_test_H = np.reshape(x_test, (5000*E_net,-1))
    y_test_ex_all = prob.y_test_ex_D
    x_test_ex_all = prob.x_test_ex_D
    y_test_H_ex = np.reshape(y_test_ex_all, (20000*E_net,-1))
    x_test_H_ex = np.reshape(x_test_ex_all, (20000*E_net,-1))
    y_test_iw3 = y_test_ex_all[0:5000*E,:]
    x_test_iw3 = x_test_ex_all[0:5000*E,:]
    y_test_H_iw3 = np.reshape(y_test_iw3, (5000*E_net,-1))
    x_test_H_iw3 = np.reshape(x_test_iw3, (5000*E_net,-1))
    y_test_iw6 = y_test_ex_all[5000*E:10000*E,:]
    x_test_iw6 = x_test_ex_all[5000*E:10000*E,:]
    y_test_H_iw6 = np.reshape(y_test_iw6, (5000*E_net,-1))
    x_test_H_iw6 = np.reshape(x_test_iw6, (5000*E_net,-1))
    y_test_iw9 = y_test_ex_all[10000*E:15000*E,:]
    x_test_iw9 = x_test_ex_all[10000*E:15000*E,:]
    y_test_H_iw9 = np.reshape(y_test_iw9, (5000*E_net,-1))
    x_test_H_iw9 = np.reshape(x_test_iw9, (5000*E_net,-1))
    y_test_iw12 = y_test_ex_all[15000*E:20000*E,:]
    x_test_iw12 = x_test_ex_all[15000*E:20000*E,:]
    y_test_H_iw12 = np.reshape(y_test_iw12, (5000*E_net,-1))
    x_test_H_iw12 = np.reshape(x_test_iw12, (5000*E_net,-1))

    for name,xhat_,loss_,nmse_,train_,var_list in training_stages:
        if name in done:
            print('Already did ' + name + '. Skipping.')
            continue
        if len(var_list):
            describe_var_list = 'extending ' + ','.join([v.name for v in var_list])
        else:
            describe_var_list = 'fine tuning all ' + ','.join([v.name for v in tf.trainable_variables() ])

        print(name + ' ' + describe_var_list)
        nmse_history=[]
        for i in range(maxit+1):
            training_batch_index = i % (500*g)      # debug
            y_train_batch = y_train_H[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            x_train_batch = x_train_H[training_batch_index*batch_size:(training_batch_index+1)*batch_size,:]
            #y_train_batch = np.reshape(y_test[training_batch_index,:], (1,-1))
            #x_train_batch = np.reshape(x_test[training_batch_index,:], (1,-1))
            #xhat_temp = sess.run(xhat_, feed_dict={prob.y_:y_train_batch, prob.x_:x_train_batch})
            if i%ivl == 0:
                nmse = sess.run(nmse_,feed_dict={prob.y_:y_test_H,prob.x_:x_test_H})
                #x_est = sess.run(xhat_,feed_dict={prob.y_:prob.yval,prob.x_:prob.xval})
                if np.isnan(nmse):
                    raise RuntimeError('nmse is NaN')
                nmse_history = np.append(nmse_history,nmse)
                nmse_dB = 10*np.log10(nmse)
                nmsebest_dB = 10*np.log10(nmse_history.min())
                sys.stdout.write('\ri={i:<6d} nmse={nmse:.6f} dB (best={best:.6f})'.format(i=i,nmse=nmse_dB,best=nmsebest_dB))
                sys.stdout.flush()
                if i%(10*ivl) == 0:
                    print('')
                    age_of_best = len(nmse_history) - nmse_history.argmin()-1 # how long ago was the best nmse?
                    if age_of_best*ivl > better_wait:
                        break # if it has not improved on the best answer for quite some time, then move along
            if i%1000 == 0:
                nmse_test = sess.run(nmse_, feed_dict={prob.y_: y_test_H, prob.x_: x_test_H})
                nmse_test_dB = 10 * np.log10(nmse_test)
                print('\rSNR=0dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_dB))
                #nmse_test_ex = np.zeros([4]).astype(np.float32)
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_H_iw3, prob.x_: x_test_H_iw3})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=3dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_H_iw6, prob.x_: x_test_H_iw6})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=6dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_H_iw9, prob.x_: x_test_H_iw9})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=9dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))
                nmse_test_ex = sess.run(nmse_, feed_dict={prob.y_: y_test_H_iw12, prob.x_: x_test_H_iw12})
                nmse_test_ex_dB = 10 * np.log10(nmse_test_ex)
                print('\rSNR=12dB, nmse={nmse:.6f} dB'.format(nmse=nmse_test_ex_dB))

            #y,x = prob(sess)
            sess.run(train_,feed_dict={prob.y_:y_train_batch,prob.x_:x_train_batch} )
        done = np.append(done,name)


        log =  log+'\n{name} nmse={nmse:.6f} dB in {i} iterations'.format(name=name,nmse=nmse_dB,i=i)

        state['done'] = done
        state['log'] = log
        save_trainable_vars(sess,savefile,**state)

    my_model = saver.save(sess, savemodel)

    # nmse_test = sess.run(nmse_,feed_dict={prob.y_:y_test,prob.x_:x_test})
    x_est = sess.run(xhat_, feed_dict={prob.y_: y_test_H, prob.x_: x_test_H})
    x_est = np.reshape(x_est, (-1,E,N))
    x_est = np.transpose(x_est, (2,1,0))
    x_est_ex = np.zeros(shape=(20000*g,N*E_net)).astype(np.float32)
    b_H_size = 5000*g
    for i in range(4):
        x_est_ex[i*b_H_size:(i+1)*b_H_size,:] = sess.run(xhat_, feed_dict={prob.y_: y_test_H_ex[i*b_H_size:(i+1)*b_H_size,:], prob.x_: x_test_H_ex[i*b_H_size:(i+1)*b_H_size,:]})
    x_est_ex = np.reshape(x_est_ex, (-1,E,N))
    x_est_ex = np.transpose(x_est_ex, (2,1,0))
    D = dict(x_LAMP_H=x_est, x_LAMP_H_ex=x_est_ex)
    savemat(savefilemat, D, oned_as='column')

    return sess

def do_testing(testing_stages,prob,savefile,savefilemat,savemodel,ivl=1,maxit=1,better_wait=5000):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    #saver = tf.train.Saver()
    #saver.restore(sess, savemodel)


    #y_all = prob.yval
    #x_all = prob.xval
    #y_train = y_all[:, 5000:205000]
    #x_train = x_all[:, 5000:205000]
    #y_test = y_all[:, 0:5000]
    #x_test = x_all[:, 0:5000]
    #y_test_ex_all = prob.y_test_ex
    #x_test_ex_all = prob.x_test_ex
    #y_test_iw = y_test_ex_all[:, 5000:10000]
    #x_test_iw = x_test_ex_all[:, 5000:10000]

    A = prob.A
    M,N = A.shape
    #E = prob.E
    #b_size = 5000*E

    batch_size = 200
    y_all = prob.yval
    x_all = prob.xval
    # y_train = y_all
    # x_train = x_all
    y_train = y_all[:, 5000:105000]
    x_train = x_all[:, 5000:105000]
    y_test = y_all[:, 0:5000]
    x_test = x_all[:, 0:5000]
    y_test_ex_all = prob.y_test_ex
    x_test_ex_all = prob.x_test_ex
    y_test_iw3 = y_test_ex_all[:, 0:5000]
    x_test_iw3 = x_test_ex_all[:, 0:5000]
    y_test_iw6 = y_test_ex_all[:, 5000:10000]
    x_test_iw6 = x_test_ex_all[:, 5000:10000]
    y_test_iw9 = y_test_ex_all[:, 10000:15000]
    x_test_iw9 = x_test_ex_all[:, 10000:15000]
    y_test_iw12 = y_test_ex_all[:, 15000:20000]
    x_test_iw12 = x_test_ex_all[:, 15000:20000]


    for name,xhat_,loss_,nmse_,train_,var_list in testing_stages:
        nmse = sess.run(nmse_, feed_dict={prob.y_:y_test, prob.x_:x_test})
        #x_est = sess.run(xhat_, feed_dict={prob.y_: prob.yval, prob.x_: prob.xval})
        nmse_dB = 10 * np.log10(nmse)
        print('\rnmse={nmse:.6f} dB'.format(nmse=nmse_dB))

    return sess

def do_testing_MMV(testing_stages,prob,savefile,savefilemat,savemodel,ivl=1,maxit=1,better_wait=5000):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()
    saver.restore(sess, savemodel)


    A = prob.A
    M, N = A.shape
    E = prob.E
    #b_size = 5000
    b_size = 20000


    y_test = prob.y_test
    x_test = prob.x_test
    y_test_ex_all = prob.y_test_ex
    x_test_ex_all = prob.x_test_ex
    y_test_iw3 = y_test_ex_all[0:b_size, :]
    x_test_iw3 = x_test_ex_all[0:b_size, :]
    y_test_iw6 = y_test_ex_all[b_size:2*b_size, :]
    x_test_iw6 = x_test_ex_all[b_size:2*b_size, :]
    y_test_iw9 = y_test_ex_all[2*b_size:3*b_size, :]
    x_test_iw9 = x_test_ex_all[2*b_size:3*b_size, :]
    y_test_iw12 = y_test_ex_all[3*b_size:4*b_size, :]
    x_test_iw12 = x_test_ex_all[3*b_size:4*b_size, :]

    for name,xhat_,loss_,nmse_,nmse_1,train_,var_list in testing_stages:
        nmse1 = sess.run(nmse_, feed_dict={prob.y_:y_test, prob.x_:x_test})
    nmse_dB = 10 * np.log10(nmse1)

    x_est_ex_temp = np.zeros([4*b_size, N*E]).astype(np.float32)
    for i in range(16):
        x_est_ex_temp[i * 5000:(i + 1) * 5000, :] = sess.run(xhat_,
            feed_dict={prob.y_: prob.y_test_ex[i * 5000:(i + 1) * 5000, :],
            prob.x_: prob.x_test_ex[i * 5000:(i + 1) * 5000, :]})
    x_est_ex_temp = np.reshape(x_est_ex_temp, (-1,E,N))
    x_est_ex_temp = np.transpose(x_est_ex_temp, (2,1,0))
    D = dict(x_LAMP_ex=x_est_ex_temp)
    savemat(savefilemat, D, oned_as='column')

    print('done!')

    return sess

def do_testing_MMV_D(testing_stages,prob,savefile,savefilemat,savemodel,ivl=1,maxit=1,better_wait=5000):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()
    saver.restore(sess, savemodel)

    A = prob.A
    M, N = A.shape
    E = prob.E
    b_size = 20000

    y_all = prob.y_signal_D
    x_all = prob.x_D
    y_test = y_all[0:b_size * E, :]
    x_test = x_all[0:b_size * E, :]
    y_test_vect = np.reshape(y_test, (-1, M * E))
    x_test_vect = np.reshape(x_test, (-1, N * E))
    y_test_ex_all = prob.y_test_ex_D
    x_test_ex_all = prob.x_test_ex_D
    y_test_vect = np.reshape(y_test, (-1, M * E))
    x_test_vect = np.reshape(x_test, (-1, N * E))

    for name, xhat_, loss_, nmse_, nmse_1, train_, var_list in testing_stages:
        nmse1 = sess.run(nmse_1, feed_dict={prob.y_: y_test, prob.x_: x_test})
    nmse_dB = 10*np.log10(nmse1)

    x_est_ex_temp = np.zeros(shape=(80000 * E, N)).astype(np.float32)
    for i in range(16):
        x_est_ex_temp[i * b_size:(i + 1) * b_size, :] = sess.run(xhat_, feed_dict={
            prob.y_: y_test_ex_all[i * b_size:(i + 1) * b_size, :],
            prob.x_: x_test_ex_all[i * b_size:(i + 1) * b_size, :]})
    x_est_ex_temp = np.reshape(x_est_ex_temp, (-1, E, N))
    x_est_ex_temp = np.transpose(x_est_ex_temp, (2, 1, 0))
    D = dict(x_LAMP_ex=x_est_ex_temp)
    savemat(savefilemat, D, oned_as='column')

    return sess

def do_testing_MMV_H(testing_stages,prob,savefile,savefilemat,savemodel,ivl=1,maxit=1,better_wait=5000):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('norms xval:{xval:.7f} yval:{yval:.7f}'.format(xval=la.norm(prob.xval), yval=la.norm(prob.yval)))

    state = load_trainable_vars(sess, savefile)  # must load AFTER the initializer

    # must use this same Session to perform all training
    # if we start a new Session, things would replay and we'd be training with our validation set (no no)

    done = state.get('done', [])
    log = str(state.get('log', ''))
    saver = tf.train.Saver()
    saver.restore(sess, savemodel)

    A = prob.A
    M, N = A.shape
    E = prob.E
    E_net = 2
    g = 2
    b_size = 20000 * E

    y_all = prob.y_signal_D
    x_all = prob.x_D
    y_test = y_all[0:20000 * E, :]
    x_test = x_all[0:20000 * E, :]
    y_test_H = np.reshape(y_test, (20000 * E_net, -1))
    x_test_H = np.reshape(x_test, (20000 * E_net, -1))
    y_test_ex_all = prob.y_test_ex_D
    x_test_ex_all = prob.x_test_ex_D
    y_test_H_ex = np.reshape(y_test_ex_all, (80000 * E_net, -1))
    x_test_H_ex = np.reshape(x_test_ex_all, (80000 * E_net, -1))

    for name, xhat_, loss_, nmse_, nmse_1, train_, var_list in testing_stages:
        nmse1 = sess.run(nmse_1, feed_dict={prob.y_: y_test_H, prob.x_: x_test_H})
    nmse_dB = 10*np.log10(nmse1)
    x_est_ex_temp = np.zeros(shape=(80000 * E_net, N * g)).astype(np.float32)
    bat = 5000*E_net
    for i in range(16):
        x_est_ex_temp[i*bat:(i+1)*bat,:] = sess.run(xhat_,feed_dict={
            prob.y_: y_test_H_ex[i * bat:(i + 1) * bat, :],
            prob.x_: x_test_H_ex[i * bat:(i + 1) * bat, :]})
    x_est_ex_temp = np.reshape(x_est_ex_temp, (-1,E,N))
    x_est_ex_temp = np.transpose(x_est_ex_temp, (2,1,0))
    D = dict(x_LAMP_ex=x_est_ex_temp)
    savemat(savefilemat, D, oned_as='column')

    print('done!')

    return sess