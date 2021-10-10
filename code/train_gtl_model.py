"""
@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from keras.models import Model

import data_generator as DG
from load_save_data import save_iteration_results


# manually update 
def updateWeigths(WA, WB, P, lr=0.025):
    print("WA shape = {}, WB shape = {}".format(WA.shape, WB.shape))
    if WA.shape[0] != WB.shape[0]:
        P = P[:WA.shape[0], -WB.shape[0]:]
    WAnew = (1 - 2*lr)*WA + 2*lr*np.matmul(P, WB)
    WBnew = (1 - 2*lr)*WB + 2*lr*np.matmul(P.T, WA)
    return WAnew, WBnew


# outputs embedding for given raw data
def getEmbedding(model, graph_name, data):
    new_model = Model(inputs=model.get_layer('Input_Target'+graph_name).input,
                      outputs=model.get_layer('Embedding_Target'+graph_name).get_output_at(0))
    if data.shape[1] > 1:
        data = data.reshape((data.shape[0],1,data.shape[1]))
    return new_model.predict(data).squeeze()


# outputs prediction for given raw data
def getModelOutputs_fromRaw(model, data):
    if data.shape[1] > 1:
        data = data.reshape((data.shape[0], 1, data.shape[1]))
    return model.predict(data).squeeze()


# outputs prediction for given embedding of data
def getModelOutputs_fromEmbedding(model, data):
    return model.get_layer('Branch_Prediction').predict(data)


# compute matrix D of node-wise distance between two graphs
def computeDistanceBetweenGraphs(graphA, graphB, mode='l2'):
    if mode == 'innerprod': # inner product
        D = np.matmul(graphA, graphB.T)
    elif mode == 'l2sq': # squared l2 distance
        D = euclidean_distances(graphA, graphB, squared=True)
    else: #l2 distance
        D = euclidean_distances(graphA, graphB)
    if graphA.shape != graphB.shape:
        scale = 100
        D = euclidean_distances(graphA, graphB)
        D = np.concatenate((np.concatenate((np.ones((graphA.shape[0], graphA.shape[0]))*scale, D), axis=1),
                            np.concatenate((np.zeros(D.T.shape), np.ones((graphB.shape[0], graphB.shape[0]))*scale), axis=1)), axis=0)

    return D


# return predicted labels and regression values
def getModelPredictions(prediction):
    return prediction.max(axis=1, keepdims=1) == prediction


# compute classification and regression accuracy 
def computeAccuracy(true, prediction):
    if prediction.ndim == 2 and prediction.shape[1] > 1:
        #prediction accuracy
        prediction = getModelPredictions(prediction)
        acc = np.sum((np.argmax(true, axis=1) == np.argmax(prediction.astype(int), axis=1)).astype(int))/float(len(true)) if len(true) > 0 else 0
    else:
        try:
            prediction = np.delete(prediction, np.where(true == 1)[0][0])
            true = np.delete(true, np.where(true == 1)[0][0])
        except:
            pass
        # root mean squared error aka standard deviation of residuals
        acc = np.sqrt(np.mean(np.square(prediction - true)))
    return acc


# compute R-squared statistics aka coefficient of determination
def computeRSquared(true, prediction, avg_predictor=None):
    try:
        prediction = np.delete(prediction, np.where(true == 1)[0][0])
        true = np.delete(true, np.where(true == 1)[0][0])
    except:
        pass
    if avg_predictor is None:
        avg_predictor = np.mean(true)
    rsquared = 1 - np.sum(np.square(prediction - true))/np.sum(np.square(prediction - avg_predictor))
    return rsquared


def train(models, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels, Ptrue, transmode, topology_similarity, n_walks, walk_length, window_size, p, q, n_negative, learning_rate, beta, nepochs, early_stopping, batch_size, save_path):
    
    # file to save results
    suffix = '_' + str(nepochs) + '_'
    acc_txt = open(save_path + 'Accuracy' + suffix + '.txt', "w")
    
    # choose the period of how often results are saved
    step = max(int(nepochs/10), int(1)) if nepochs<=100 else max(int(nepochs/100), int(10))
    # variables to contain results
    epochs = []
    acc_train = []
    acc_testA = []
    acc_testB = []
    
    lr_init = 0.1
    
    patience = 0
    epoch_best = -1
    acc_curr = -float('inf') if Alabels.shape[1]>1 else float('inf')
    acc_best = -float('inf') if Alabels.shape[1]>1 else float('inf')
    weights_embed_best = None
    weights_pred_best = None
    
    EmbeddingModel = models['EmbeddingModel']
    EmbeddingABranch = models['EmbeddingABranch']
    EmbeddingBBranch = models['EmbeddingBBranch']
    PredictionModel = models['PredictionModel']
    
    # prepare data for training
    print("\n=======================================\nDATA IS LOADING")
    from graph_utils import getNodeSimilarity
    Atarget, Acontext, Asimilarity, Anegative = getNodeSimilarity(A, mode=topology_similarity, n_walks=n_walks,
                                                                  walk_length=walk_length, window_size=window_size, p=p,
                                                                  q=q, n_negative=n_negative)
    if B is None:
        Btarget, Bcontext, Bsimilarity, Bnegative = None, None, None, None
    else:
        Btarget, Bcontext, Bsimilarity, Bnegative = getNodeSimilarity(B, mode=topology_similarity, n_walks=n_walks,
                                                                      walk_length=walk_length, window_size=window_size,
                                                                      p=p, q=q, n_negative=n_negative)
    print("\n=======================================\nDATA IS LOADED")
    
    # initialize model
    if transmode not in ['1graph', 'noP']:
        # initialize model
        print("\n=======================================\nMODEL INITIALIZATION")
        # train embedding of graph A
        Data = DG.EmbeddingNodesGenerator(batch_size, A, Atarget, Acontext, Asimilarity, Anegative, n_negative=n_negative, Afeatures=Afeatures)
        print("\nEMBEDDING BRANCH A INITIALIZATION:")
        EmbeddingABranch.fit_generator(Data, epochs=1, verbose=2)
        # adjust embedding of graph A by learning class labels
        print("\nPREDICTION BRANCH INITIALIZATION:")
        PredictionModel.fit_generator(DG.PredictionNodesGenerator(batch_size, Afeatures[Atrain,:], Alabels[Atrain,:]), epochs=1, verbose=2)
        # compute training and testA accuracies
        epochs.append(0)
        Aembedding = getEmbedding(EmbeddingModel, 'A', Afeatures[np.concatenate((Atrain, Atest)),:])
        Alabels_output = getModelOutputs_fromEmbedding(PredictionModel, Aembedding)
        acc_train.append(computeAccuracy(Alabels[Atrain,:], Alabels_output[Atrain,:]))
        acc_testA.append(computeAccuracy(Alabels[Atest,:], Alabels_output[Atest,:]))
        # initialize P
        print("\nP INITIALIZATION:")
        if A.shape[0] != B.shape[0]:
            factor = 0.5
            Aext = np.concatenate((np.concatenate((A, np.ones((A.shape[0],B.shape[1]))*factor), axis=1),
                                   np.ones((B.shape[0], A.shape[1]+B.shape[1]))*factor), axis=0)
            Bext = np.concatenate((np.ones((A.shape[0], A.shape[1]+B.shape[1]))*factor,
                                   np.concatenate((np.ones((B.shape[0],A.shape[1]))*factor, B), axis=1)), axis=0)
        else:
            Aext, Bext = A, B
        if transmode.find('trueP') != -1:
            P = Ptrue
        else:
            from updateP import updateP
            P = updateP(Aext, Bext, np.zeros(Aext.shape), None, 'opt')
        # update A,B embeddings w.r.t. transfer loss
        W_TargetA = EmbeddingABranch.get_layer('Embedding_TargetA').get_weights()[0]
        W_TargetB = np.zeros(EmbeddingBBranch.get_layer('Embedding_TargetB').get_weights()[0].shape)
        W_TargetA, W_TargetB = updateWeigths(W_TargetA, W_TargetB, P, lr=lr_init)
        EmbeddingModel.get_layer('Embedding_TargetA').set_weights([W_TargetA])
        EmbeddingModel.get_layer('Embedding_TargetB').set_weights([W_TargetB])
        try:
            W_ContextA = EmbeddingABranch.get_layer('Embedding_ContextA').get_weights()[0]
            W_ContextB = np.zeros(EmbeddingBBranch.get_layer('Embedding_ContextB').get_weights()[0].shape)
            W_ContextA, W_ContextB = updateWeigths(W_ContextA, W_ContextB, P, lr=lr_init)
            EmbeddingModel.get_layer('Embedding_ContextA').set_weights([W_ContextA])
            EmbeddingModel.get_layer('Embedding_ContextB').set_weights([W_ContextB])
        except:
            pass
        # adjust embedding of graph B
        Data = DG.EmbeddingNodesGenerator(batch_size, B, Btarget, Bcontext, Bsimilarity, Bnegative, n_negative=n_negative, Afeatures=Bfeatures)
        print("\nEMBEDDING BRANCH B INITIALIZATION:")
        EmbeddingBBranch.fit_generator(Data, epochs=1, verbose=2)
        # compute testB accuracy
        Bembedding = getEmbedding(EmbeddingModel, 'B', Bfeatures)
        Blabels_output = getModelOutputs_fromEmbedding(PredictionModel, Bembedding)
        acc_testB.append(computeAccuracy(Blabels, Blabels_output))
        print(f"\n---------------------------------------\n"
              f"After model initialization\n"
              f"\tTrain accuracy = {acc_train[-1]:.4f} (graph A)\n"
              f"\tTest  accuracy = {acc_testA[-1]:.4f} (graph A)\n"
              f"\tTest  accuracy = {acc_testB[-1]:.4f} (graph B)")
        acc_txt.write("{:4d}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(epochs[-1], acc_train[-1], acc_testA[-1], acc_testB[-1]))
    else:
        P = None
            
    print("\n*************** Model Training ***************".upper())
    # train model
    for epoch in range(nepochs):
        print("\n=======================================\nEPOCH #{:4d}".format(epoch+1))
        
        # optimize permutation matrix and update A,B embeddings w.r.t. it
        print("\nTRANSFER TRAINING:")
        if transmode not in ['1graph', 'noP']:
            if transmode == 'trueP':
                print("True permutation P is used")
            elif transmode == 'trueP_DS':
                print("True doubly-stochastic P is used")
            else:
                D = computeDistanceBetweenGraphs(getEmbedding(EmbeddingModel,'A',Afeatures), getEmbedding(EmbeddingModel,'B',Bfeatures))
                # scale loss w.r.t. epoch number: give more value to second part of the loss as training progress
                if beta:
                    scalefunc = lambda x: x/(1+abs(x)) # S-shaped function
                    scalefactor = 5 # loss is scaled in range [1/scalefactor ... 1 ... scalefactor]
                    scale = scalefactor^(scalefunc(epoch-nepochs/2))
                    D *= scale
                P = updateP(Aext, Bext, D, P, transmode) # optimize permutation matrix P
                print("P is updated with '{}' method".format(transmode))
        else:
            print("No P is used (no transfer)")
        
        # train Embedding model for 1 epoch
        print("\nEMBEDDING TRAINING:")
        Data = DG.EmbeddingNodesGenerator(batch_size, A, Atarget, Acontext, Asimilarity, Anegative, n_negative, B, Btarget, Bcontext, Bsimilarity, Bnegative, P, Afeatures=Afeatures, Bfeatures=Bfeatures)
        EmbeddingModel.fit_generator(Data, epochs=1, verbose=2)
        
        # train Prediction model for 1 epoch
        print("\nPREDICTION TRAINING:")
        PredictionModel.fit_generator(DG.PredictionNodesGenerator(batch_size, Afeatures[Atrain, :], Alabels[Atrain, :]), epochs=1, verbose=2)
        
        # compute training accuracy
        epochs.append(epoch+1)
        Aembedding = getEmbedding(EmbeddingModel, 'A', Afeatures[np.concatenate((Atrain, Atest)), :])
        Alabels_output = getModelOutputs_fromEmbedding(PredictionModel, Aembedding)
        acc_curr = computeAccuracy(Alabels[Atrain,:], Alabels_output[Atrain,:])
        acc_train.append(acc_curr)
        
        # check for training accuracy improvement
        if (Alabels.shape[1]>1 and acc_curr>acc_best) or (Alabels.shape[1]==1 and acc_curr<acc_best):
            print(f"\nEPOCH #{epoch+1:4d} :: Train accuracy has improved from the previous best value "
                  f"at epoch #{epoch_best+1:4d}: {acc_best:.4f} -> {acc_curr:.4f}")
            patience = 0
            epoch_best = epoch
            acc_best = acc_curr
            weights_embed_best = EmbeddingModel.get_weights()
            weights_pred_best = PredictionModel.get_weights()
        else:
            print("\nEPOCH #{:4d} :: Train accuracy ({:.4f}) has not improved from the previous best value at epoch #{:4d} ({:.4f})".format(epoch+1, acc_curr, epoch_best+1, acc_best))
            patience += 1
        
        # output intermediate training and test results at given interval
        if epoch==0 or (epoch+1)%step==0 or epoch==nepochs-1:
            acc_testA.append(computeAccuracy(Alabels[Atest,:], Alabels_output[Atest,:])) # compute A test accuracy
            if B is None:
                print("\n---------------------------------------\n\tTrain accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph A)\n---------------------------------------\n".format(acc_train[-1], acc_testA[-1]))
                acc_txt.write("{:4d}\t{:.4f}\t{:.4f}\n".format(epochs[-1], acc_train[-1], acc_testA[-1]))
            else:
                Bembedding = getEmbedding(EmbeddingModel, 'B', Bfeatures)
                Blabels_output = getModelOutputs_fromEmbedding(PredictionModel, Bembedding)
                acc_testB.append(computeAccuracy(Blabels, Blabels_output))
                print("\n---------------------------------------\n\tTrain accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph B)\n---------------------------------------\n".format(acc_train[-1], acc_testA[-1], acc_testB[-1]))
                acc_txt.write("{:4d}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(epochs[-1], acc_train[-1], acc_testA[-1], acc_testB[-1]))
        
        # check for early stopping 
        if (early_stopping>0 and patience>early_stopping-1):
            print("Early stopping.")
            break
        # end of training 
        if epoch==nepochs-1:
            print("\nMaximum # of epochs (" + str(nepochs) + ") is reached.\n")
    
    # rename output file 
    newsuffix = suffix + str(epoch_best+1)
    os.rename(save_path + 'Accuracy' + suffix + '.txt', 
              save_path + 'Accuracy' + newsuffix + '.txt')
    suffix = newsuffix
    
    # restore weights for best epoch
    EmbeddingModel.set_weights(weights_embed_best)
    PredictionModel.set_weights(weights_pred_best)
    
    # save best model states (with weights)
    EmbeddingModel.save(save_path + 'EmbeddingModel' + suffix + '.h5')
    PredictionModel.save(save_path + 'PredictionModel' + suffix + '.h5')
    
    # compute, save and output best iteration results
    Aembedding = getEmbedding(EmbeddingModel, 'A', Afeatures[np.concatenate((Atrain, Atest)),:]) # graph A node embeddings
    Alabels_output = getModelOutputs_fromEmbedding(PredictionModel, Aembedding)
    Alabels_predict = getModelPredictions(Alabels_output) if Alabels.shape[1]>1 else None
    epochs.append(epoch_best+1)
    acc_train.append(acc_best)
    acc_testA.append(computeAccuracy(Alabels[Atest,:], Alabels_output[Atest,:]))
    acc_results = dict(zip(['epochs', 'acc_train', 'acc_testA'], [epochs, acc_train, acc_testA]))
    best_results = dict(zip(['epochs', 'acc_train', 'acc_testA'], [epochs[-1], acc_train[-1], acc_testA[-1]]))
    print("\n---------------------------------------\nRestoring best model state from epoch #{:4d}:".format(epochs[-1]))
    if B is None:
        Bembedding = None
        Blabels_output = None
        Blabels_predict = None
        if Alabels.shape[1]>1:
            print("Train accuracy = {:.4f} (graph A)\t:\tTest accuracy = {:.4f} (graph A)".format(acc_train[-1], acc_testA[-1]))
            acc_txt.write("-"*24 + "\n{:4d}\t{:.4f}\t{:.4f}".format(epochs[-1], acc_train[-1], acc_testA[-1]))
        else:
            rsquared = computeRSquared(Alabels[Atest,:], Alabels_output[Atest,:], np.mean(Alabels[Atrain,:]))
            best_results['acc_rsquaredA'] = rsquared
            print("\tTrain accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph A)\t:\tR-squared = {:.4f} (graph A)".format(acc_train[-1], acc_testA[-1], rsquared))
            acc_txt.write("-"*32 + "\n{:4d}\t{:.4f}\t{:.4f}\t{:.4f}".format(epochs[-1], acc_train[-1], acc_testA[-1], rsquared))
    else:
        Bembedding = getEmbedding(EmbeddingModel, 'B', Bfeatures) # graph B node embeddings
        Blabels_output = getModelOutputs_fromEmbedding(PredictionModel, Bembedding)
        Blabels_predict = getModelPredictions(Blabels_output) if Alabels.shape[1]>1 else None
        acc_testB.append(computeAccuracy(Blabels, Blabels_output))
        acc_results['acc_testB'] = acc_testB
        best_results['acc_testB'] = acc_testB[-1]
        if Alabels.shape[1]>1:
            print("\tTrain accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph B)".format(acc_train[-1], acc_testA[-1], acc_testB[-1]))
            acc_txt.write("-"*32 + "\n{:4d}\t{:.4f}\t{:.4f}\t{:.4f}".format(epochs[-1], acc_train[-1], acc_testA[-1], acc_testB[-1]))
        else:
            rsquaredA = computeRSquared(Alabels[Atest,:], Alabels_output[Atest,:], np.mean(Alabels[Atrain,:]))
            rsquaredB = computeRSquared(Blabels, Blabels_output, np.mean(Alabels[Atrain,:]))
            best_results['acc_rsquaredA'] = rsquaredA
            best_results['acc_rsquaredB'] = rsquaredB
            print("\tTrain accuracy = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph A)\t:\tR-squared = {:.4f} (graph A)\n\tTest  accuracy = {:.4f} (graph B)\t:\tR-squared = {:.4f} (graph B)".format(acc_train[-1], acc_testA[-1], rsquaredA, acc_testB[-1], rsquaredB))
            acc_txt.write("-"*48 + "\n{:4d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epochs[-1], acc_train[-1], acc_testA[-1], rsquaredA, acc_testB[-1], rsquaredB))
    
    acc_txt.close()

    save_iteration_results(save_path, suffix, Atrain, Atest, Aembedding, Alabels_output, Alabels_predict, B, Bfeatures,
                           Bembedding, Blabels, Blabels_output, Blabels_predict, Ptrue, P, acc_results)
    
    return best_results















