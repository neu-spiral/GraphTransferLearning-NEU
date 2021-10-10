"""
Module contains data generator for batch training

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""


import numpy as np
from keras.utils import Sequence


class EmbeddingNodesGenerator(Sequence):
    """
    Generates pairs of nodes for Embedding model
    
    Inputs:
        A Adjacency matrix of the first (labeled) graph
        BATCH_SIZE Number of datapoints in each batch
        LABELS List of node labels of the first graph in one-hot coding format
        B Adjacency matrix of the second (unlabeled) graph
        P Permutation matrix 
        AFEATURES Matrix of the first graph nodes feature vectors
        BFEATURES Matrix of the second graph nodes feature vectors
        RANDOM_ORDER Flag that specifies whether datapoints should be shuffled between batches
        
    Returns batch data:
      Neural network input:
        ATARGET, ACONTEXT List of node pairs for the first graph
        BTARGET, BCONTEXT List of node pairs for the second graph
      Neural network output:
        ASIM Similarity between first graph nodes in each pair
        ATARGET_LABEL, ACONTEXT_LABEL Labels of first graph nodes in each pair
        BSIM Similarity between second graph nodes in each pair
        PT1T2, PT1C2, PC1T2, PC1C2 Pairwise distances between first and second graphs' nodes
        
    """

    def __init__(self, batch_size, A, Atarget, Acontext, Asimilarity, Anegative, n_negative=0, B=None, Btarget=None, Bcontext=None, Bsimilarity=None, Bnegative=None, P=None, Afeatures=None, Bfeatures=None):
        self.batch_size = batch_size
#        self.topology_similarity = topology_similarity
        self.n_negative = n_negative

        self.Anodes = np.array(range(len(A)))
        if Afeatures is not None:
            self.Afeatures = Afeatures
        else:
            self.Afeatures = self.Anodes.reshape(len(A),1) # 'dictionary' setting
        
        self.Atarget, self.Acontext, self.Asimilarity, self.Anegative = Atarget, Acontext, Asimilarity, Anegative

        if B is not None:
            self.two_graphs = True
            self.Bnodes = np.array(range(len(B)))  
            if Bfeatures is not None:
                self.Bfeatures = Bfeatures
            else:
                self.Bfeatures = self.Bnodes.reshape(len(B),1)
            
            self.Btarget, self.Bcontext, self.Bsimilarity, self.Bnegative = Btarget, Bcontext, Bsimilarity, Bnegative
            
            if len(self.Btarget) < len(self.Atarget):
                idx = np.random.permutation(len(self.Atarget))[0:len(self.Btarget)]
                self.Atarget = self.Atarget[idx]
                self.Acontext = self.Acontext[idx]
                self.Asimilarity = self.Asimilarity[idx,:]
                self.Anegative = self.Anegative[idx,:]
            elif len(self.Btarget) > len(self.Atarget):
                idx = np.random.permutation(len(self.Atarget))[0:len(self.Btarget)]
                self.Btarget = self.Btarget[idx]
                self.Bcontext = self.Bcontext[idx]
                self.Bsimilarity = self.Bsimilarity[idx,:]
                self.Bnegative = self.Bnegative[idx,:]

            if P is not None:
                self.transfer_learning = True
                self.P = P
            else:
                self.transfer_learning = False
        else:
            self.two_graphs = False
            self.transfer_learning = False
        
    def __len__(self):
        return int(np.ceil(len(self.Atarget) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        Atarget = self.Afeatures[self.Atarget[idx*self.batch_size : (idx+1)*self.batch_size],:]
        Acontext = self.Afeatures[self.Acontext[idx*self.batch_size : (idx+1)*self.batch_size],:]
        if self.Afeatures.shape[-1]>1:
            Atarget = Atarget.reshape((-1,1,self.Afeatures.shape[-1]))
            Acontext = Acontext.reshape((-1,1,self.Afeatures.shape[-1]))
        inputs = [np.array(Atarget), np.array(Acontext)]
        
        if self.n_negative > 0:
            Anegsampl = self.Afeatures[self.Anegative[idx*self.batch_size : (idx+1)*self.batch_size,:].reshape(-1,1),:]
            if self.Afeatures.shape[-1]>1:
                Anegsampl = Anegsampl.reshape((-1,self.n_negative,self.Afeatures.shape[-1]))
            else:
                Anegsampl = Anegsampl.reshape((-1,self.n_negative))
            inputs.extend([Anegsampl])
        
        Asim = self.Asimilarity[idx*self.batch_size : (idx+1)*self.batch_size, :].reshape((-1,1,self.n_negative+1))
        outputs = [np.array(Asim)]
        
        if self.two_graphs:
            Btarget = self.Bfeatures[self.Btarget[idx*self.batch_size : (idx+1)*self.batch_size],:]
            Bcontext = self.Bfeatures[self.Bcontext[idx*self.batch_size : (idx+1)*self.batch_size],:]
            if self.Bfeatures.shape[-1]>1:
                Btarget = Btarget.reshape((-1,1,self.Bfeatures.shape[-1]))
                Bcontext = Bcontext.reshape((-1,1,self.Bfeatures.shape[-1]))
            inputs.extend([np.array(Btarget), np.array(Bcontext)])
        
            if self.n_negative > 0:
                Bnegsampl = self.Bfeatures[self.Bnegative[idx*self.batch_size : (idx+1)*self.batch_size,:].reshape(-1,1),:]
                if self.Bfeatures.shape[-1]>1:
                    Bnegsampl = Bnegsampl.reshape((-1,self.n_negative,self.Bfeatures.shape[-1]))
                else:
                    Bnegsampl = Bnegsampl.reshape((-1,self.n_negative))
                inputs.extend([Bnegsampl])
            
            Bsim = self.Bsimilarity[idx*self.batch_size : (idx+1)*self.batch_size, :].reshape((-1,1,self.n_negative+1))
            outputs.extend([np.array(Bsim)])
        
            if self.transfer_learning:
                PtAtB = self.P[self.Atarget[idx*self.batch_size : (idx+1)*self.batch_size], self.Btarget[idx*self.batch_size : (idx+1)*self.batch_size]]
                outputs.extend([np.array(PtAtB)])
        
        return inputs, outputs


class PredictionNodesGenerator(Sequence):
    """
    Generates data for Graph Transfer Learning model
    
    Inputs:
        A Adjacency matrix of the first (labeled) graph
        BATCH_SIZE Number of datapoints in each batch
        LABELS List of node labels of the first graph in one-hot coding format
        B Adjacency matrix of the second (unlabeled) graph
        P Permutation matrix 
        AFEATURES Matrix of the first graph nodes feature vectors
        BFEATURES Matrix of the second graph nodes feature vectors
        RANDOM_ORDER Flag that specifies whether datapoints should be shuffled between batches
        
    Returns batch data:
      Neural network input:
        ATARGET, ACONTEXT List of node pairs for the first graph
        BTARGET, BCONTEXT List of node pairs for the second graph
      Neural network output:
        ASIM Similarity between first graph nodes in each pair
        ATARGET_LABEL, ACONTEXT_LABEL Labels of first graph nodes in each pair
        BSIM Similarity between second graph nodes in each pair
        PT1T2, PT1C2, PC1T2, PC1C2 Pairwise distances between first and second graphs' nodes
        
    """

    def __init__(self, batch_size, X, Y):
        self.batch_size = batch_size
        self.data = X
        self.labels = Y
        
    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        if self.data.shape[-1]>1:
            batch_data = batch_data.reshape((-1,1,self.data.shape[-1]))
        inputs = [batch_data]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        outputs = [batch_labels]
        return inputs, outputs
