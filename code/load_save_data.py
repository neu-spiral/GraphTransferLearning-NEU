"""
Module is responsible for results saving

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import os
import numpy as np
import pickle

from graph_utils import PermuteGraph, ReverseEdges, GetDoublyStochasticMatrix


def load_data(dataset_path, labels, transmode, B_from_A='permute', disc_pect=None):
    """
    Loads dataset

    INPUTS:
        LOAD_PATH Path the data directory
        FEATUES Specifies type of features to be used
        LABELS Specifies type of labels to be used
        TWO_GRAPHS Specifies whether two graphs should be loaded
        TRANSFER_LEARNING Specifies if permutation matrix P should be created
        VISUALIZE Specifies whether graphs should be plotted
        SAVE_PATH Path where graph plots should be saved

    OUTPUTS:
        NNODES Number of nodes in graphs
        NFEATURES Number of node features
        NLABELS Numbeer of node labels
        A Adjacency matrix of the first graph
        AFEATURES Node features of the first graph
        ALABELS Node labels of the first graph
        ATRAIN Indices of nodes from the first graph in the train set
        ATEST Indices of nodes from the second graph in the test set
        B Adjacency matrix of the second graph, if it is used
        BFEATURES Node features of the second graph, if it is used
        BLABELS Node labels of the second graph, if it is used
    """

    A = np.loadtxt(dataset_path + 'GraphA.txt').astype(int)
    nnodes = A.shape[0]
    if all(os.path.exists(os.path.join(dataset_path, f"GraphA{split}.txt")) for split in ['Train', 'Test']):
        Atrain = np.loadtxt(os.path.join(dataset_path, 'GraphATrain.txt')).astype(int)
        Atest = np.loadtxt(os.path.join(dataset_path, 'GraphATest.txt')).astype(int)
    else:
        ntrain = 0.8
        Aindices = np.random.permutation(len(A))
        Atrain = Aindices[:np.ceil(ntrain * len(A)).astype(int)]
        Atest = Aindices[np.ceil(ntrain * len(A)).astype(int):]

    if os.path.exists(os.path.join(dataset_path, 'GraphAFeatures.txt')):
        Afeatures = np.loadtxt(os.path.join(dataset_path, 'GraphAFeatures.txt'))
    else:
        Afeatures = np.arange(len(A)).reshape(-1, 1)
    if Afeatures.ndim == 1:
        Afeatures = Afeatures.reshape(-1, 1)
    nfeatures = Afeatures.shape[1]

    Alabels = np.loadtxt(os.path.join(dataset_path, 'GraphALabels_' + labels + '.txt'))
    if Alabels.ndim == 1:
        Alabels = Alabels.reshape(-1, 1)
    nlabels = Alabels.shape[1]

    if transmode == '1graph':
        B, Bfeatures, Blabels, P = None, None, None, None
    else:
        B, Blabels = np.copy(A), np.copy(Alabels)
        B, Blabels, P = PermuteGraph(B)
        if B_from_A == 'modify':
            B = ReverseEdges(B, disc_pect)
        if transmode == 'trueP_ds':
            P = GetDoublyStochasticMatrix(Alabels, P)
        if os.path.exists(os.path.join(dataset_path, 'GraphAFeatures.txt')):
            Bfeatures = np.matmul(np.matmul(P.T, Afeatures), P)
        else:
            Bfeatures = np.array(range(len(B))).reshape((len(B), 1))
        if Bfeatures.ndim == 1:
            Bfeatures = Bfeatures.reshape(-1, 1)

    return nnodes, nfeatures, nlabels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels, P


def lookup_lod(lod, **kw):
    res = []
    idx = []
    for ind in range(len(lod)):
        row = lod[ind]
        for k, v in kw.iteritems():
            if row[k] != str(v):
                break
        else:
            res.append(row)
            idx.append(ind)
    return res, idx


# save results for each iteration
def save_iteration_results(save_path, suffix, Atrain, Atest, Aembedding, Alabels_output, Alabels_predicted, B, Bfeatures, Bembedding, Blabels, Blabels_output, Blabels_predicted, Ptrue, P, acc_results):
    output_dict = dict(zip(['GraphATrain', 'GraphATest', 'GraphAEmbedding', 'GraphALabels_outputs'], [Atrain, Atest, Aembedding, Alabels_output]))
    if B is not None:
        output_dict.update(dict(zip(['GraphB', 'GraphBFeatures', 'GraphBLabels', 'GraphBEmbedding', 'GraphBLabels_outputs', 'P_true'], [B, Bfeatures, Blabels, Bembedding, Blabels_output, Ptrue])))
        if P is not None:
            output_dict.update(dict(zip(['P_predicted'], [P])))
    
    if Alabels_predicted is not None:
        output_dict.update(dict(zip(['GraphALabels_predicted'], [Alabels_predicted])))
        if B is not None:
            output_dict.update(dict(zip(['GraphBLabels_predicted'], [Blabels_predicted])))
    
    output_dict.update(dict(zip(['Accuracy'], [acc_results])))
    with open(save_path + 'results' + suffix + '.pkl', 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# add results to global results file
def save_global_results(args, iter, best_results, picklename):
    save_path = args.save_path[0:args.save_path.find('results/')+8]
    
    labels = args.labels
    transmode = args.transfer_mode
    
    epochs_mean = np.mean(list(best_results['epochs'].values()))
    epochs_std = np.std(list(best_results['epochs'].values()))
    train_mean = np.mean(list(best_results['train'].values()))
    train_std = np.std(list(best_results['train'].values()))
    testA_mean = np.mean(list(best_results['testA'].values()))
    testA_std = np.std(list(best_results['testA'].values()))
    
    with open(args.save_path + 'Accuracy_(' + str(args.iterations) + ')_' + str(args.epochs) + '.txt', "w") as f:
        if transmode == '1graph':
            if labels == 'cluster':
                f.write("%4d\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (iter, epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std))
            else:
                rsquaredA_mean = np.mean(list(best_results['rsquaredA'].values()))
                rsquaredA_std = np.std(list(best_results['rsquaredA'].values()))
                f.write("%4d\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (iter, epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, rsquaredA_mean, rsquaredA_std))
        else:
            testB_mean = np.mean(list(best_results['testB'].values()))
            testB_std = np.std(list(best_results['testB'].values()))
            if labels == 'cluster':
                f.write("%4d\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (iter, epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, testB_mean, testB_std))
            else:
                rsquaredA_mean = np.mean(list(best_results['rsquaredA'].values()))
                rsquaredA_std = np.std(list(best_results['rsquaredA'].values()))
                rsquaredB_mean = np.mean(list(best_results['rsquaredB'].values()))
                rsquaredB_std = np.std(list(best_results['rsquaredB'].values()))
                f.write("%4d\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\t%.4f (%.4f)\n" % (iter, epochs_mean, epochs_std, train_mean, train_std, testA_mean, testA_std, rsquaredA_mean, rsquaredA_std, testB_mean, testB_std, rsquaredB_mean, rsquaredB_std))
    
    new_res = {'Dataset':args.dataset, 'Cliq':args.ncliq if args.dataset=='synthetic' else 'N/A', 'Labels':args.labels, 'Features':args.features, 'Embedding':args.embedding_type, 'TopSim':args.topology_similarity, 'EmbSim':args.embedding_similarity, 'SimLoss':args.similarity_loss, 'TransMode':transmode, 'GraphDist':args.graph_distance, 'Iterations':iter, 'Epochs_mean':epochs_mean, 'Epochs_std':epochs_std, 'TrainAcc_mean':train_mean, 'TrainAcc_std':train_std, 'TestAccA_mean':testA_mean, 'TestAccA_std':testA_std, 'RsquaredA_mean':rsquaredA_mean if labels!='cluster' else 'N/A', 'RsquaredA_std':rsquaredA_std if labels!='cluster' else 'N/A', 'TestAccB_mean':testB_mean if transmode!='1graph' else 'N/A', 'TestAccB_std':testB_std if transmode!='1graph' else 'N/A', 'RsquaredB_mean':rsquaredB_mean if (labels!='cluster' and transmode!='1graph') else 'N/A', 'RsquaredB_std':rsquaredB_std if (labels!='cluster' and transmode!='1graph') else 'N/A'}
    try:
        with open(save_path + picklename + '.pkl', 'rb') as handle:
            results = pickle.load(handle)
        _, idx = lookup_lod(results, Dataset=args.dataset, Cliq=args.ncliq, Labels=args.labels, Features=args.features, Embedding=args.embedding_type, TopSim=args.topology_similarity, EmbSim=args.embedding_similarity, SimLoss=args.similarity_loss, TransMode=transmode, GraphDist=args.graph_distance)
        if len(idx)==0:
            results.append(new_res)
        else:
            results[idx[0]] = new_res
        print("SAVING: Update an existing record")
    except:
        results = [new_res]
        print("SAVING: Create a new record")
    with open(save_path + picklename + '.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
