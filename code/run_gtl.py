"""
Main module to run GTL model. Depending on the input parameters,
it loads certain data, creates and trains a specific GTL model

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import argparse
import os
import numpy as np
from pathlib import Path

from load_save_data import load_data
from create_gtl_model import getGTLmodel
from train_gtl_model import train
from load_save_data import save_global_results


def str2bool(v):
    if v.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train Graph Transfer Learning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # dataset parameters
    parser.add_argument('-lp', '--load_path',
                        type=str, default='../datasets/',
                        help='Full path to folder with datasets')
    parser.add_argument('-d', '--dataset',
                        type=str, default='sb-4', choices=['bp-2', 'sb-4', 'sb-6', 'zachary', 'disease', 'email'],
                        help='Dataset to be used')
    # general graph parameters
    parser.add_argument('--labels',
                        type=str, default='cluster', choices=['cluster', 'infection'],
                        help='Node labels of the synthetic data')
    # graph embedding parameters
    parser.add_argument('--nembedding',
                        type=int, default=5,
                        help='Size of the output embedding vector')
    parser.add_argument('-sg', '--topology_similarity',
                        type=str, default='randomwalk', choices=['randomwalk', 'adjacency'],
                        help='Similarity measure between nodes of the same graph in graph topological space')
    parser.add_argument('-et', '--embedding_type',
                        type=str, default='skipgram', choices=['unified', 'skipgram'],
                        help='Type of embedding function: skipgram, unified')
    parser.add_argument('-se', '--embedding_similarity',
                        type=str, default='softmax', choices=['softmax', 'innerprod', 'cossim', 'l2'],
                        help='Similarity measures between nodes of the same graph in embedding space')
    parser.add_argument('-sl', '--similarity_loss',
                        type=str, default='crossentropy', choices=['crossentropy', 'innerprod', 'l2'],
                        help='Loss function between similarities in topological and embedding '
                             'spaces for nodes of the same graph')
    # prediction branch parameters
    parser.add_argument('--depth',
                        type=int, default=1,
                        help='Number of hidden layers in Prediction Branch')
    parser.add_argument('-af', '--activation_function',
                        type=str, default='tanh', choices=['tanh', 'sigmoid', 'relu'],
                        help='Activation function for Prediction Branch neurons')
    parser.add_argument('-prl', '--prediction_loss',
                        type=str, default='mean_squared_error',
                        choices=['mean_squared_error', 'mean_absolute_percentage_error'],
                        help='Loss function for Prediction Branch')
    # randomwalk parameters
    parser.add_argument('--nwalks',
                        type=int, default=20,
                        help='Number of node2vec random walks')
    parser.add_argument('--walk_length',
                        type=int, default=10,
                        help='Length of random walk')
    parser.add_argument('--window_size',
                        type=int, default=4,
                        help='Width of sliding window in random walks')
    parser.add_argument('--p',
                        type=float, default=0.25,
                        help='Parameter p for node2vec random walks')
    parser.add_argument('--q',
                        type=float, default=4.0,
                        help='Parameter q for node2vec random walks')
    parser.add_argument('--nnegative',
                        type=int, default=5,
                        help='Number of negative samples used in skip-gram')
    parser.add_argument('--scale_negative',
                        type=str2bool, default=False,
                        help='Specifies whether to scale outputs for negative samples')
    # second graph parameters
    parser.add_argument('--transfer_mode',
                        type=str, default='1graph', choices=['1graph', 'noP', 'iterP', 'optP', 'trueP', 'trueP_DS'],
                        help='Specifies transfer learning mode')
    parser.add_argument('--b_from_a',
                        type=str, default='permute', choices=['permute', 'modify'],
                        help='Specifies whether to permute or add/remove edges to graph A to generate graph B')
    parser.add_argument('-dp', '--discrepancy_percent',
                        type=float, default=0,
                        help='Specifies percentage of edges to be removed/added when generating second graph')
    parser.add_argument('-gd', '--graph_distance',
                        type=str, default='l2', choices=['l2', 'innerprod', 'cossim'],
                        help='Pairwise distance measure between nodes in the embedding space (matrix D)')
    # neural net train/test parameters
    parser.add_argument('--alpha',
                        type=float, default=1.0,
                        help='Weight of graph matching loss')
    parser.add_argument('--beta',
                        type=str2bool, default=False,
                        help='Specifies whether to scale parts of P-optimization loss')
    parser.add_argument('-lr', '--learning_rate',
                        type=float, default=0.025,
                        help='Learning rate')
    parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='Number of instances in each batch')
    parser.add_argument('--epochs',
                        type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--early_stopping',
                        type=int, default=0,
                        help='Number of epochs with no improvement after which training will be stopped. '
                             'If <=0, no early stopping is used')
    parser.add_argument('--iterations',
                        type=int, default=1,
                        help='Number of iterations for model to initialize and run. '
                             'Output results are averaged across iterations')
    # CUDA parameters
    parser.add_argument('--id_gpu',
                        default=-1, type=int,
                        help='Specifies which gpu to use. If <0, model is run on cpu')
    # results parameters
    parser.add_argument('-sp', '--save_path', type=str, default='',
                        help='Full path to folder where results are saved')
    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
    
    # save configuration file
    n_iter = max(1, args.iterations)
    save_path = args.save_path
    print("*************** Configuration ***************")
    args_dic = vars(args)
    for arg, value in args_dic.items():
        line = arg + ' : ' + str(value)
        print(line)
    print("*********************************************\n")
    
    # load data
    dataset_path = str(Path(args.load_path) / args.dataset.lower())
    labels = args.labels.lower()
    transmode = args.transfer_mode
    B_from_A = args.b_from_a
    disc_pect = args.discrepancy_percent
    
    n_layers = max(10, args.depth)

    n_embedding = args.nembedding
    topology_similarity = args.topology_similarity
    embedding_type = args.embedding_type
    embedding_similarity = args.embedding_similarity
    if embedding_type == 'skipgram':
        embedding_similarity = 'softmax'
    if embedding_similarity == 'softmax':
        n_negative = args.nnegative
        scale_negative = args.scale_negative
    else:
        n_negative = 0
        scale_negative = False
    similarity_loss = args.similarity_loss
    prediction_loss = args.prediction_loss
    activation_function = args.activation_function
    
    graph_distance = args.graph_distance
    n_walks = args.nwalks
    walk_length = args.walk_length
    window_size = args.window_size
    p = args.p
    q = args.q
    
    alpha = args.alpha
    beta = args.beta
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    early_stopping = args.early_stopping
    
    n_epochs = args.epochs
    results = {'epochs': {}, 'train': {}, 'testA': {}}
    if labels != 'cluster':
        results['rsquaredA'] = {}
    if transmode != '1graph':
        results['testB'] = {}
        if labels != 'cluster':
            results['rsquaredB'] = {}
    for iter in range(n_iter):
        save_path_iter = save_path + str(iter+1) + '/'
        if not os.path.exists(save_path_iter):
            os.makedirs(save_path_iter)
        # load data
        n_nodes, n_features, n_labels, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels, Ptrue = \
            load_data(dataset_path, labels, transmode, B_from_A, disc_pect)
        # create model
        models = getGTLmodel(n_nodes, n_features, n_embedding, labels, n_labels, n_layers, n_negative, scale_negative,
                             embedding_type, embedding_similarity, similarity_loss, prediction_loss,
                             activation_function, transmode, graph_distance, learning_rate, alpha, save_path)
        
        if iter == 0:
            print("\nEmbedding model summary:".upper())
            print(models['EmbeddingModel'].summary())
            print("\nEmbedding similarity branch summary:".upper())
            print(models['EmbeddingModel'].get_layer('Branch_SimilarityA').summary())
            print("\nPrediction model summary:".upper())
            print(models['PredictionModel'].summary())
            print("\nPrediction branch summary:".upper())
            print(models['PredictionModel'].get_layer('Branch_Prediction').summary())
        
        # train/test model
        print("\n ============================================== ")
        print("|*************** ITERATION #{:3d} ***************|".format(iter+1) if n_iter > 1 else
              "|************ GRAPH TRANSFER LEARNING *********|")
        print(" ============================================== ")
        iter_results = train(models, A, Afeatures, Alabels, Atrain, Atest, B, Bfeatures, Blabels, Ptrue, transmode,
                             topology_similarity, n_walks, walk_length, window_size, p, q, n_negative, learning_rate,
                             beta, n_epochs, early_stopping, batch_size, save_path_iter)
        results['epochs'][iter] = iter_results['epochs']
        results['train'][iter] = iter_results['acc_train']
        results['testA'][iter] = iter_results['acc_testA']
        if labels != 'cluster':
            results['rsquaredA'][iter] = iter_results['acc_rsquaredA']
        if transmode != '1graph':
            results['testB'][iter] = iter_results['acc_testB']
            if labels != 'cluster':
                results['rsquaredB'][iter] = iter_results['acc_rsquaredB']
    
        # save global results
        picklename = "GlobalResults"
        save_global_results(args, iter + 1, results, picklename)
    
    epochs_mean = np.mean(list(results['epochs'].values()))
    epochs_std = np.std(list(results['epochs'].values()))
    train_mean = np.mean(list(results['train'].values()))
    train_std = np.std(list(results['train'].values()))
    testA_mean = np.mean(list(results['testA'].values()))
    testA_std = np.std(list(results['testA'].values()))
    res_str = ""
    if labels != 'cluster':
        rsquaredA_mean = np.mean(list(results['rsquaredA'].values()))
        rsquaredA_std = np.std(list(results['rsquaredA'].values()))
        res_str += "\tR-squared (graph A) = {0:.4f} (\u00B1{1:.4f})\n".format(rsquaredA_mean, rsquaredA_std)
    if transmode != '1graph':
        testB_mean = np.mean(list(results['testB'].values()))
        testB_std = np.std(list(results['testB'].values()))
        res_str += "\tTest accuracy (graph B) = {0:.4f} (\u00B1{1:.4f})\n".format(testB_mean, testB_std)
        if labels != 'cluster':
            rsquaredB_mean = np.mean(list(results['rsquaredB'].values()))
            rsquaredB_std = np.std(list(results['rsquaredB'].values()))
            res_str += "\tR-squared (graph B) = {0:.4f} (\u00B1{1:.4f})\n".format(rsquaredB_mean, rsquaredB_std)
    print("\n\n ============================================== ")
    print("|*************** FINAL  RESULTS ***************|")
    print(" ============================================== ")
    print(f"After {n_iter:2d} iteration(s), the average\n"
          f"\tConvergence rate = {epochs_mean:.4f} (\u00B1{epochs_std:.4f})\n"
          f"\tTrain accuracy (graph A) = {train_mean:.4f} (\u00B1{train_std:.4f})\n"
          f"\tTest accuracy (graph A) = {testA_mean:.4f} (\u00B1{testA_std:.4f})")
    print(res_str)
