"""
Main GTL module responsible for creating, training and testing GTL NN

@author: Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import numpy as np

from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.merge import Dot, Concatenate
from keras import backend as K
from keras import optimizers


# l2 distance between two network layers
def l2_dist(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
# shape of l2 distance between two embeddings in similarity branch
def l2_output_shape_sim(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1, 1)
# shape of l2 distance between two embeddings in distance branch
def l2_output_shape_dist(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def createNetworks(n_nodes, input_size, labels, n_layers, embedding_type, embedding_similarity, transmode, graph_distance, scale_negative, activation_function):
    """
    Creates neural network that learns node embeddings of given graph(s)
    
    Inputs:
        INPUT_SIZE List [n,m,k,l] where:
            N Number of samples, 
            M Number of original features (equal to n for one-hot coding)
            K Number of embedding features
            L Number of node labels
        EMBEDDING_TYPE Type of embedding approach, e.g. 'unified' (unified embedding for target and context nodes) or 'skipgram' (different embeddings for target and context nodes)
        EMBEDDING_SIMILARITY Measure of similarity between node embeddings within one graph
        TRANSMODE Flag to specify transfer learning mode
        GRAPH_DISTANCE Distance between node embeddings of different graphs
        
    Outputs: 
        Neural network for graph node embeddings
    """
    
    dict_sizeA, dict_sizeB = n_nodes if type(n_nodes) in [list, tuple] else [n_nodes]*2
    feature_size, embedding_size, class_size, negative_size = input_size
    inputsEmbedding = []
    inputsEmbeddingA = []
    inputsEmbeddingB = []
    outputsEmbedding = []
    inputsPrediction = []
    outputsPrediction = []
    
    if embedding_similarity == 'l2':
        from keras.constraints import UnitNorm
        constraints = UnitNorm(axis=1)
    else:
        constraints = None
    
    # create embedding branch for graph A
    if feature_size == 1:
        input_shape = (1,)
        input_type = 'int32'
        Embedding_targetA = Embedding(dict_sizeA, embedding_size, embeddings_constraint=constraints, 
                                      name='Embedding_TargetA')
        Embedding_contextA = Embedding(dict_sizeA, embedding_size, embeddings_constraint=constraints, 
                                       name='Embedding_ContextA')
    else:
        input_shape = (1, feature_size,)
        input_type = 'float'
        Embedding_targetA = Dense(embedding_size, activation='tanh', kernel_constraint=constraints, 
                                  name='Embedding_TargetA')
        Embedding_contextA = Dense(embedding_size, activation='tanh', kernel_constraint=constraints, 
                                   name='Embedding_ContextA')
        
    input_targetA = Input(shape=input_shape, dtype=input_type, 
                          name='Input_TargetA')
    input_contextA = Input(shape=input_shape, dtype=input_type, 
                           name='Input_ContextA')
    inputsEmbeddingA.extend([input_targetA, input_contextA])
    
    # initialize graph A embedding weights from multivariate gaussian distribution
    embedding_targetA = Embedding_targetA(input_targetA)
    Embedding_targetA.set_weights([np.random.multivariate_normal(np.zeros(embedding_size), 0.1*np.identity(embedding_size), dict_sizeA)])
    if embedding_type == 'skipgram':
        embedding_contextA = Embedding_contextA(input_contextA)
        Embedding_contextA.set_weights([np.random.multivariate_normal(np.zeros(embedding_size), 0.1*np.identity(embedding_size), dict_sizeA)])
    elif embedding_type == 'unified':
        embedding_contextA = Embedding_targetA(input_contextA)
    
    # add more dense layers to embedding branch if predicting pagerank
    if labels == 'pagerank':
        embedding_targetA = Dense(embedding_size, activation='tanh')(embedding_targetA)
        embedding_contextA = Dense(embedding_size, activation='tanh')(embedding_contextA)
    
    # create similarity branch for graph A
    inputsSimilarityA = [embedding_targetA, embedding_contextA]
    if embedding_similarity == 'softmax':
        # add negative samples
        input_negativeA = Input(shape=(negative_size,)+input_shape[1:], dtype=input_type, name='Input_NegativeA')
        inputsEmbeddingA.extend([input_negativeA])
        embedding_negativeA = Embedding_targetA(input_negativeA)
        # add more dense layers to embedding branch if predicting pagerank
        if labels == 'pagerank':
            embedding_negativeA = Dense(embedding_size, activation='tanh')(embedding_negativeA)
        inputsSimilarityA.extend([embedding_negativeA])
    similarityA = createSimilarityBranch(embedding_size, mode=embedding_similarity, negative_size=negative_size, graph='A', scale_negative=scale_negative)(inputsSimilarityA)
    outputsEmbedding.extend([similarityA])
    inputsEmbedding.extend(inputsEmbeddingA)
    
    # create prediction branch
    inputsPrediction.extend([input_targetA])
    predictionBranch = createPredictionBranch(embedding_size, n_layers, class_size, activation_function)(embedding_targetA)
    predictionOutput = Reshape((class_size,), name='PredictionOutput')(predictionBranch)
    outputsPrediction.extend([predictionOutput])
    
    if transmode != '1graph':
        input_targetB = Input(shape=input_shape, dtype=input_type, name='Input_TargetB')
        input_contextB = Input(shape=input_shape, dtype=input_type, name='Input_ContextB')
        inputsEmbeddingB.extend([input_targetB, input_contextB])
        
        # create embedding branch for graph B
        if feature_size == 1:
            Embedding_targetB = Embedding(dict_sizeB, embedding_size, embeddings_constraint=constraints, name='Embedding_TargetB')
            Embedding_contextB = Embedding(dict_sizeB, embedding_size, embeddings_constraint=constraints, name='Embedding_ContextB')
        else:
            Embedding_targetB = Dense(embedding_size, activation='tanh', kernel_constraint=constraints, name='Embedding_TargetB')
            Embedding_contextB = Dense(embedding_size, activation='tanh', kernel_constraint=constraints, name='Embedding_ContextB')
        
        # initialize graph B embedding weights as zeros when graph embeddings are linked, or from multivariate gaussian distribution, otherwise
        embedding_targetB = Embedding_targetB(input_targetB)
        Embedding_targetB.set_weights([np.random.multivariate_normal(np.zeros(embedding_size), 0.1*np.identity(embedding_size), dict_sizeB)])
        if embedding_type == 'skipgram': # separate embeddings for target and context nodes
            embedding_contextB = Embedding_contextB(input_contextB)
            Embedding_contextB.set_weights([np.random.multivariate_normal(np.zeros(embedding_size), 0.1*np.identity(embedding_size), dict_sizeB)])
        elif embedding_type == 'unified': # unified embedding
            embedding_contextB = Embedding_targetB(input_contextB)
    
        # add more dense layers to embedding branch if predicting pagerank
        if labels == 'pagerank':
            embedding_targetB = Dense(embedding_size, activation='tanh')(embedding_targetB)
            embedding_contextB = Dense(embedding_size, activation='tanh')(embedding_contextB)
        
        # create similarity branch for graph B
        inputsSimilarityB = [embedding_targetB, embedding_contextB]
        if embedding_similarity == 'softmax':
            # add negative samples
            input_negativeB = Input(shape=(negative_size,)+input_shape[1:], dtype=input_type, name='Input_NegativeB')
            inputsEmbeddingB.extend([input_negativeB])
            embedding_negativeB = Embedding_targetB(input_negativeB)
            # add more dense layers to embedding branch if predicting pagerank
            if labels == 'pagerank':
                embedding_negativeB = Dense(embedding_size, activation='tanh')(embedding_negativeB)
            inputsSimilarityB.extend([embedding_negativeB])
        similarityB = createSimilarityBranch(embedding_size, mode=embedding_similarity, negative_size=negative_size, graph='B', scale_negative=scale_negative)(inputsSimilarityB)
        outputsEmbedding.extend([similarityB])
        inputsEmbedding.extend(inputsEmbeddingB)
        
        # create graph distance branch
        if transmode != 'noP':
            distanceAB = createDistanceBranch(embedding_size, mode=graph_distance)([embedding_targetA, embedding_targetB])
            outputsEmbedding.extend([distanceAB])
        
    modelEmbedding = Model(inputs=inputsEmbedding, outputs=outputsEmbedding)
    branchEmbeddingA = Model(inputs=inputsEmbeddingA, outputs=outputsEmbedding[0])
    branchEmbeddingB = Model(inputs=inputsEmbeddingB, outputs=outputsEmbedding[1]) if transmode != '1graph' else None
    modelPrediction = Model(inputs=inputsPrediction, outputs=outputsPrediction)

    return modelEmbedding, branchEmbeddingA, branchEmbeddingB, modelPrediction


def createSimilarityBranch(embedding_size, mode='innerprod', negative_size=0, graph='A', scale_negative=False):
    """
    Branch of global network: computes similarity between embeddings of given two nodes in a graph
    """
    inputT = Input(shape=(1,embedding_size,), name='Embedding_Target')
    inputC = Input(shape=(1,embedding_size,), name='Embedding_Context')
    inputs = [inputT, inputC]
    
    layer_name = 'Output_Similarity'
    if mode=='l2': # l2 distance
        similarity =  Lambda(l2_dist, output_shape=l2_output_shape_sim, name=layer_name)(inputs)
    elif mode=='cossim': # cosine similarity
        similarity = Dot(axes=-1, normalize=True, name=layer_name)(inputs)
    elif mode=='innerprod': # inner product
        similarity = Dot(axes=-1, name=layer_name)(inputs)
    else: # softmax (default)
        inputNS = Input(shape=(negative_size,embedding_size,), name='Embedding_NS')
        inputs.append(inputNS)
        similarityTC = Dot(axes=-1)([inputT, inputC])
        similarityCNS = Dot(axes=-1)([inputC, inputNS])
        similarity = Concatenate(axis=-1)([similarityTC, similarityCNS])
        similarity = Activation('softmax', name=layer_name)(similarity)
    
    # normalize negative samples loss
    if scale_negative:
        def normalizeNS(x):
            return x/negative_size
        similarity = Activation(normalizeNS)(similarity)
    
    return Model(inputs, similarity, name='Branch_Similarity'+graph)


def createPredictionBranch(embedding_size, n_layers, class_size, activation):
    """
    Branch of global network: predicts node label for a given embedding
    """
    input = Input(shape=(embedding_size,), name='Input_Embedding')

    # intermediate layer(s) structure
    prediction = Dense(embedding_size, activation=activation)(input)
    for _ in range(n_layers-1):
        prediction = Dense(embedding_size, activation=activation)(prediction)
    # last layer structure
    layer_name = 'Output_Label'
    if class_size > 1:
        # classification task case
        prediction = Dense(class_size, activation='softmax', name=layer_name)(prediction)
    else: 
        # regression task case
        prediction = Dense(class_size, activation='sigmoid', name=layer_name)(prediction)
    return Model(input, prediction, name='Branch_Prediction')


def createDistanceBranch(embedding_size, mode='l2'):
    """
    Branch of global network: computes all pairwise distances between node embeddings of different graphs
    """
    input1 = Input(shape=(embedding_size,), name='Input_EmbeddingA')
    input2 = Input(shape=(embedding_size,), name='Input_EmbeddingB')
    
    layer_name = 'Output_DistanceAB'
    if mode == 'innerprod':
        distance12 = Dot(axes=-1, name=layer_name)([input1, input2])
    elif mode == 'cossim':
        distance12 = Dot(axes=-1, normalize=True, name=layer_name)([input1, input2])
    else:
        distance12 = Lambda(l2_dist, output_shape=l2_output_shape_dist, name=layer_name)([input1, input2])
    return Model([input1, input2], distance12, name='Branch_Distance')


# generic function that specifies loss function to be used 
def getLoss(loss='crossentropy'):
    if loss == 'l2':
        return L2Loss
    elif loss == 'innerprod':
        return InnerProdLoss
    else:
        return 'binary_crossentropy'  # multiple true labels are possible in output vector
#        return 'categorical_crossentropy' # only one true label is possible in output vector


# outputs inner product loss between true and predicted labels
def InnerProdLoss(yTrue, yPred):
    return K.sum(yTrue * yPred)


# outputs l2 loss between true and predicted labels
def L2Loss(yTrue, yPred):
    return K.sqrt(K.sum(K.square(yTrue - yPred)))


# root mean squared error
def RMSELoss(yTrue, yPred):
    return K.sqrt(K.mean(K.square(yTrue - yPred)))


def getGTLmodel(n_nodes, n_features, n_embedding, labels, n_labels, n_layers, 
                n_negative, scale_negative, embedding_type, embedding_similarity, similarity_loss, 
                prediction_loss, activation_function, 
                transmode, graph_distance, learning_rate, alpha, save_path):
    
    input_size = (n_features, n_embedding, n_labels, n_negative)
    
    Optimizer = optimizers.SGD(lr=learning_rate, nesterov=True)
    
    print("\n*************** Creating Model ***************".upper())
    EmbeddingModel, EmbeddingABranch, EmbeddingBBranch, PredictionModel = \
        createNetworks(n_nodes, input_size, labels, n_layers, 
                       embedding_type, embedding_similarity, transmode, graph_distance, scale_negative, activation_function)
    
    # Neural Net learning node embeddings
    EmbLoss = {"Branch_SimilarityA": getLoss(similarity_loss)}
    loss_weights = [1]
    if transmode != '1graph':
        EmbLoss.update({"Branch_SimilarityB": getLoss(similarity_loss)})
        loss_weights.append(1)
        if transmode != 'noP':
            EmbLoss.update({"Branch_Distance": InnerProdLoss})
            loss_weights.append(alpha)
    EmbeddingModel.compile(optimizer=Optimizer, loss=EmbLoss, loss_weights=loss_weights)
    with open(save_path+"EmbeddingModel.json", "w") as json_file:
        json_file.write(EmbeddingModel.to_json())
    
    # Branch of neural net learning node embeddings only for graph A
    EmbeddingABranch.compile(optimizer=Optimizer, loss=getLoss(similarity_loss))
    with open(save_path+"EmbeddingABranch.json", "w") as json_file:
        json_file.write(EmbeddingABranch.to_json())
    
    if transmode != '1graph':
        # Branch of neural net learning node embeddings only for graph B
        EmbeddingBBranch.compile(optimizer=Optimizer, loss=getLoss(similarity_loss))
        with open(save_path+"EmbeddingBBranch.json", "w") as json_file:
            json_file.write(EmbeddingBBranch.to_json())
    
    # Neural Net predicting labels
    if n_labels > 1:
        prediction_loss = "categorical_crossentropy"
    elif prediction_loss is None:
        prediction_loss = "binary_crossentropy"
    elif prediction_loss == 'root_mean_squared_error':
        prediction_loss = RMSELoss
    PredictionModel.compile(optimizer=Optimizer, loss=prediction_loss)
        
    with open(save_path+"PredictionModel.json", "w") as json_file:
        json_file.write(PredictionModel.to_json())
    with open(save_path+"PredictionBranch.json", "w") as json_file:
        json_file.write(PredictionModel.get_layer('Branch_Prediction').to_json())
    
    model_names = ['EmbeddingModel', 'EmbeddingABranch', 'EmbeddingBBranch', 'PredictionModel']
    models = dict(zip(model_names, [EmbeddingModel, EmbeddingABranch, EmbeddingBBranch, PredictionModel]))
    
    return models
