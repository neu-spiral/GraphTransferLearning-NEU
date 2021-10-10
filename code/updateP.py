"""
Module is responsible for permutation matrix P optimization

@author: Yuan Guo
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def updateP(A, B, D, P=None, mode='opt'):
    """
    Wrapper for updating P.
    
    Inputs:
        A Adjacency matrix of the first graph
        B Adjacency matrix of the second graph
        D Matrix of pairwise distances between nodes of two graphs in the ebedding space
        P Original value of matrix P (used only in iterative methods)
        MODE Algorithm used to compute new value of matrix P
        
    Outputs:
        NEWP New value of matrix P
    """
    
    mode = mode.lower()
    if mode.find('iter')>=0 and P is not None: # iterative P solution (projected gradient descent)
        gam = 0.001
        newP = ADMM_update(A,B,D,P,gam,norm=2,mode='norm')
    else: # optimal P solution
        prec = 100
        newP = Frank_optimal(A,B,D,prec,norm=2,mode='norm')
    
    return newP


def ADMM_update(A,B,D,P,gam,norm=2,mode='norm'):
    """
    projection of the ||AP-PB||_{p}+tr(P^{T}D)
    with gamma
    """
    dep = Derivative(A,B,D,P,norm,mode)
    V = P - gam*dep
    pnew = ADMMV(V)
    return pnew


def Derivative(A,B,D,P,order,mode):
    C = np.dot(A,P) - np.dot(P,B) 
    if mode == 'square':       
        grad_P = 2*(np.dot(A.T,C) - np.dot(C,B.T))
    else:
        Corder = np.sum(C**order)
        if Corder <= 10**(-9):
            cp=0
        else:
            cp = Corder**(1./order-1)
        Cp1 = C**(order-1)
        grad_P = cp*(np.dot(A.T,Cp1) - np.dot(Cp1,B.T))
    grad_P += D
    return grad_P


def ADMMV(V,rho=1,epsilon=0.001,itr_num=10000):
    obj = ADMM_OBJ(V)
    Pnew = obj.loop(rho,epsilon,itr_num)
    return Pnew


class ADMM_OBJ:
    """
    The ADMM class to solve:
    solve min ||P-V||_{2}^{2}
    P1=P^{T}1=0
    P>=0
    """
    def __init__(self,V):
        self.V = V
        self.X = {}
        self.Y = {}
        self.Z = np.zeros(V.shape)
    
    def loop(self,rho,epsilon,itr_num):
        self.Y['P'] = np.zeros(self.V.shape)
        self.Y['row'] = np.zeros(self.V.shape)
        self.Y['column'] = np.zeros(self.V.shape)
    
        dual = np.inf
        primal = np.inf
        itr = 0
        while ((dual >= epsilon) | (primal >= epsilon)):
            #t1=time.time()
            Z_old = self.Z.copy()
            self.X['P'] = (2*self.V + rho*self.Z - self.Y['P']) / (2+rho)
            self.X['row'] = MapZero(self.Z, self.Y['row'], rho, 'row')
            self.X['column'] = MapZero(self.Z, self.Y['column'], rho, 'column')
            self.Z = np.mean(np.array(list(self.X.values())),0)
            primal = 0
            for key in self.Y:
                y_d = self.X[key] - self.Z
                self.Y[key] = self.Y[key] + rho*y_d
                primal += np.linalg.norm(y_d)**2
            primal = np.sqrt(primal)
            dual = np.sqrt(3) * rho * np.linalg.norm(self.Z-Z_old) 
            itr += 1
            if itr >= itr_num:
                break
            else:
                pass
        return self.Z


def MapZero(Z,Y,rho,mode):
    V = Z - Y/rho
    Xtrace = V.copy()
    if mode == 'column':
        for i in range(V.shape[1]):
            Xtrace[:,i] = VecSimplex(V[:,i],1)
    else:
        for i in range(V.shape[0]):
            Xtrace[i,:] = VecSimplex(V[i,:],1)
    return Xtrace


def projectToVA(x,A,r):
    Ac = set(x.keys()).difference(set(A))
    offset = 1.0 / (len(Ac))*(sum([x[i] for i in Ac]) - r)
    y = dict([(i,0.0) for i in A] + [(i,x[i] - offset) for i in Ac])
    return y


def projectToPositiveSimplex(x,r):
    """
    A function that projects a vector x to the face of the positive simplex.
    Given x as input, where x is a dictionary, and a r>0,  the algorithm returns a dictionary y with the same keys as x such that:
        (1) sum( [ y[key] for key in y] ) == r,
        (2) y[key]>=0 for all key in y
        (3) y is the closest vector to x in the l2 norm that satifies (1) and (2)
        The algorithm terminates in at most O(len(x)) steps, and is described in:

             Michelot, Christian. "A finite algorithm for finding the projection of a point onto the canonical simplex of R^n." Journal of Optimization Theory and Applications 50.1 (1986): 195-200iii
        and a short summary can be found in Appendix C of:

         http://www.ece.neu.edu/fac-ece/ioannidis/static/pdf/2010/CR-PRL-2009-07-0001.pdf
    """
    A = []
    y = projectToVA(x,A,r)
    B = [i for i in y.keys() if y[i]<0.0]
    while len(B) > 0:
        A += B
        y = projectToVA(y,A,r)
        B = [i for i in y.keys() if y[i]<0.0]
    return y


def VecSimplex(x,r):
    """
    max ||s-v||_{2}^{2}
    subject to <s,1>=r, s>=0
    """
    size = len(x)
    xd = {}
    for i in range(size):
        xd[i] = x[i]
    y = projectToPositiveSimplex(xd,r)
    return np.array([y[i] for i in range(size)])


"""
Frank-Wolfe Implementation
"""
def V_dot(v):
    """
    the ADMM algorithm to solve:
    min <S, V>
    subject to S>=0, S1=1, 1^{T}S=1^{T}
    """
    row_ind, col_ind = linear_sum_assignment(v)
    b = np.zeros(v.shape)
    b[row_ind,col_ind] = 1
    return b


class Frank_P:
    """
    The Frank Wolfe class 
    To solve ||AP-PB||_{2}^{2}+tr(P^{T}D) using cvxopt
    """
    def __init__(self, A, B, D, P, order, norm_mode, gamma=0.1):
        self.A = A
        self.B = B
        self.D = D
        self.P = P
        self.gamma = gamma
        self.order = order
        self.norm = norm_mode
        
    def initialize(self):
        pass
        
    def first_fun(self):
        nablaP = Derivative(self.A,self.B,self.D,self.P,self.order,self.norm)
        return nablaP  
    
    def iteration(self, epsilon, itr_num):
        itr=0
        while(True):
            nabla_P = self.first_fun()
            S_optimal = V_dot(nabla_P)
            delta_P = S_optimal - self.P  
            eta = 2/(itr+2)
            dual = -np.sum(nabla_P*delta_P)
            P_new = self.P + eta*delta_P
            self.P = P_new.copy()
            #print (np.linalg.norm(np.dot(self.A,self.P)-np.dot(self.P,self.B))**2+np.sum(self.P*self.D))
            itr += 1
            if itr >= itr_num:
                break
            else:
                pass
            if dual <= epsilon:
                break
            else:
                pass
        return P_new


def Frank_optimal(A,B,D,prec=100,norm=2,mode='norm'):
    """
    Frank-Wolfe method to solve:
    ||AP-PB||_{2}^{2}+tr(P^{T}D)
    P1=P^T1=1
    P>=0
    """
    P = np.identity(A.shape[0])
    obj = Frank_P(A,B,D,P,norm,mode)
    pnew = obj.iteration(0.001,prec)
    return pnew
