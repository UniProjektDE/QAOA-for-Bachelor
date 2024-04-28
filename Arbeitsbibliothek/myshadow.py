import itertools
from scipy.optimize import minimize
import numpy as np
import pprint
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

def isequallist_helper(a1, a2):
    if np.isscalar(a1) and np.isscalar(a2):
        return a1 == a2
    res = True
    for i, j in zip(a1,a2):
        res = res and isequallist_helper(i,j)
    return res

# ShadowCollection > shadowList > ShadowShot > shadow
class ShadowCollection:
    shadowList = []
    graph_weights = None
    def __init__(self, shadows, weights):
        self.shadowList = shadows
        self.graph_weights = weights
    
    def get_energy_list(self):
        E_list = []
        shadowList = copy.deepcopy(self.shadowList) # strenge Kopie, da sonst diese Funktion den Shadow ändern kann (siehe Bitstring Konvention)
        for shadow in shadowList:
            E_list.append(shadow.get_energy_from_CodeExample(self.graph_weights))
        return E_list
    
    def get_mean_list(self):
        list = []
        for shadow in self.shadowList:
            list.append(shadow.get_mean())
        return list
    
    def get_fidelity_list(self, grundzustand):
        li = []
        for s in self.shadowList:
            li.append(s.get_fidelity(grundzustand))
        return li
    
    # gibt aufgeteilte ShadowCollection als Tuple zurück
    def split_shadow_list(self, percentage):
        list1 = []
        list2 = []
        i = 1
        how_many_in_first_list = len(self.shadowList) * percentage
        for item in self.shadowList:
            if i <= how_many_in_first_list:
                list1.append(copy.deepcopy(item))
            else:
                list2.append(copy.deepcopy(item))
            i += 1
        collect1 = ShadowCollection(list1, self.graph_weights)
        collect2 = ShadowCollection(list2, self.graph_weights)
        return collect1, collect2
    
    def eliminated_equivalent(self):
        for s in self.shadowList:
            s.eliminated_equivalent()
    
    def __add__(self, other):
        # bei sum Funktion wird am Anfang mit Null addiert; wird hier durch leere ShadowCollection ersetzt
        if np.isscalar(self):
            self = ShadowCollection([], other.graph_weights)
        elif np.isscalar(other):
            other = ShadowCollection([], self.graph_weights)

        shadowList = self.shadowList + other.shadowList
        if not np.array_equal(self.graph_weights, other.graph_weights):
            raise ValueError('Shadows haben nicht denselben Graphen!!')
        return ShadowCollection(shadowList, self.graph_weights)
    
    def __radd__(self, other):
        return self + other
    
    '''
    def data_nn_helper(self):
        inputs = [i.parameters for i in self.shadowList]
        labels = [i.shadow for i in self.shadowList]
        return inputs, labels

    def data_nn(self):
        inp, lab = self.data_nn_helper()
        inputs = torch.tensor(inp)
        labels = torch.tensor(lab)
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset)
    '''

class ShadowShot:
    shadow = []
    parameters = []
    dim = 0 # Anzahl der Layers von QAOA
    mean_shadow = []
    mean_calculated = False
    minusoneandone_convention = False

    def __init__(self, shadow, param, dim = 0):
        self.shadow = shadow
        self.parameters = param
        self.dim = dim

    def count_unequals(self):
        k = self.shadow
        k.sort()
        a = list(k for k,_ in itertools.groupby(k))
        return len(a)
    
    def count_equals(self):
        k = self.shadow
        k.sort()
        a = list(k for k,_ in itertools.groupby(k))
        counter = np.zeros(len(a))
        for lis in self.shadow:
            for i in range(len(a)):
                if (a[i] == lis):
                    counter[i] += 1
                    break
        more_than_one = 0
        for co in counter:
            if co > 1: more_than_one +=1
        index = np.argmax(counter)
        return a[index], counter[index], more_than_one, counter
    
    def hit_by(O,P):
        """ Returns whether o is hit by p """
        for o,p in zip(O,P):
            if not (o==0 or p==0 or o==p):
                return False
        return True
    
    # convention: 0 -> 1 und 1 -> -1
    def makeZeroToMinusOne(self):
        self.minusoneandone_convention = True
        for i in range(len(self.shadow)):
            for j in range(len(self.shadow[0])):
                if self.shadow[i][j] == 0:
                    self.shadow[i][j] = 1
                else:
                    self.shadow[i][j] = -1
    
    def reverse_makeZeroToMinusOne(self):
        self.minusoneandone_convention = False
        for i in range(len(self.shadow)):
            for j in range(len(self.shadow[0])):
                if self.shadow[i][j] == 1:
                    self.shadow[i][j] = 0
                else:
                    self.shadow[i][j] = 1

    def eliminated_equivalent(self):
        newShadow = []
        for s in self.shadow:
            if s[0] != 1:
                newShadow.append(s)
            else:
                new_s = [0 if i==1 else 1 for i in s]
                newShadow.append(new_s)
        self.shadow = newShadow

    def get_fidelity(self, grundzustand):
        sum = 0
        for s in self.shadow:
            for state in grundzustand:
                l1 = np.array(state)
                l2 = np.array(s)
                if np.array_equal(l1,l2):
                    sum += 1
        return sum / len(self.shadow)

    
    # weights is adjacency matrix of graph
    def get_energy_qaoa(self, weights, offset=0):
        outcomes = np.array(self.shadow)
        energy = 0
        for outcome in outcomes:
            for i in range(len(weights)):
                energy += np.dot(weights[i], outcome)
        energy /= len(outcomes) * len(weights)  # average over all outcomes
        return energy + offset
    
    # weights ist weight Matrix; Energie wird mit QUBO Formel C = x^T * W * x + c^T * x brechnet (Formel für Convention -1,1 Bitstrings)
    def get_energy_from_QUBO(self, weights, offset=False):
        outcomes = np.array(self.shadow)
        c = [np.sum(i) for i in weights]
        c = np.array(c)

        energy = 0
        for x in outcomes:
            energy += np.dot(x.T, np.dot(weights, x)) + np.dot(c, x)
        energy /= len(outcomes) * len(weights)  # average over all outcomes
        offset_ = 0
        if (offset):
            offset_ = self.get_energie_offset(weights, c)
        return energy + offset_
    
    def get_mean(self): # TODO, check ob das funktioniert
        if (not self.mean_calculated):
            self.mean_calculated = True
            self.mean_shadow = np.array([float(sum(col))/len(col) for col in zip(*self.shadow)])
        return self.mean_shadow
    
    # siehe meine Spezialisierung (57) (Convention -1,1 Bitstring)
    def get_energie_offset(self, weights, c):
        Q_sum = 0
        c_sum = 0
        for i in weights:
            for j in i:
                Q_sum += j
        for i in c:
            c_sum += i
        print(Q_sum, c_sum)
        return -Q_sum / 4 #+ c_sum / 2
    
    # siehe meine Spezialisierung (57) (Convention -1,1 Bitstring)
    def get_energie_from_MaxCutObjective(self, weights):
        outcomes = np.array(self.shadow)
        num_nodes = len(outcomes[0])
        counts = num_nodes * len(outcomes) # Durchschnitt vom Durchschnitt pro String
        cost = 0
        for x in outcomes:
            for i in range(num_nodes):
                for j in range(num_nodes):
                    cost -= weights[i][j] * x[i] * (1 - x[j])
        return cost / counts
    
    def get_energy_from_CodeExample(self, weights):
        if not self.minusoneandone_convention:
            self.makeZeroToMinusOne()
        outcomes = np.array(self.shadow)
        num_nodes = len(weights)
        counts = num_nodes * len(outcomes) # Durchschnitt vom Durchschnitt pro String
        res = 0
        for x in outcomes:
            obj = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if x[i] != x[j]: obj -= weights[i][j]
                
            res += obj

        return 2 * res/ counts
    
    # macht aus eindimensionale Parameterliste eine dim-dimensionale
    # Parameter Konvention: beta1, gamma1, beta2, gamma1, ...
    def get_parameters(self):
        p = []
        old_convention = self.parameters # in QAOA ist Konvention: beta1, beta2,..., gamma1, gamma2,...
        new_convention = [] # neue Konvention: beta1, gamma1, beta2, gamma2, ...
        half = int(len(old_convention)/2)
        for i in range(half):
            new_convention.append(old_convention[i])
            new_convention.append(old_convention[i + half])
        for i in range(0, len(new_convention), 2):
            p.append([new_convention[i], new_convention[i+1]])
        return p
    
    