import networkx as nx
import numpy as np
from numpy import pi
###### importiert .py Daten vom PC
import sys
from Arbeitsbibliothek.qiskit_qaoa import run_qaoa, run_single_qaoa_cir
from Arbeitsbibliothek.myshadow import ShadowShot, ShadowCollection
from Arbeitsbibliothek.mytoolbox import make_dia, make_multi_in_dia
from Arbeitsbibliothek.mynntools import model_for_shadows, save_model, load_model, sample_shadow_from_NN
import copy
import matplotlib.pyplot as plt
import random


def make_compare_dia(x, ys, errs, labels, title=""):
    plt.figure()
    plt.title(title)
    for y, err, label in zip(ys, errs, labels):
        #plt.plot(x, y, label=label)
        plt.errorbar(x, y, yerr=err, label=label)
    plt.legend()
    plt.show()

def compare_product_state_shadow_strings(shadow, NNshadow):
    realBitstrings = shadow.shadowList[0].shadow
    hitProbability = []
    for s, real in zip(NNshadow.shadowList, realBitstrings):
        count = 0
        hits = np.zeros(len(real))
        for bitstring in s.shadow:
            for i in range(len(bitstring)):
                #print(bitstring[i],real[i])
                if bitstring[i] == real[i]: hits[i] +=1
            count += 1
        hitProbability.append(hits / count)
    return hitProbability


# Shadow: ein ShadowCollection oder eine Liste von denen; recalculate=False, falls man NNs schon berechnet hat, anzahlNNs: wieviele NN initialisierungen sollen für die Statistik verwendet werden
def run_real_NN_comparison(shadow, path, anzahlNNs=1, recalculate=True, test_set_shadow=None, max_iter=20, title=""):
    print(shadow)
    # macht Liste zur einer Collection
    shadow_ = shadow
    realData = shadow
    if hasattr(shadow, "__len__"):
        realData = sum(shadow_)
    if np.isscalar(realData):
        raise ValueError('Anscheinend leere Liste von ShadowCollections (oder Skalar) wurde übergeben!')
    gw = realData.graph_weights
    Ereal = realData.get_energy_list() # Energie aus der Messung
    x = [i for i in range(len(Ereal))]

    para_list = []
    for sh in realData.shadowList:
        para_list.append(sh.parameters)
    
    ENNs = []
    model_list = []
    for i in range(anzahlNNs):
        model = None
        path_ = path + "_index" + str(i)
        if(recalculate):
            model, _ = model_for_shadows(shadow_, max_epoch=max_iter, learning_rate=0.001)
            save_model(model, path_)
        else:
            model = load_model(path_)
        model_list.append(model)
        for param in model.parameters():
            print(param.data)
        NNData = sample_shadow_from_NN(model, para_list, weights=gw)

        ### 
        temp = compare_product_state_shadow_strings(realData,NNData)
        print(temp)
        #####

        ENN = NNData.get_energy_list() # Energie aus NN
        ENNs.append(ENN)
    ENNs = np.array(ENNs)

    ENN_sum = np.mean(ENNs, axis=0)
    ENN_errorbar = np.std(ENNs, axis=0)
    #make_multi_in_dia([x], [Ereal, ENN_sum], ["Energy Measurement", "Energy NN"], title="Trainingsdaten")
    make_compare_dia(x, [Ereal, ENN_sum], [None, ENN_errorbar] ,["Energy Measurement", "Energy NN"], title="Trainingsdaten" + " " + title)

    # Test set wird hier dargestellt
    if(test_set_shadow != None):
        test_real_shadow = sum(test_set_shadow)
        test_Ereal = test_real_shadow.get_energy_list()
        para_list_test = []
        test_ENNs = []
        for sh in test_real_shadow.shadowList:
            para_list_test.append(sh.parameters)
        
        for mod in model_list:
            NNData = sample_shadow_from_NN(mod, para_list_test, weights=gw)
            ENN = NNData.get_energy_list() # Energie aus NN
            test_ENNs.append(ENN)
        test_ENNs = np.array(test_ENNs)
        test_ENN_sum = np.mean(test_ENNs, axis=0)
        test_ENN_errorbar = np.std(test_ENNs, axis=0)
        x = [i for i in range(len(Ereal))]
        #make_multi_in_dia([x], [test_Ereal, test_ENN_sum], ["Energy Measurement", "Energy NN"], title="Testdaten")
        make_compare_dia(x, [test_Ereal, test_ENN_sum], [None, test_ENN_errorbar] ,["Energy Measurement", "Energy NN"], title="Testdaten" + " " + title)

 