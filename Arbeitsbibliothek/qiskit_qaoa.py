from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
#from qiskit import Aer, execute
from qiskit_aer import Aer
from qiskit.circuit import Parameter

from scipy.optimize import minimize
import numpy as np
import networkx as nx

from Arbeitsbibliothek.mytoolbox import make_dia
from Arbeitsbibliothek.myshadow import ShadowShot, ShadowCollection
import itertools
import copy


shadowList = []
expectionList = []


        

def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    
    Args:
        x: str
           solution bitstring
           
        G: networkx graph
        
    Returns:
        obj: float
             Objective
    """
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
            
    return obj


def compute_expectation(counts, G):
    
    """
    Computes expectation value based on measurement results
    
    Args:
        counts: dict
                key as bitstring, val as count
           
        G: networkx graph
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():       
        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count
        
    return avg/sum_count

running_index = 0

# We will also bring the different circuit components that
# build the qaoa circuit under a single function
def create_qaoa_circ(G, theta, p):
    
    """
    Creates a parametrized qaoa circuit
    
    Args:  
        G: networkx graph
        theta: list
               unitary parameters
                     
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = len(G.nodes())
    #p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    
    beta = theta[:p]
    gamma = theta[p:]
    
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    
    for irep in range(0, p):
        
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
            
    qc.measure_all()
        
    return qc

def shadowOfMemory(result, shots=0):
    eigenval_array = [] # Liste der gemessenen Bitstrings
    memory = result.get_memory()
    if (shots != 0):
        memory = memory[:shots] # nur eine gewisse Anzahl von Strings
    for mem in memory:
        array = [int(x) for x in mem]
        eigenval_array.append(array)
    return eigenval_array

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(G, p, shots=1024, callByExc=None, seed=10):
    
    """
    Runs parametrized circuit
    
    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    
    def execute_circ(theta):
        
        qc = create_qaoa_circ(G, theta, p)
        #result = backend.run(qc, seed_simulator=seed, 
        #                     nshots=shots, memory=True).result()
        result = execute(qc, backend, shots=shots, seed_simulator=10, memory=True).result()
        counts = result.get_counts()
        
        shadow = shadowOfMemory(result)
        theta_ = copy.deepcopy(theta)
        
        
        expec = compute_expectation(counts, G)
        expectionList.append(expec)
        if callByExc!=None:
            global running_index
            expec, shadow = callByExc(expec, copy.deepcopy(G), copy.deepcopy(theta), copy.deepcopy(result), copy.deepcopy(shadow), running_index)
            running_index +=1
        s = ShadowShot(shadow, theta_, dim = p)
        shadowList.append(s)
        return expec
    
    return execute_circ # Funktion wird zurückgegeben


def run_qaoa(graph, layers=1, shots=1024, callByExp=None, seed=10, max_iter=None):
    shadowList.clear()
    expectionList.clear()
    global running_index
    running_index = 0
    expectation = get_expectation(graph, p=layers, shots=shots, callByExc=callByExp, seed=seed)
    initial_parameters = [1.0, 1.0]

    #Append two further parameters for each "extra" layer (insgesamt p * 2 Parameter)
    for i in range(layers - 1):
        initial_parameters.append(1.0)
        initial_parameters.append(1.0)
    
    # Minimaze Hamiltonian
    res = minimize(expectation, 
                        initial_parameters, 
                      method='COBYLA',
                      options={'maxiter': max_iter})
    
    gw = nx.to_numpy_array(graph)
    shadowList_ = copy.deepcopy(shadowList) # muss gemacht werden, da sonst Referenz wiedergeben werden kann
    gw_ = copy.deepcopy(gw) # Sicherheitshalber
    return res, ShadowCollection(shadowList_, gw_), expectionList

# gibt Ergebnis von QAOA ohne Optimierungsverfahren an
# WICHTIG: Parameterkonvention - die erste Hälfe stellte die beta Parameter dar, die andere die Gamma (1. Parameter ist für erste Schicht)
def run_single_qaoa_cir(graph, parameters, layers=1):
    global running_index
    running_index = 0
    if (len(parameters) != layers * 2):
        raise ValueError('Parameteranzahl passt nicht zu Layeranzahl!!! (Erwarte '+ str(layers * 2) + ' Parameter, doch ' + str(len(parameters) + ' sind vorhanden)'))
    shadowList.clear()
    expectionList.clear()
    expectation = get_expectation(graph, p=layers)
    res = expectation(parameters)
    gw = nx.to_numpy_array(graph)
    shadowList_ = copy.deepcopy(shadowList)
    gw_ = copy.deepcopy(gw)
    return res, ShadowCollection(shadowList_, gw_), expectionList
