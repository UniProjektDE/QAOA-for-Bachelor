from scipy.optimize import minimize
from numpy import pi
import numpy as np


global_func = None # hier wird die Expection Funktion global gespeichert

# index, sagt auf welches theta_i sich bezogen wird 
def shift_rule(func, thetas, index):
    thetas1 = thetas
    thetas2 = thetas
    thetas1[index] += pi/2
    thetas2[index] -= pi/2
    return ( func(thetas1) - func(thetas2) ) / 2

# gradient mit Shift Rule
def gradient(thetas):
    vector = []
    for i in range(len(thetas)):
        vector.append(shift_rule(global_func, thetas, i))
    vector = np.array(vector)
    return vector

# Optinion der einzelnen Funktionen: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.show_options.html#scipy.optimize.show_options

# jac: Methode, die den Gradientenvektor berechnet (Nur für CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr)
# jac : {callable, ‘2-point’, ‘3-point’, ‘cs’, bool}
# maxiter in options: ist maximale Anzahl von Iterationen
def minimize_function(function, initial_guess, maxiiter=100000000, method='CG'):
    global global_func # erlaubt die Änderung einer Globalen Variable
    global_func = function
    validMethod = False
    methodList = ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact', 'trust-constr']
    for m in methodList:
        if method == m: validMethod = True
    if not validMethod:
        raise ValueError('Ausgewählte Optimierungsmethod ist nicht unterstützt!!!')
    result = minimize(
        function, 
        initial_guess, 
        method=method, 
        jac=gradient, 
        options={'maxiter':maxiiter}
    )
    return result

'''
def func1(list):
    a = list[0]
    b = list[1]
    c = list[2]
    print("Hallo")
    return  4 * a * a + b * b + c * c / 2 + 2

res = minimize_function(func1, 
                        [1,1,1])

print(res)
'''