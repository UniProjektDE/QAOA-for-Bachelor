import numpy as np
import matplotlib.pyplot as plt


'''
TODO-List:
-Mit Pickles Listen in temporäre Daten speichern (mit Id als Zuordnung)
-gewichtete Fehlerbalken
-zum Erhalten des Paths des py Files
'''

# trennzeichen='; |, |\*|\n' möglich
# letzter angegebener Typ wird für alle restlichen verwendet
def load_file(name, types=[float], trennzeichen=';|,|\*|', skip=0, array=False):
    lists = []
    with open(name) as f:
        lines = f.readlines()
        lines = lines[skip:]
        for line in lines:
            #strings = re.split(trennzeichen,line) # \s ist whitespace, was immer geskipt werden soll
            for c in trennzeichen:
                line.replace(c, ' ')
            strings = line.split()
            list = []
            for i in range( len(strings) ):
                type_ = float
                if i >= len(types):
                    type_ = types[-1] # letztes Element
                else:
                    type_ = types[i]
                list.append( type_(strings[i]) )
            lists.append(list)
    
    # Die Listen sollen Spalten repräsentieren und nicht Zeilen
    new_lists = []
    for column in range( len(lists[0]) ):
        new_lists.append( [row[column] for row in lists] )
    if(array): new_lists = np.array(new_lists)
    return new_lists

def make_dia(x, y, xlabel='', ylabel='', title='', x_lim=None, y_lim=None, lin_reg=False, data_name='Data', legend=False, marker_type=None, errorbars=None):
    plt.figure()
    plt.title(title)
    if (marker_type != None): # bei Auswahl eines Markers, wird automatisch scatter verwendet
        plt.scatter(x, y, label=data_name, marker=marker_type)
    else:
        plt.plot(x, y, label=data_name, marker=marker_type)
    if (lin_reg):
        fit = np.polyfit(x, y, 1)
        fit_data = x * fit[0] + fit[1]
        plt.plot(x, fit_data, label="Linear Regression")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(x_lim != None): plt.xlim(x_lim)
    if(y_lim != None): plt.ylim(y_lim)
    if (errorbars != None): plt.errorbar(x, y, yerr=errorbars)
    if (legend): plt.legend()
    plt.show()

# Diagramm für mehrere Arrays; diese müssen in Listen angeben werden xs = [x1, x2, ...]
# wenn alle y dasselbe x verwenden, dann braucht man es nur alleine anzugeben, wobei es trotzdem als Liste angeben werden muss, also xs = [x]
def make_multi_in_dia(xs, ys, data_names, xlabel='', ylabel='', title='', x_lim=None, y_lim=None, marker_types=None):
    plt.figure()
    plt.title(title)

    # eine x Liste wird für alle y ggf. verwendet

    if (len(xs) == 1):
        temp = xs[0]
        xs = [temp for i in range( len(ys) )]

    for i in range( len(xs) ):
        x = xs[i]
        y = ys[i]
        name = data_names[i]
        if (marker_types != None): # bei Auswahl eines Markers, wird automatisch scatter verwendet
            if(marker_types[i] != None):
                plt.scatter(x, y, label=name, marker=marker_types[i])
            else:
                plt.plot(x, y, label=name)
        else:
            plt.plot(x, y, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(x_lim != None): plt.xlim(x_lim)
    if(y_lim != None): plt.ylim(y_lim)
    plt.legend()
    plt.show()

