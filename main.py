import numpy as np
from numpy.random import rand, randint
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# weights = x

def predict(row, weights):
    activation = weights[0]
    for i in range(1, len(row)):
        activation += weights[i] * row[i - 1]
    return 1.0 if activation >= 0.0 else 0.0


def objective(weights, data, imbalance_ratio):
    predictions = []
    err = 0
    nr_atribute = len(data[0])
    for row in data:
        predictions.append(predict(row, weights))

    for i in range(len(data)):
        if data[i][nr_atribute - 1] != predictions[i]:
            err += 1
    return err


def balanced_objective(weights, data, imbalance_ratio):
    predictions = []
    err = 0
    nr_atribute = len(data[0])
    for row in data:
        predictions.append(predict(row, weights))
    for i in range(len(data)):
        if data[i][nr_atribute - 1] != predictions[i]:
            if data[i][nr_atribute - 1] == 1:
                err += imbalance_ratio
            else:
                err += 1.0
    return err


def auc_objective(weights, data, imbalance_ratio):
    predictions = []
    list_y = []
    err = 0
    nr_atribute = len(data[0])
    for row in data:
        predictions.append(predict(row, weights))
    for i in range(len(data)):
        list_y.append(data[i][nr_atribute - 1])
    return -roc_auc_score(list_y, predictions)


def recprec_objective(weights, data, imbalance_ratio):
    predictions = []
    list_y = []
    err = 0
    nr_atribute = len(data[0])
    for row in data:
        predictions.append(predict(row, weights))
    for i in range(len(data)):
        list_y.append(data[i][nr_atribute - 1])
    return -f1_score(list_y, predictions)


def algorithm_test(weights, data):
    predictions = []

    for row in data:
        predictions.append(int(predict(row, weights)))

    return predictions


def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        start = i * n_bits
        end = (i * n_bits) + n_bits
        substring = bitstring[start:end]

        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)

        # scalarea valorilor la intervalul ales pentru weights
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])

        decoded.append(value)
    return decoded


def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):

        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]


def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1) - 2)

        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def algoritm_genetic(fnc_obj, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, data, imbalance_ratio=1.0):
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    print(pop)
    best_index = 0
    best_eval = fnc_obj(decode(bounds, n_bits, pop[0]), data, imbalance_ratio)

    for generation in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in pop]
        scores = [fnc_obj(d, data, imbalance_ratio) for d in decoded]
        for i in range(n_pop):
            if scores[i] < best_eval:
                best_index, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (generation, decoded[i], scores[i]))
        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)

        pop = children

    return [best_index, best_eval]


def get_data(inputfile):
    dataframe = pd.read_csv(inputfile)
    x = dataframe.iloc[:, 0:-1].values
    y = dataframe.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    index = len(dataframe.columns) - 1
    train_data = np.insert(x_train, index, y_train, axis=1)
    test_data = np.insert(x_test, index, y_test, axis=1)
    y_test = list(y_test)
    imbalance_ratio = sum(y == 0) / sum(y == 1)
    bnds = [[-1.0, 1.0]] * (index + 1)
    return train_data, test_data, y_test, imbalance_ratio, bnds


if __name__ == '__main__':
    train, test, test_y, imb_ratio, bounds = get_data('vowel0.csv')
    n_iter = 100
    n_bits = 16
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0 / (float(n_bits) * len(bounds))
    best, score = algoritm_genetic(balanced_objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut, train,
                                   imbalance_ratio=imb_ratio)
    print('Done!')
    best_weights = decode(bounds, n_bits, best)

    predicted = algorithm_test(best_weights, test)
    acc = accuracy_score(test_y, predicted)
    recall = recall_score(test_y, predicted)
    prec = precision_score(test_y, predicted)
    auc = roc_auc_score(test_y, predicted)
    print('accuracy : {}'.format(acc))
    print('recall : {}'.format(recall))
    print('precision : {}'.format(prec))
    print('ROC AUC : {}'.format(auc))
