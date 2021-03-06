import numpy as np

from tempfile import NamedTemporaryFile
import pandas as pd

import os
import subprocess


path = os.path.dirname(os.path.realpath(__file__))


def model(X_train, y_train, X_test):
    where_are_NaNs = np.isnan(X_train)
    X_train[where_are_NaNs] = -1
    where_are_NaNs = np.isnan(X_test)
    X_test[where_are_NaNs] = -1

    y_train_shape_dim = len(y_train.shape)
    if y_train_shape_dim == 1:
        y_train = y_train[:, np.newaxis]


    header = ["TARGET"] + ["v%d" % (i,) for i in xrange(X_train.shape[1])]

    train = np.concatenate((y_train, X_train), axis=1)
    train = pd.DataFrame(train, columns=header)

    test = pd.DataFrame(X_test, columns=header[1:])

    temp_dir = '.'
    train_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
    train.to_csv(train_tmp_file, header=header, index=False)
    train_tmp_file.close()

    test_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
    test.to_csv(test_tmp_file, header=header[1:], index=False)
    test_tmp_file.close()

    proba_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
    proba_tmp_file.close()

    targets_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
    targets_tmp_file.close()

    try:

        subprocess.call("cd %s; luajit model_generic.lua --train %s --test %s --outputproba %s --outputtargets %s" % (path, train_tmp_file.name, test_tmp_file.name, proba_tmp_file.name, targets_tmp_file.name), shell=True)

        proba = pd.read_csv(proba_tmp_file.name, header=None).values
        targets = pd.read_csv(targets_tmp_file.name, header=None).values
        if len(targets.shape) == 2:
            targets = targets[:, 0]

        return targets, proba
    except Exception, e:
        print str(e)

    finally:
        os.remove(train_tmp_file.name)
        os.remove(test_tmp_file.name)
        os.remove(proba_tmp_file.name)
        os.remove(targets_tmp_file.name)

if __name__ == "__main__":

    train = pd.read_csv("input/train.csv").values
    test = pd.read_csv("input/test.csv").values
    test = test[:, 1:]
    X = train[:, 1:]
    y = train[:, 0]
    targets, proba = model(X, y, test)
    print len(targets), len(proba), len(test)
