import os
import tensorflow as tf
import numpy as np
import random
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Do other imports now...

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

seed_value = 42


def get_mlp_model(save_initial_weights_path=None, save_fresh_mlp_path=None, input_shape=480):
    """
    Creates A fresh-mlp model for evaluation
    :param save_initial_weights_path:
    :param save_fresh_mlp_path:
    :param input_shape:
    :return:
    """
    classifier = Sequential()
    classifier.add(Dense(200, activation='relu', kernel_initializer='random_normal', input_dim=input_shape))
    classifier.add(Dense(100, activation='relu', kernel_initializer='random_normal', input_dim=200))
    classifier.add(Dense(50, activation='relu', kernel_initializer='random_normal', input_dim=100))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    classifier.compile(optimizer='ADAM', loss='binary_crossentropy', metrics=['mse', 'accuracy'])

    if save_initial_weights_path is not None:
        classifier.save_weights(save_initial_weights_path)

    if save_fresh_mlp_path is not None:
        classifier.save(save_fresh_mlp_path)
    return classifier


def init_seeds():
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def reduce_by_indexes(arr, indexes):
    """
    :param arr: Context-Feature arr
    :param indexes: indexes of chosen features
    :return: reduced arr
    """
    res = []

    for i in range(len(arr)):
        if i in indexes:
            res += [arr[i]]
        else:
            res += [0]

    return np.array(res)


def reduce_all_by_indexes(matrix, indexes):
    """
    reduces every vector in the arr
    :param matrix: matrix of contextual features
    :param indexes: indexes of chosen features
    :return:
    """
    return np.array(list(map(lambda element: reduce_by_indexes(element, indexes), matrix)))


def run_feed_forward_selection(context_train, context_val, class_weights, model, fresh_weights_path, y_train, y_val,
                               alive_path):
    """

    :param context_train: training data
    :param context_val: validation data
    :param class_weights: class-weights for training
    :param model: Model for evaluation
    :param fresh_weights_path: Fresh Weights for initialization
    :param y_train: training labels
    :param y_val: validation labels
    :param alive_path: Path to write results to
    :return: Selected indexes
    """

    selected_features = []
    sample = context_train[0]
    best_score = -float('inf')

    while True:
        print(f"Starting iteration : Best AUC {best_score} , len(featurs) = {len(selected_features)}")
        best_feature = -1

        current_size = len(selected_features)

        for i in tqdm(range(len(sample))):
            np.save(alive_path, np.array(selected_features))

            model.load_weights(fresh_weights_path)

            assert len(selected_features) == current_size

            if i in selected_features:
                continue

            current_features = selected_features.copy()
            current_features += [i]

            assert len(current_features) == current_size + 1

            x_train = reduce_all_by_indexes(context_train, current_features)
            x_val = reduce_all_by_indexes(context_val, current_features)

            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
            mcp = ModelCheckpoint('../data_cars_ds/FF_best_mlp_feature_selection.h5', save_best_only=True, mode='min')
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=100, batch_size=256, verbose=0, callbacks=[es, mcp], class_weights=class_weights)

            model = load_model('../data_cars_ds/FF_best_mlp_feature_selection.h5')
            val_predict = model.predict(x_val, batch_size=256)
            current_auc = roc_auc_score(y_val, val_predict)

            if current_auc > best_score:
                print(f"{current_auc} took over {best_score} , with index {i}")
                best_feature = i
                best_score = current_auc

            del es, mcp, current_features

        # no improvement
        if best_feature == -1:
            break

        selected_features.append(best_feature)

    return selected_features


def run_backwards_elimination_selection(context_train, context_val, class_weights, model, fresh_weights_path, y_train,
                                        y_val, alive_path):
    """

    :param context_train: training data
    :param context_val: validation data
    :param class_weights: class-weights for training
    :param model: Model for evaluation
    :param fresh_weights_path: Fresh Weights for initialization
    :param y_train: training labels
    :param y_val: validation labels
    :param alive_path: Path to write results to
    :return: Selected indexes
    """
    sample = context_train[0]
    dims = len(sample)

    selected_features = list(range(dims))
    best_score = -float('inf')

    while True:
        print(f"Starting iteration : Best AUC {best_score} , len(featurs) = {len(selected_features)}")
        worst_feature = -1

        current_size = len(selected_features)

        for i in tqdm(range(len(sample))):
            np.save(alive_path, np.array(selected_features))
            model.load_weights(fresh_weights_path)

            assert len(selected_features) == current_size

            if i not in selected_features:
                continue

            current_features = selected_features.copy()
            current_features.remove(i)

            assert len(current_features) == current_size - 1

            x_train = reduce_all_by_indexes(context_train, current_features)
            x_val = reduce_all_by_indexes(context_val, current_features)

            es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
            mcp = ModelCheckpoint('../data_cars_ds/BE_best_mlp_feature_selection.h5', save_best_only=True, mode='min')
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=100, batch_size=256, verbose=0, callbacks=[es, mcp], class_weights=class_weights)

            model = load_model('../data_cars_ds/BE_best_mlp_feature_selection.h5')
            val_predict = model.predict(x_val, batch_size=256)
            current_auc = roc_auc_score(y_val, val_predict)

            if current_auc > best_score:
                print(f"{current_auc} took over {best_score} , with index {i}")
                worst_feature = i
                best_score = current_auc

            del es, mcp, current_features

        # no improvement
        if worst_feature == -1:
            break

        selected_features.remove(worst_feature)

    return selected_features


def get_train_test_val_split(arr):
    x_train, x_test = train_test_split(arr, test_size=0.3, shuffle=False)
    x_val, x_test = train_test_split(x_test, test_size=0.66, shuffle=False)
    return x_train, x_test, x_val


if __name__ == '__main__':
    init_seeds()

    Y_train, _, Y_val = get_train_test_val_split(np.load('../data_hero_ds/y.npy'))
    X_train, _, X_val = get_train_test_val_split(np.load('../data_hero_ds/X.npy'))

    density = sum(Y_train) / len(Y_train)
    ones_weight = (1 - density) / density
    zero_weight = 1
    print(ones_weight, density)

    class_Weights = {0: zero_weight, 1: ones_weight}
    fresh_weights_Path = '../feature_selections/FF_mlp_initial.h5'
    model_ = get_mlp_model(fresh_weights_Path, None, len(X_train[0]))

    save_path = '../feature_selections/FF_selection.npy'
    features = run_feed_forward_selection(X_train, X_val, class_Weights, model_, fresh_weights_Path, Y_train,
                                          Y_val, save_path)
    np.save(save_path, features)
