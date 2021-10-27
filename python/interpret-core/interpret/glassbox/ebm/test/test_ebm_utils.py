from math import ceil, floor
from ..utils import EBMUtils
from ....utils import unify_data, unify_vector
from ....test.utils import (
    synthetic_regression,
    adult_classification
)

import numpy as np

from ..bin import unify_data2


def test_ebm_train_test_split_regression():
    data = synthetic_regression()

    X_orig = data["full"]["X"]
    y_orig = data["full"]["y"]


    X0 = X_orig
    y0 = y_orig
    w0 = None
    feature_types0 = None
    feature_names0 = None

    X, y, feature_names1, feature_types1 = unify_data(X_orig, y_orig)

    X1 = X
    y1 = y
    w1 = None
    are_classifier = None if y0 is None else not issubclass(np.array(y0).dtype.type, np.floating)

    if feature_types0 is not None:
        feature_types0 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types0]
    feature_types1 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types1]
    X2, y2, w2, feature_names2, feature_types2 = unify_data2(are_classifier, X0, y0, w0, feature_names0, feature_types0)

    if y1 is not None:
        if not np.array_equal(y1, y2):
            raise NotImplementedError("oh no EBM y!")

    if w0 is not None:
        if not np.array_equal(w1, w2):
            raise NotImplementedError("oh no EBM w!")

    if feature_names1 != feature_names2:
        raise NotImplementedError("oh no EBM feature_names!")

    if feature_types1 != feature_types2:
        raise NotImplementedError("oh no EBM feature_types!")

    X1 = X1.astype(np.object_)
    for idx in range(len(feature_types1)):
        if feature_types1[idx] == 'continuous':
            X1[:, idx] = X1[:, idx].astype(np.float64).astype(np.object_)
    X1 = X1.astype(np.unicode_)
    X2 = X2.astype(np.unicode_)
    if not np.array_equal(X1, X2):
        raise NotImplementedError("oh no EBM X!")


    w = np.ones_like(y, dtype=np.float64)
    w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

    test_size = 0.20

    X_train, X_val, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=1,
        is_classification=False
    )

    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_test_expected = ceil(test_size * num_samples)
    num_train_expected = num_samples - num_test_expected

    assert X_train.shape == (num_features, num_train_expected)
    assert X_val.shape == (num_features, num_test_expected)
    assert y_train.shape == (num_train_expected, )
    assert y_val.shape == (num_test_expected, )
    assert w_train.shape == (num_train_expected, )
    assert w_val.shape == (num_test_expected, )

    X_all = np.concatenate((X_train.T, X_val.T))
    np.array_equal(np.sort(X, axis=0), np.sort(X_all, axis=0))

def test_ebm_train_test_split_classification():
    data = adult_classification()

    X_orig = data["full"]["X"]
    y_orig = data["full"]["y"]

    X0 = X_orig
    y0 = y_orig
    w0 = None
    feature_types0 = None
    feature_names0 = None


    X, y, feature_names1, feature_types1 = unify_data(X_orig, y_orig)

    X1 = X
    y1 = y
    w1 = None
    are_classifier = None if y0 is None else not issubclass(np.array(y0).dtype.type, np.floating)

    if feature_types0 is not None:
        feature_types0 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types0]
    feature_types1 = ["nominal" if feature_type == "categorical" else feature_type for feature_type in feature_types1]
    X2, y2, w2, feature_names2, feature_types2 = unify_data2(are_classifier, X0, y0, w0, feature_names0, feature_types0)

    if y1 is not None:
        if not np.array_equal(y1, y2):
            raise NotImplementedError("oh no EBM y!")

    if w0 is not None:
        if not np.array_equal(w1, w2):
            raise NotImplementedError("oh no EBM w!")

    if feature_names1 != feature_names2:
        raise NotImplementedError("oh no EBM feature_names!")

    #if feature_types1 != feature_types2:
    #    raise NotImplementedError("oh no EBM feature_types!")

    X1 = X1.astype(np.object_)
    for idx in range(len(feature_types1)):
        if feature_types1[idx] == 'continuous':
            X1[:, idx] = X1[:, idx].astype(np.float64).astype(np.object_)
    X1 = X1.astype(np.unicode_)
    X2 = X2.astype(np.unicode_)
    if not np.array_equal(X1, X2):
        raise NotImplementedError("oh no EBM X!")



    w = np.ones_like(y, dtype=np.float64)
    w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

    test_size = 0.20

    X_train, X_val, y_train, y_val, w_train, w_val = EBMUtils.ebm_train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=1,
        is_classification=True
    )

    num_samples = X.shape[0]
    num_features = X.shape[1]
    num_test_expected = ceil(test_size * num_samples)
    num_train_expected = num_samples - num_test_expected

    # global guarantee: correct number of overall train/val/weights returned
    assert X_train.shape == (num_features, num_train_expected)
    assert X_val.shape == (num_features, num_test_expected)
    assert y_train.shape == (num_train_expected, )
    assert y_val.shape == (num_test_expected, )
    assert w_train.shape == (num_train_expected, )
    assert w_val.shape == (num_test_expected, )

    X_all = np.concatenate((X_train.T, X_val.T))
    np.array_equal(np.sort(X, axis=0), np.sort(X_all, axis=0))

    # per class guarantee: train/val count should be no more than one away from ideal
    class_counts = np.bincount(y)
    train_class_counts = np.bincount(y_train)
    val_class_counts = np.bincount(y_val)
    ideal_training = num_train_expected / num_samples
    ideal_val = num_test_expected / num_samples
    for label in set(y):
        ideal_training_count = ideal_training * class_counts[label]
        ideal_val_count = ideal_val * class_counts[label]

        assert (train_class_counts[label] == ceil(ideal_training_count) 
            or train_class_counts[label] == floor(ideal_training_count) 
            or train_class_counts[label] == ideal_training_count)
        assert (val_class_counts[label] == ceil(ideal_val_count) 
            or val_class_counts[label] == floor(ideal_val_count) 
            or val_class_counts[label] == ideal_val_count)