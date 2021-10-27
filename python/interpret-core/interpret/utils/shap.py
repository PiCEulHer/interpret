# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.templates import FeatureValueExplanation
from . import gen_name_from_class, unify_data, gen_perf_dicts, gen_local_selector

from ..glassbox.ebm.bin import unify_data2

def shap_explain_local(explainer, X, y=None, name=None, is_classification=False, check_additivity=True):
    if name is None:
        name = gen_name_from_class(explainer)

    X0 = X
    y0 = y
    w0 = None
    feature_types0 = explainer.feature_types
    feature_names0 = explainer.feature_names

    X, y, feature_names1, feature_types1 = unify_data(X, y, explainer.feature_names, explainer.feature_types)


    X1 = X
    y1 = y
    w1 = None
    are_classifier = None if y0 is None else not issubclass(y0.dtype.type, np.floating)

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

    if is_classification:
        all_shap_values = explainer.shap.shap_values(X, check_additivity=check_additivity)[1]
        expected_value = explainer.shap.expected_value[1]
    else:
        all_shap_values = explainer.shap.shap_values(X, check_additivity=check_additivity)
        expected_value = explainer.shap.expected_value

    predictions = explainer.predict_fn(X)

    data_dicts = []
    scores_list = all_shap_values
    perf_list = []
    perf_dicts = gen_perf_dicts(predictions, y, False)
    for i, instance in enumerate(X):
        shap_values = all_shap_values[i]
        perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

        perf_list.append(perf_dict_obj)

        data_dict = {
            "type": "univariate",
            "names": explainer.feature_names,
            "perf": perf_dict_obj,
            "scores": shap_values,
            "values": instance,
            "extra": {
                "names": ["Base Value"],
                "scores": [expected_value],
                "values": [1],
            },
        }
        data_dicts.append(data_dict)

    internal_obj = {
        "overall": None,
        "specific": data_dicts,
        "mli": [
            {
                "explanation_type": "local_feature_importance",
                "value": {
                    "scores": scores_list,
                    "intercept": expected_value,
                    "perf": perf_list,
                },
            }
        ],
    }
    internal_obj["mli"].append(
        {
            "explanation_type": "evaluation_dataset",
            "value": {"dataset_x": X, "dataset_y": y},
        }
    )
    selector = gen_local_selector(data_dicts, is_classification=False)

    return FeatureValueExplanation(
        "local",
        internal_obj,
        feature_names=explainer.feature_names,
        feature_types=explainer.feature_types,
        name=name,
        selector=selector,
    )
