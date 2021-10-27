# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation

from ..utils import gen_name_from_class, gen_local_selector
from ..utils import gen_perf_dicts
from ..utils import unify_data, unify_predict_fn
import warnings

import numpy as np
from ..glassbox.ebm.bin import unify_data2


# TODO: Make kwargs explicit.
class LimeTabular(ExplainerMixin):
    """ Exposes LIME tabular explainer from lime package, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/marcotcr/lime/blob/master/citation.bib
    """

    available_explanations = ["local"]
    explainer_type = "blackbox"

    def __init__(
        self,
        predict_fn,
        data,
        sampler=None,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
        n_jobs=1,
        **kwargs
    ):
        """ Initializes class.

        Args:
            predict_fn: Function of blackbox that takes input, and returns prediction.
            data: Data used to initialize LIME with.
            sampler: Currently unused. Due for deprecation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            explain_kwargs: Kwargs that will be sent to lime's explain_instance.
            n_jobs: Number of jobs to run in parallel.
            **kwargs: Kwargs that will be sent to lime at initialization time.
        """
        from lime.lime_tabular import LimeTabularExplainer


        X0 = data
        y0 = None
        w0 = None
        feature_types0 = feature_types
        feature_names0 = feature_names


        self.data, _, self.feature_names, self.feature_types = unify_data(
            data, None, feature_names, feature_types
        )


        X1 = self.data
        y1 = None
        w1 = None
        feature_types1 = self.feature_types
        feature_names1 = self.feature_names
        are_classifier = None

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


        self.predict_fn = unify_predict_fn(predict_fn, self.data)
        self.n_jobs = n_jobs

        if sampler is not None:  # pragma: no cover
            warnings.warn("Sampler interface not currently supported.")

        self.sampler = sampler
        self.explain_kwargs = explain_kwargs

        self.kwargs = kwargs
        final_kwargs = {"mode": "regression"}
        if self.feature_names:
            final_kwargs["feature_names"] = self.feature_names
        final_kwargs.update(self.kwargs)

        self.lime = LimeTabularExplainer(self.data, **final_kwargs)

    def explain_local(self, X, y=None, name=None):
        """ Generates local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """

        X0 = X
        y0 = y
        w0 = None
        feature_types0 = self.feature_types
        feature_names0 = self.feature_names


        if name is None:
            name = gen_name_from_class(self)
        X, y, feature_names1, feature_types1 = unify_data(X, y, self.feature_names, self.feature_types)



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



        predictions = self.predict_fn(X)
        pred_fn = self.predict_fn

        data_dicts = []
        scores_list = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, False)
        for i, instance in enumerate(X):
            lime_explanation = self.lime.explain_instance(
                instance, pred_fn, **self.explain_kwargs
            )

            names = []
            scores = []
            values = []
            feature_idx_imp_pairs = lime_explanation.as_map()[1]
            for feat_idx, imp in feature_idx_imp_pairs:
                names.append(self.feature_names[feat_idx])
                scores.append(imp)
                values.append(instance[feat_idx])
            intercept = lime_explanation.intercept[1]

            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]

            scores_list.append(scores)
            perf_list.append(perf_dict_obj)

            data_dict = {
                "type": "univariate",
                "names": names,
                "perf": perf_dict_obj,
                "scores": scores,
                "values": values,
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
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
                        "intercept": intercept,
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
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )
