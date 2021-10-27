# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

import numpy as np

from ..utils.shap import shap_explain_local
from sklearn.base import is_classifier

from ..api.base import ExplainerMixin
from ..utils import unify_predict_fn, unify_data

from ..glassbox.ebm.bin import unify_data2

class ShapTree(ExplainerMixin):
    """ Exposes tree specific SHAP approximation, in interpret API form.
    If using this please cite the original authors as can be found here: https://github.com/slundberg/shap
    """

    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(
        self,
        model,
        data,
        feature_names=None,
        feature_types=None,
        explain_kwargs=None,
        n_jobs=1,
        **kwargs
    ):
        """ Initializes class.

        Args:
            model: A tree object that works with Tree SHAP.
            data: Data used to initialize SHAP with.
            sampler: Currently unused. Due for deprecation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            explain_kwargs: Currently unused. Due for deprecation.
            n_jobs: Number of jobs to run in parallel.
            **kwargs: Kwargs that will be sent to SHAP at initialization time.
        """
        import shap

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


        self.model = model
        self.is_classifier = is_classifier(self.model)
        if is_classifier:
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        self.predict_fn = unify_predict_fn(predict_fn, self.data)

        self.n_jobs = n_jobs

        self.explain_kwargs = explain_kwargs
        self.kwargs = kwargs

        self.shap = shap.TreeExplainer(model, data, **self.kwargs)

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided instances.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        # NOTE: Check additivity is set to false by default as there is a problem with Mac OS that
        # doesn't always reach the specified precision.
        check_additivity = self.kwargs.get("check_additivity", False)
        return shap_explain_local(
            self, X, y=y, name=name, is_classification=self.is_classifier, check_additivity=check_additivity
        )
