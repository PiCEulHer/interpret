# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ..api.base import ExplainerMixin
from ..api.templates import FeatureValueExplanation
from ..utils import gen_name_from_class, unify_data, gen_perf_dicts, gen_local_selector

from sklearn.base import is_classifier

from ..glassbox.ebm.bin import unify_data2

# TODO: Remove pragma when tree interpreter updates.
# NOTE: Code coverage disabled, upstream dependency failure.
class TreeInterpreter(ExplainerMixin):  # pragma: no cover
    """ Provides 'Tree Explainer' algorithm for specific sklearn trees.

        Wrapper around andosa/treeinterpreter github package.

        https://github.com/andosa/treeinterpreter

        Currently supports (copied from README.md):

        - DecisionTreeRegressor
        - DecisionTreeClassifier
        - ExtraTreeRegressor
        - ExtraTreeClassifier
        - RandomForestRegressor
        - RandomForestClassifier
        - ExtraTreesRegressor
        - ExtraTreesClassifier

    """

    available_explanations = ["local"]
    explainer_type = "specific"

    def __init__(
        self,
        model,
        data,
        feature_names=None,
        feature_types=None,
        explain_kwargs={},
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


        self.explain_kwargs = explain_kwargs
        self.kwargs = kwargs
        self.model = model
        self.is_classifier = is_classifier(self.model)

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
        from treeinterpreter import treeinterpreter as ti

        if name is None:
            name = gen_name_from_class(self)

        X0 = X
        y0 = y
        w0 = None
        feature_types0 = self.feature_types
        feature_names0 = self.feature_names

        X, y, feature_names1, feature_types1 = unify_data(X, y, self.feature_names, self.feature_types)

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


        if self.is_classifier:
            predictions = self.model.predict_proba(X)
        else:
            predictions = self.model.predict(X)

        _, biases, contributions = ti.predict(self.model, X, **self.explain_kwargs)

        data_dicts = []
        perf_list = []
        perf_dicts = gen_perf_dicts(predictions, y, self.is_classifier)
        for i, instance in enumerate(X):
            data_dict = {}
            data_dict["data_type"] = "univariate"

            # Performance related (conditional)
            perf_dict_obj = None if perf_dicts is None else perf_dicts[i]
            data_dict["perf"] = perf_dict_obj
            perf_list.append(perf_dict_obj)

            # Names/scores
            data_dict["names"] = self.feature_names
            if self.is_classifier:
                data_dict["scores"] = contributions[i, :, 1]
            else:
                data_dict["scores"] = contributions[i, :]

            # Values
            data_dict["values"] = instance
            # TODO: Value 1 doesn't make sense for this bias, consider refactoring values to take None.
            bias = biases[0, 1] if self.is_classifier else biases[0]
            data_dict["extra"] = {"names": ["Bias"], "scores": [bias], "values": [1]}
            data_dicts.append(data_dict)

        internal_obj = {"overall": None, "specific": data_dicts}
        selector = gen_local_selector(data_dicts, is_classification=self.is_classifier)

        return FeatureValueExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )
