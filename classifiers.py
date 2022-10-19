import sklearn
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import torch


class AbstractClassifier:
    def __init__(self):
        self.params = {}

    def name(self):
        raise NotImplemented()

    def fit(self, XTrain, yTrain):
        raise NotImplemented()

    def predict(self, XTest):
        raise NotImplemented()

    def setMetaParams(self, concrete_parameters):
        self.params = concrete_parameters

    def getMetaParamsDescription(self):
        raise NotImplemented()


class DummyClassifier(AbstractClassifier):
    def __init__(self):
        super().__init__()
        self.model = sklearn.dummy.DummyClassifier()

    def name(self):
        return 'I am a dummy classifier'

    def setMetaParams(self, concrete_parameters):
        self.params = concrete_parameters

    def fit(self, XTrain, yTrain):
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)

    def getMetaParamsDescription(self):
        return {
            'dummyC': {
                'type': 'continuous',
                'min': 0.01,
                'max': 1000,
                'distribution': 'uniform'
            }
        }


class AdaBoostC(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "AdaBoost_Classifier"

    def getMetaParamsDescription(self):
        return {
            'n_estimators': {
                'type': 'discrete',
                'min': 40,
                'max': 100
            },
            'learning_rate': {
                'type': 'continuous',
                'min': 0.01,
                'max': 1.5,
                'distribution': 'uniform'
            },
            'algorithm': ['SAMME'],
            'random_state': 0
        }

    def fit(self, XTrain, yTrain):
        self.model = AdaBoostClassifier(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class LogisticRegressor(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Logistic_Regression_Classifier"

    def getMetaParamsDescription(self):
        return {
            'class_weight': 'balanced',
            'multi_class': 'multinomial',
            'random_state': 0,
            'C': {
                'type': 'continuous',
                'distribution': 'uniform',
                'min': 0.5,
                'max': 4
            },
            'solver': ['lbfgs', 'sag']
        }

    def fit(self, XTrain, yTrain):
        self.model = sklearn.linear_model.LogisticRegression(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class SupportVectorC(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'Support_Vector_Classifier'

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'gamma': {
                'type': 'continuous',
                'min': 1,
                'max': 20,
                'distribution': 'uniform'
            },
            'C': {
                'type': 'discrete',
                'min': 100,
                'max': 250,
            },
            'kernel': ['linear', 'rbf']
        }

    def fit(self, XTrain, yTrain):
        self.model = sklearn.svm.SVC(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class DecisionTreeC(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'Decision_Tree_Classifier'

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'max_depth': {
                'type': 'discrete',
                'min': 5,
                'max': 12,
            }
        }

    def fit(self, XTrain, yTrain):
        self.model = sklearn.tree.DecisionTreeClassifier(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class RandomForestC(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return 'Random_Forest_Classifier'

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'n_estimators': {
                'type': 'discrete',
                'min': 50,
                'max': 100
            },
            'max_depth': {
                'type': 'discrete',
                'min': 10,
                'max': 15,
            },
        }

    def fit(self, XTrain, yTrain):
        self.model = sklearn.ensemble.RandomForestClassifier(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class GaussianNB(AbstractClassifier):

    def __init__(self):
        super().__init__()

    def name(self):
        return 'GaussianNB_Classifier'

    def getMetaParamsDescription(self):
        return {}

    def fit(self, XTrain, yTrain):
        self.model = sklearn.naive_bayes.GaussianNB()
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class MlpClassifier(AbstractClassifier):

    def __init__(self):
        super().__init__()

    def name(self):
        return 'Neural_Network_MLP_Classifier'

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'max_iter': {
                'type': 'discrete',
                'min': 150,
                'max': 200
            },
            'learning_rate_init': {
                'type': 'continuous',
                'distribution': 'uniform',
                'min': 0.01,
                'max': 2
            },
            'solver': ['lbfgs', 'sgd']
        }

    def fit(self, XTrain, yTrain):
        self.model = MLPClassifier(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class SupportVectorR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Support_Vector_Regressor"

    def getMetaParamsDescription(self):
        return {
            'kernel': ['rbf'],
            'gamma': {
                'type': 'continuous',
                'min': 1,
                'max': 2,
                'distribution': 'uniform'
            },
            'C': {
                'type': 'continuous',
                'min': 500,
                'max': 1000,
            },
        }

    def fit(self, XTrain, yTrain):
        self.model = SVR(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class DecisionTreeR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Decision_Tree_Regressor"

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'max_depth': {
                'type': 'discrete',
                'min': 1,
                'max': 5
            }
        }

    def fit(self, XTrain, yTrain):
        self.model = DecisionTreeRegressor(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class RandomForestR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Random_Forest_Regressor"

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'n_estimators': {
                'type': 'discrete',
                'min': 200,
                'max': 500
            }
        }

    def fit(self, XTrain, yTrain):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class AdaBoostR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "AdaBoost_Regressor"

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'n_estimators': {
                'type': 'discrete',
                'min': 80,
                'max': 100
            },
            'learning_rate': {
                'type': 'continuous',
                'min': 0.01,
                'max': 1.5,
                'distribution': 'uniform'
            }
        }

    def fit(self, XTrain, yTrain):
        self.model = AdaBoostRegressor(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class GaussianProcessR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Gaussian_Process_Regressor"

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'kernel': [DotProduct()+WhiteKernel(), RBF()+WhiteKernel(), RationalQuadratic()]
        }

    def fit(self, XTrain, yTrain):
        self.model = GaussianProcessRegressor(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class LinearR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Linear_Regressor"

    def getMetaParamsDescription(self):
        return {
            'fit_intercept': ['True'],
            'normalize': ['False']
        }

    def fit(self, XTrain, yTrain):
        self.model = LinearRegression(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class NeuralNetworkR(AbstractClassifier):
    def __init__(self):
        super().__init__()

    def name(self):
        return "Neural_Network_Regressor"

    def getMetaParamsDescription(self):
        return {
            'random_state': 0,
            'max_iter': {
                'type': 'discrete',
                'min': 200,
                'max': 1000
            },
            'activation': ['tanh','logistic']
        }

    def fit(self, XTrain, yTrain):
        self.model = MLPRegressor(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)


class KNearest(AbstractClassifier):
    def __init__(self):
        super().__init__()


    def name(self):
        return 'K_neighbors_classifier'

    def getMetaParamsDescription(self):
        return {
            'n_neighbors': {
                'type': 'discrete',
                'min': 10,
                'max': 20,
            }
        }

    def fit(self, XTrain, yTrain):
        self.model = sklearn.neighbors.KNeighborsClassifier(**self.params)
        self.model.fit(XTrain, yTrain)

    def predict(self, XTest):
        return self.model.predict(XTest)




