import random
import statistics
from math import sqrt
from sklearn.metrics import precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, f1_score


class AbstractEvaluator:
    def name(self):
        raise NotImplemented()

    def evaluateKFold(self, yKPred, yKTest):
        raise NotImplemented()


class PrecisionRecallEval(AbstractEvaluator):

    def name(self):
        return 'PrecisionRecallEval'

    def evaluateKFold(self, yKPred, yKTest):
        precisions = []
        recalls = []
        f1_scores = []
        for fold_pred, fold_test in zip(yKPred, yKTest):
            precisions.append(precision_score(fold_test, fold_pred, average='weighted'))
            recalls.append(recall_score(fold_test, fold_pred, average='weighted'))
            f1_scores.append((f1_score(fold_test, fold_pred, average='weighted')))
        return {'precisions': statistics.mean(precisions), 'recalls': statistics.mean(recalls),
                'f1_measure': statistics.mean(f1_scores)}


class ConfusionMatrixEval(AbstractEvaluator):
    def name(self):
        return 'ConfusionMatrixEval'


class ROCEval(AbstractEvaluator):
    def name(self):
        return 'ROCEval'


class RegressionEval(AbstractEvaluator):

    def name(self):
        return 'Regression_Evaluation'

    def evaluateKFold(self, yKPred, yKTest):
        mean_abs_error = []
        root_mean_sq_error = []
        r2score = []
        for fold_pred, fold_test in zip(yKPred, yKTest):
            mean_abs_error.append(mean_absolute_error(fold_test, fold_pred))
            root_mean_sq_error.append(sqrt(mean_squared_error(fold_test, fold_pred)))
            r2score.append(r2_score(fold_test, fold_pred))
        return {'mean_abs_err': statistics.mean(mean_abs_error), 'RMSE': statistics.mean(root_mean_sq_error),
                'R2_score': statistics.mean(r2score)}
