import itertools
import json
import time, os
import multiprocessing as mp

from classifiers import AbstractClassifier, LogisticRegressor, AdaBoostC, MlpClassifier, GaussianNB, AdaBoostR, \
    SupportVectorC, DecisionTreeC, RandomForestC, SupportVectorR, DecisionTreeR, RandomForestR, GaussianProcessR, \
    LinearR, NeuralNetworkR, KNearest
from datasets import AbstractDataset, Faults, Adults, SeismicBumps, ThoracicSurgery, Yeast, RedWine, CreditCard, \
    Diabetic, BreastCancer, AusCredit, GermanCredit, WhiteWine, AquaticToxicity, ParkinsonSpeech, FacebookMetrics, \
    CommunitiesAndCrime, BikeSharing, StudentPerformance, ConcreteCompressiveStrength, SGEMMPerformance, \
    MerckChallengeData1, MerckChallengeData2, TwitterHate

from evaluators import PrecisionRecallEval, ConfusionMatrixEval, ROCEval, AbstractEvaluator, RegressionEval
from hyperparameters import RandomParameterGenerator

from matplotlib import pyplot as plt
import pandas as pd

USE_MULTIPROCESSING = False
NUMBER_OF_K_FOLDS = 1
HYPER_PARAMS_COMBINATION_COUNT = 1
DIR = os.getcwd() + '/../Output/'

timestr = time.strftime("%m-%d-%H:%M:%S")
path = DIR + timestr + '/'
print(path)


def run_pipeline(list_of_datasets, list_of_classifiers, list_of_evaluators):
    meta_parameter_generator = RandomParameterGenerator()

    pool = mp.Pool(mp.cpu_count())
    pipeline_results = []

    try:
        if not os.path.isdir(path):
            os.mkdir(path)
    except OSError as e:
        print("Couldn't create directory")

    if USE_MULTIPROCESSING:
        pipeline_results = pool.starmap(evaluate_combination, [
            (c, d, list_of_evaluators, meta_parameter_generator)
            for d, c in itertools.product(list_of_datasets, list_of_classifiers)
        ])
    else:
        for d, c in itertools.product(list_of_datasets, list_of_classifiers):
            pipeline_results.append(evaluate_combination(c, d, list_of_evaluators, meta_parameter_generator))

    # print(json.dumps(pipeline_results, indent='  '))


def evaluate_combination(classifier: AbstractClassifier, dataset: AbstractDataset, list_of_evaluators,
                         meta_parameter_generator):
    start = time.time()
    result_path = path + dataset.name() + '/'
    result_table = pd.DataFrame([],
                                columns=['param1', 'param2', 'param3', 'param4', 'param5'])

    print(dataset.name(), ' | ', classifier.name(), ' started')
    meta_params = classifier.getMetaParamsDescription()
    run_results = []
    eval_results_within_classifier = {}  # Used for storing evaluation results for plotting table
    for param_search_iteration in range(HYPER_PARAMS_COMBINATION_COUNT):

        concrete_parameters = meta_parameter_generator.generate(meta_params)
        classifier.setMetaParams(concrete_parameters)

        eval_results_within_metaparam = {}
        result_table_data = {}  # Dictionary to contain final results to dump on the file as a latex table
        yKPred = []
        yKTest = []
        for XTrain, XTest, yTrain, yTest in dataset.KfFoldData(NUMBER_OF_K_FOLDS):
            # Train scaler based on training dataset
            scaler = dataset.get_data_scaled(XTrain)
            classifier.fit(scaler.transform(XTrain), yTrain)
            yKPred.append(classifier.predict(scaler.transform(XTest)))
            yKTest.append(yTest)
        for evaluator in list_of_evaluators:
            evaluator: AbstractEvaluator
            for k, v in evaluator.evaluateKFold(yKPred, yKTest).items():
                if k not in eval_results_within_metaparam:
                    eval_results_within_metaparam[k] = []
                eval_results_within_metaparam[k].append(v)

        run_results.append({
            'iteration': param_search_iteration,
            'params': concrete_parameters,
            'evaluation': eval_results_within_metaparam
        })
        for k, v in eval_results_within_metaparam.items():
            if k not in eval_results_within_classifier:
                eval_results_within_classifier[k] = []
            eval_results_within_classifier[k] += v
            result_table_data[k] = v[0]

        # result_table_data = {}
        pi = 1
        for k, v in concrete_parameters.items():
            result_table_data[f"param{pi}"] = f"{k} = {v}"
            pi += 1

        # result_table_data['precision'] = eval_results_within_metaparam['precisions'][0]
        # result_table_data['recall'] = eval_results_within_metaparam['recalls'][0]
        result_table = result_table.append(result_table_data, ignore_index=True)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    fig, ax = plt.subplots()

    for k, data in eval_results_within_classifier.items():
        ax.plot(data, label=k)
    ax.legend()
    ax.set_title(f"'{dataset.name()}' with '{classifier.name()}'")
    fig.savefig(result_path + classifier.name() + '.png')
    # ax.show()
    with open(result_path + classifier.name() + '.txt', 'w') as file:
        file.write(result_table.to_latex())

    took_time = time.time() - start
    print(dataset.name(), ' | ', classifier.name(), ' finished, took:', took_time)
    # Used only to write output on console - can delete this later
    return {
        'took_time': took_time,
        'dataset': dataset.name(),
        'classifier': classifier.name(),
        'run_results': run_results,
        'eval_results_within_classifier': eval_results_within_classifier,
    }


classification_data_set = [
    Faults(),
    Adults(),
    SeismicBumps(),
    ThoracicSurgery(),
    CreditCard(),
    Diabetic(),
    BreastCancer(),
    AusCredit(),
    GermanCredit(),
    Yeast(),
    TwitterHate()
]

classifiers = [
    AdaBoostC(),
    LogisticRegressor(),
    MlpClassifier(),
    GaussianNB(),
    SupportVectorC(),
    DecisionTreeC(),
    RandomForestC(),
    KNearest()
]

classifier_evaluators = [
    PrecisionRecallEval()
]

regression_data_set = [
    RedWine(),
    WhiteWine(),
    AquaticToxicity(),
    ParkinsonSpeech(),
    FacebookMetrics(),
    CommunitiesAndCrime(),
    BikeSharing(),
    ConcreteCompressiveStrength(),
    StudentPerformance(),
    SGEMMPerformance(),
    MerckChallengeData1(),
    MerckChallengeData2()
]

regressors = [
    AdaBoostR(),
    SupportVectorR(),
    DecisionTreeR(),
    RandomForestR(),
    GaussianProcessR(),
    LinearR(),
    NeuralNetworkR()
]
regression_evaluators = [
    RegressionEval()
]

#run_pipeline(classification_data_set, classifiers, classifier_evaluators)
#run_pipeline(regression_data_set, regressors, regression_evaluators)
run_pipeline([Adults()],[SupportVectorC()], classifier_evaluators)