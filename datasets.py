import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pandas as pd
from tempfile import TemporaryFile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging

import os

path = f'{os.path.dirname(os.path.realpath(__file__))}/../../Project/DataSet/'


class AbstractDataset:
    def __init__(self) -> None:
        self.raw_data = None
        self.string_data_raw = None
        self.raw_labels = None
        self.cont_data_raw = None
        self.missing_data = None

    def name(self):
        raise NotImplemented()

    def get_params(self):
        raise NotImplemented()

    def get_raw_data(self):
        if not self.raw_data:
            if 'io' in self.get_params():
                pd_data = pd.read_excel(**self.get_params())
                self.raw_data = np.array(pd_data)
            else:
                pd_data = pd.read_csv(**self.get_params())
                pd_data.replace('', np.nan, inplace=True)
                pd_data.replace('?', np.nan, inplace=True)
                self.raw_data = np.array(pd_data)

        return self.raw_data

    def handle_missing_values(self):
        # substituting the missing nan values with the mean of the column
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp_mean.fit(self.missing_data)
        return imp_mean.transform(self.missing_data)

    def getXTrain(self):
        raise NotImplemented()

    def getYTrain(self):
        raise NotImplemented()

    def get_one_hot_encoding(self):
        one_hot_encoder = OneHotEncoder(categories=self.define_categories(), handle_unknown='ignore')
        return one_hot_encoder.fit_transform(self.string_data_raw)

    def get_label_encoded(self):
        le = LabelEncoder()
        return le.fit_transform(self.raw_labels)

    def define_categories(self):
        raise NotImplemented()

    def get_data_scaled(self, Xtrain):
        scaler = StandardScaler()
        return scaler.fit(Xtrain)

    def getXTest(self):
        raise NotImplemented()

    def getYTest(self):
        raise NotImplemented()


    def KfFoldData(self, n_folds):
        kf = KFold(n_splits=n_folds)
        X = self.getXTrain()
        y = self.getYTrain()
        for train_index, test_index in kf.split(X):
            XTrain, XTest = X[train_index], X[test_index]
            yTrain, yTest = y[train_index], y[test_index]
            yield XTrain, XTest, yTrain, yTest

    def loadMerckData(self, file_name):
        with open(file_name) as f:
            cols = f.readline().rstrip('\n').split(',')  # Read the header line and get list of column names
        # Load the actual data, ignoring first column and using second column as targets.
        X = np.loadtxt(file_name, delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8)
        y = np.loadtxt(file_name, delimiter=',', usecols=[1], skiprows=1)
        return X, y

    def saveArrayToFile(self, X, y):
        outfile = TemporaryFile()
        np.savez(outfile, x=X, y=y)
        return outfile

    def unpickle(self,file):
        with open(file,'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict


class Cifar(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data_batch_1 = self.unpickle(path + 'Interpretability/cifar-10-python/data_batch_1')
        self.data_batch_2 = self.unpickle(path + 'Interpretability/cifar-10-python/data_batch_2')
        self.data_batch_3 = self.unpickle(path + 'Interpretability/cifar-10-python/data_batch_3')
        self.data_batch_4 = self.unpickle(path + 'Interpretability/cifar-10-python/data_batch_4')
        self.data_batch_5 = self.unpickle(path + 'Interpretability/cifar-10-python/data_batch_5')
        self.test_batch = self.unpickle(path + 'Interpretability/cifar-10-python/test_batch')

    def name(self):
        return "CIFAR_10"

    def getXTrain(self):
        self.cont_data_raw = np.vstack((self.data_batch_1[b'data'], self.data_batch_2[b'data'], self.data_batch_3[b'data'],
                                        self.data_batch_4[b'data'], self.data_batch_5[b'data']))
        return self.cont_data_raw

    def getXTest(self):
        return self.test_batch[b'data']

    def getYTrain(self):
        return np.hstack((np.array(self.data_batch_1[b'labels']), np.array(self.data_batch_2[b'labels']), np.array(self.data_batch_3[b'labels']),
                          np.array(self.data_batch_4[b'labels']), np.array(self.data_batch_5[b'labels'])))

    def getYTest(self):
        return np.array(self.test_batch[b'labels'])


class Diabetic(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Diabetic"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/Diabetic/messidor_features.arff',
            'sep': ',',
            'skiprows': 24
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, :-1]
        scale = self.get_data_scaled(self.cont_data_raw)
        return scale.transform(self.cont_data_raw)

    def getYTrain(self):
        return self.raw_data[:, -1]


# ToDo read_csv cannt read the xls file
class CreditCard(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Credit_Card"

    def get_params(self):
        return {
            'io': path + 'Classification/CreditCard/default of credit card clients.xls',
            'skiprows': 2
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, :-1]
        scale = self.get_data_scaled(self.cont_data_raw)
        return scale.transform(self.cont_data_raw)

    def getYTrain(self):
        return self.raw_data[:, -1]


class BreastCancer(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Breast_Cancer"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/BreastCancer/breast-cancer-wisconsin.data',
            'sep': ',',
            'skiprows': 0
        }

    def getXTrain(self):
        self.missing_data = self.raw_data[:, 1:-1]
        self.cont_data_raw = self.handle_missing_values().astype("float")
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1].astype('int')


class AusCredit(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Australian_credit_card"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/AustralianCredit/australian.dat',
            'sep': ' '
        }

    def define_categories(self):
        A1 = [0, 1]
        A4 = [1, 2, 3]
        A5 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        A6 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        A8 = [0, 1]
        A9 = [0, 1]
        A11 = [0, 1]
        A12 = [1, 2, 3]

        return [A1, A4, A5, A6, A8, A9, A11, A12]

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, [1, 2, 6, 9, 12, 13]].astype(np.float)
        self.string_data_raw = self.raw_data[:, [0, 3, 4, 5, 7, 8, 10, 11]].tolist()
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))

    def getYTrain(self):
        return self.raw_data[:, -1]


class GermanCredit(AbstractDataset):
    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "German_credit_card"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/GermanCredit/german.data',
            'sep': ' '
        }

    def define_categories(self):
        checking_acc = ['A11', 'A12', 'A13', 'A14']
        credit_history = ['A30', 'A31', 'A32', 'A33', 'A34']
        purpose = ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410']
        saving_acc = ['A61', 'A62', 'A63', 'A64', 'A65']
        present_employment = ['A71', 'A72', 'A73', 'A74', 'A75']
        status_sex = ['A91', 'A92', 'A93', 'A94', 'A95']
        guarantors = ['A101', 'A102', 'A103']
        property = ['A121', 'A122', 'A123', 'A124']
        installment_plans = ['A141', 'A142', 'A143']
        housing = ['A151', 'A152', 'A153']
        job = ['A171', 'A172', 'A173', 'A174']
        telephone = ['A191', 'A192']
        foreign_worker = ['A201', 'A202']

        return [checking_acc, credit_history, purpose, saving_acc, present_employment, status_sex, guarantors,
                property, installment_plans, housing, job, telephone, foreign_worker]

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, [1, 4, 7, 10, 12, 15, 17]].astype(np.float)
        self.string_data_raw = self.raw_data[:, [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]].tolist()
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))

    def getYTrain(self):
        return self.raw_data[:, -1].astype('int')


class SeismicBumps(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Seismic_Bumps"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/seismic-bumps.arff',
            'sep': ',',
            'skiprows': 154
        }

    def define_categories(self):
        seismic = ['a', 'b', 'c', 'd']
        seismoacoustic = ['a', 'b', 'c', 'd']
        shift = ['W', 'N']
        ghazard = ['a', 'b', 'c', 'd']

        return [seismic, seismoacoustic, shift, ghazard]

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].astype('float')
        self.string_data_raw = self.raw_data[:, [0, 1, 2, 7]].tolist()
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))
        # return np.column_stack((self.get_data_scaled(), data_string_encoded.toarray()))

    def getYTrain(self):
        return self.raw_data[:, -1].astype('int')


class Faults(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Faults"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/Faults.NNA',
            'sep': '\t'
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, :27]
        return self.cont_data_raw

    def getYTrain(self):
        return np.array([list(row).index(1) for row in self.raw_data[:, 27:]])


class Adults(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Adults"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/Adults/adult.data',
            'sep': ', '
        }

    def define_categories(self):
        workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                     'Without-pay', 'Never-worked']
        education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                     '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
        marital_status = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                          'Married-spouse-absent',
                          'Married-AF-spouse']
        occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                      'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                      'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
        relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
        sex = ['Female', 'Male']
        native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                          'Outlying-US(Guam-USVI-etc)',
                          'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
                          'Italy', 'Poland',
                          'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos',
                          'Ecuador',
                          'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
                          'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
        return [workclass, education, marital_status, occupation, relationship, race, sex, native_country]

    def getXTrain(self):
        self.missing_data = self.raw_data[:,0:-1]
        self.cont_data_raw = self.handle_missing_values()[:, [0, 2, 4, 10, 11, 12]].astype(np.float)
        self.string_data_raw = self.handle_missing_values()[:, [1, 3, 5, 6, 7, 8, 9, 13]].tolist()
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))

    def getYTrain(self):
        self.raw_labels = self.raw_data[:, -1]
        return self.get_label_encoded()


class Yeast(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Yeast"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/Yeast/yeast.data',
            'delim_whitespace': True
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 1:9].astype('float')
        print("")
        return self.cont_data_raw

    def getYTrain(self):
        self.raw_labels = self.raw_data[:, -1]
        return self.get_label_encoded()


class ThoracicSurgery(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Thoracic_Surgery"

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Classification/ThoraricSurgery.arff',
            'sep': ',',
            'skiprows': 21
        }

    def define_categories(self):
        DGN = ['DGN3', 'DGN2', 'DGN4', 'DGN6', 'DGN5', 'DGN8', 'DGN1']
        PRE6 = ['PRZ2', 'PRZ1', 'PRZ0']
        PRE7 = ['T', 'F']
        PRE8 = ['T', 'F']
        PRE9 = ['T', 'F']
        PRE10 = ['T', 'F']
        PRE11 = ['T', 'F']
        PRE14 = ['OC11', 'OC14', 'OC12', 'OC13']
        PRE17 = ['T', 'F']
        PRE19 = ['T', 'F']
        PRE25 = ['T', 'F']
        PRE30 = ['T', 'F']
        PRE32 = ['T', 'F']

        return [DGN, PRE6, PRE7, PRE8, PRE9, PRE10, PRE11, PRE14, PRE17, PRE19, PRE25, PRE30, PRE32]

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, [1, 2, 15]].astype('float')
        self.string_data_raw = self.raw_data[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].tolist()
        string_data_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, string_data_encoded.toarray()))

    def getYTrain(self):
        self.raw_labels = self.raw_data[:, -1]
        return self.get_label_encoded()


class RedWine(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Red_Wine"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/Wine Quality/winequality-red.csv',
            'sep': ';',
            'header': 0
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 0:-1]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1]


class WhiteWine(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "White_Wine"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/Wine Quality/winequality-white.csv',
            'sep': ';',
            'header': 0
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 0:-1]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1]


class AquaticToxicity(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.raw_data = np.array(self.get_raw_data())

    def name(self):
        return "Aquatic_Toxicity"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/QSAR Aquatic toxicity/qsar_aquatic_toxicity.csv',
            'sep': ';',
            'header': None
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 0:-1]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1]


class ParkinsonSpeech(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Parkinsons_Speech"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/Parkinson Speech/train_data.txt',
            'sep': ',',
            'header': None
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 1:-2]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -2]


class FacebookMetrics(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Facebook_Metrics"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/Facebook metrics/dataset_Facebook.csv',
            'sep': ';',
            'header': 0
        }

    def define_categories(self):
        fb_type = ['Link', 'Photo', 'Status', 'Video']
        return [fb_type]

    def getXTrain(self):
        self.missing_data = self.raw_data[:, [i for i in range(19) if i != 1]].astype('float')
        self.cont_data_raw = self.handle_missing_values().astype('float')
        self.string_data_raw = self.raw_data[:, 1].reshape(-1, 1)
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))

    def getYTrain(self):
        return self.raw_data[:, -1]


class CommunitiesAndCrime(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Communities_and_Crime"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/Communities and Crime/communities.data',
            'sep': ',',
            'header': None
        }

    def getXTrain(self):
        self.missing_data = self.raw_data[:, 5:].astype('float')
        self.cont_data_raw = self.handle_missing_values().astype('float')
        return self.cont_data_raw[:, 0:-1]

    def getYTrain(self):
        return self.raw_data[:, -1]


class DummyDataset(AbstractDataset):
    feature_size = 10
    data_size_train = 100
    data_size_test = 100

    def name(self):
        return 'I am a dummy dataset'

    def getXTrain(self):
        return np.random.rand(DummyDataset.feature_size, DummyDataset.data_size_train)

    def getYTrain(self):
        return np.random.rand(DummyDataset.data_size_train)

    def getYTest(self):
        return np.random.rand(DummyDataset.data_size_test)


class BikeSharing(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Bike_Sharing"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/BikeSharing/hour.csv',
            'sep': ',',
            'header': 0
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 2:-3]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1]


class StudentPerformance(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Student_Performance"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/StudentPerformance/student-mat.csv',
            'sep': ';',
            'header': 0
        }

    def define_categories(self):
        std_school = ['GP','MS']
        std_sex = ['M','F']
        std_address = ['U','R']
        std_famsize = ['GT3','LE3']
        std_Pstatus = ['T','A']
        std_Mjob = ['at_home','health','other','services','teacher']
        std_Fjob = ['at_home','health','other','services','teacher']
        std_reason = ['course','home','other','reputation']
        std_gaurdian = ['father','mother','other']
        std_schoolsup = ['yes','no']
        std_famsup = ['yes','no']
        std_paid = ['yes','no']
        std_activities = ['yes','no']
        std_nursery = ['yes','no']
        std_higher = ['yes','no']
        std_internet = ['yes','no']
        std_romantic = ['yes','no']

        return [std_school, std_sex,  std_address, std_famsize, std_Pstatus, std_Mjob,
                std_Fjob, std_reason, std_gaurdian, std_schoolsup, std_famsup, std_paid,
                std_activities, std_nursery, std_higher,std_internet,std_romantic]

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, [2,6,7,12,13,14,23,24,25,26,27,28,29,30,31]].astype('float')
        self.string_data_raw = self.raw_data[:, [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]].tolist()
        data_string_encoded = self.get_one_hot_encoding()
        return np.column_stack((self.cont_data_raw, data_string_encoded.toarray()))

    def getYTrain(self):
        return self.raw_data[:,32]


class ConcreteCompressiveStrength(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "Concrete_Compressive_Strength"

    def get_params(self):
        return {
            'io': path + '/Regression/ConcreteCompressiveStrength/Concrete_Data.xls',
            'skiprows': 1
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 0:-1]
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:, -1]


class SGEMMPerformance(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()

    def name(self):
        return "SGEMM_Performance"

    def get_params(self):
        return {
            'filepath_or_buffer': path + '/Regression/SGEMM/sgemm_product.csv',
            'sep': ',',
            'header': 0
        }

    def getXTrain(self):
        self.cont_data_raw = self.raw_data[:, 0:14]
        return self.cont_data_raw

    def getYTrain(self):
        return np.mean(self.raw_data[:, 14:18], axis=1)


class MerckChallengeData1(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        X, y = self.loadMerckData(path+'/Regression/Merck/ACT2_competition_training.csv')
        outfile = self.saveArrayToFile(X,y)
        _ = outfile.seek(0)
        self.merck_file = np.load(outfile)

    def name(self):
        return "Merck_Challenge_Dataset1"

    def getXTrain(self):
        return self.merck_file['x']

    def getYTrain(self):
        return self.merck_file['y']


class MerckChallengeData2(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        X, y = self.loadMerckData(path+'/Regression/Merck/ACT4_competition_training.csv')
        outfile = self.saveArrayToFile(X,y)
        _ = outfile.seek(0)
        self.merck_file = np.load(outfile)

    def name(self):
        return "Merck_Challenge_Dataset2"

    def getXTrain(self):
        return self.merck_file['x']

    def getYTrain(self):
        return self.merck_file['y']


class TwitterHate(AbstractDataset):

    def __init__(self) -> None:
        super().__init__()
        self.get_raw_data()
        self.get_embed()

    def name(self):
        return 'Twitter_Hate_Speech'

    def get_params(self):
        return {
            'filepath_or_buffer': path + 'Novelty_component/train_en.tsv',
            'sep': '\t',
            'usecols': ['id', 'text', 'HS']
        }
    def get_embed(self):
        params = {
            'dm': 1,
            'alpha': 0.05,
            'vector_size': 100,
            'window': 5,
            'epochs': 10,
            'min_count': 1,
            'sample': 1e-4,
            'hs': 0,
            'negative': 5,
            'workers': 8,
            'seed': 100,
        }
        logging.basicConfig(format=f'%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        documents = [TaggedDocument(row[1].lower().split(' '), [str(row[0])]) for row in self.raw_data]
        model = Doc2Vec(documents, **params)
        #model.save('./Model/doc_embed.model')
        self.cont_data_raw = model.docvecs.vectors_docs


    def getXTrain(self):
        #model = Doc2Vec.load('./Model/doc_embed.model')
        #self.cont_data_raw = model.docvecs.vectors_docs
        return self.cont_data_raw

    def getYTrain(self):
        return self.raw_data[:,-1].astype('int')


data = TwitterHate()
print(data.getXTrain())
# print(data.getYTrain().shape)
