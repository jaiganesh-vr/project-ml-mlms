'''
This script consists of classes general and specific classes for each dataset to do relevant preprocessing before
returning the training and test datasets required for the model to train upon.
'''

import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

path = '../../Project/DataSet/'


class General:
    def __init__(self, params):
        self.data = np.genfromtxt(**params)

    def get_data(self):
        return np.array(self.data.tolist())

    def get_label_counts(self, labels):
        print(Counter(labels).keys())
        print(Counter(labels).values())

    def get_label_encoded(self, labels):
        le = LabelEncoder()
        y = le.fit_transform(labels)
        return y

    def get_one_hot_encoding(self, features, feature_values):
        one_hot_encoder = OneHotEncoder(categories=features, handle_unknown='ignore')
        return one_hot_encoder.fit_transform(feature_values)

    def get_data_scaled(self, data):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        return scaled

class Faults:
    def __init__(self):
        self.params = {
            'fname': path+'Classification/Faults.NNA',
            'dtype': None
        }

    def get_data(self):
        general = General(self.params)
        cont_data_raw = general.get_data()[:, :27]
        label_raw = (general.get_data()[:, 27:])

        X = general.get_data_scaled(cont_data_raw)

        # Converts one hot vector of y label into a single value as class
        # the position of the one hot vector is the class value of the particular row
        y = [list(row).index(1)for row in label_raw]
        print(general.get_label_counts(y))
        return X, y



class Adults:
    # https://archive.ics.uci.edu/ml/datasets/Adult
    def __init__(self):
        self.params = {
            'fname': path+'Classification/Adults/adult.data',
            'delimiter': ', ',
            'dtype': np.dtype([('f0', 'U2'), ('f1', 'U20'), ('f2', 'U10'), ('f3', 'U15'),
                                ('f4', 'U10'), ('f5', 'U25'), ('f6', 'U17'), ('f7', 'U15'),
                                ('f8', 'U20'), ('f9', 'U6'), ('f10', 'U10'), ('f11', 'U10'),
                                ('f12', 'U3'), ('f13', 'U25'), ('f14', 'U5')])

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
                          'Yugoslavia','El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
        return [workclass, education, marital_status, occupation, relationship, race, sex, native_country]

    def get_data(self):
        general= General(self.params)
        data = general.get_data()

        # Extracting data
        cont_data_raw = data[:, [0, 2, 4, 10, 11, 12]].astype(np.float)
        string_data_raw = data[:, [1, 3, 5, 6, 7, 8, 9, 13]].tolist()
        label_raw = data[:, 14]

        # Convert data
        mod_data_string_encoded = general.get_one_hot_encoding(self.define_categories(), string_data_raw)
        X = np.column_stack((cont_data_raw, mod_data_string_encoded.toarray()))

        # Use label encoder for the class label
        y = general.get_label_encoded(label_raw)
        return X, y

class Yeast:
    def __init__(self):
        self.params = {
            'fname': path + 'Classification/Yeast/yeast.data',
            'dtype': np.dtype([('f0', 'U12'), ('f1', 'f8'), ('f2', 'f8'), ('f3', 'f8'),
                                ('f4', 'f8'), ('f5', 'f8'), ('f6', 'f8'), ('f7', 'f8'),
                                ('f8', 'f8'), ('f9', 'U4')])
        }

    def get_data(self):
        general = General(self.params)
        data = general.get_data()

        # Excluded the first column values since they are just unique database values
        # Nothing to generalize
        # No Scaling required since all the attribute values fall between 0 and 1
        X = data[:, 1:9].astype('float')
        raw_label = data[:, -1]
        y = general.get_label_encoded(raw_label)
        return X, y


class ThoracicSurgery:
    def __init__(self):
        self.param = {
            'fname': path + 'Classification/ThoraricSurgery.arff',
            'delimiter': ',',
            'skip_header': 21,
            'dtype': np.dtype('U7')
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

    def get_data(self):
        general = General(self.param)
        data = general.get_data()

        # Extract data
        cont_data_raw = data[:, [1, 2, 15]].astype('float')
        string_data_raw = data[:, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].tolist()
        label_raw = data[:, -1]

        # Convert data
        mod_data_string_encoded = general.get_one_hot_encoding(self.define_categories(), string_data_raw)
        scaled_data_cont = general.get_data_scaled(cont_data_raw)
        X = np.column_stack((mod_data_string_encoded.toarray(), scaled_data_cont))
        y = general.get_label_encoded(label_raw)
        return X, y


class SeismicBumps:
    def __init__(self):
        self.params = {
            'fname': path + 'Classification/seismic-bumps.arff',
            'dtype': np.dtype('U15'),
            'delimiter': ',',
            'skip_header': 154
        }

    def get_data(self):
        general = General(self.params)
        data = general.get_data()
        cont_data_raw = data[:, [3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]].astype('float')
        string_data_raw = data[:, [0, 1, 2, 7]]
        y = data[:, -1].astype('int')

        print(data)

class Facebook():
    def __init__(self):
        self.params = {
            'fname': path + 'Regression/Facebook metrics/dataset_Facebook.csv',
            'dtype': np.dtype('U10'),
            'delimiter': ';',
            'skip_header': 1
    }

    def get_data(self):
        general = General(self.params)
        data = general.get_data()
        np.delete(data, -1)
        cont_data_raw = data[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]].astype('float')
        '''for i in range(0, 500):
            cont_data_raw = data[i, [0, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
            try:
                list1 = [float(x) for x in cont_data_raw]
            except ValueError as e:
                print("error", e, "on line", i)'''



data = Facebook()
data.get_data()

