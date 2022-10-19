import numpy as np
import torch
import os
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
import PIL.Image as pilimg
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler,normalize
from datasets import Cifar
#import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
import pydotplus
from sklearn import tree
#from IPython.display import Image

path = f'{os.path.dirname(os.path.realpath(__file__))}/../Output/Activation'

class CNN:
    def __init__(self):
        cifar = Cifar()
        self.train_data_set = cifar.getXTrain()
        self.train_data_label = cifar.getYTrain()
        self.test_data_set = cifar.getXTest()
        self.test_data_label = cifar.getYTest()
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_data_scaled = None
        self.test_data_scaled = None
        self.train_data_tensor = None
        self.test_data_tensor = None
        self.train_label_tensor = None
        self.test_label_tensor = None


    def preprocess_data(self):
        scaler = StandardScaler().fit(self.train_data_set)
        train_data_scaled = scaler.transform(self.train_data_set).astype(np.float32).reshape(-1, 3, 32, 32)
        test_data_scaled = scaler.transform(self.test_data_set).astype(np.float32).reshape(-1, 3, 32, 32)

        # Convert scaled data to tensors
        self.train_data_tensor = torch.from_numpy(train_data_scaled)
        self.test_data_tensor = torch.from_numpy(test_data_scaled)
        self.train_label_tensor = torch.from_numpy(self.train_data_label).type(torch.long)
        self.test_label_tensor = torch.from_numpy(self.test_data_label).type(torch.long)
        print("I scaled the data")

    def read_model_from_file(self):
        print()

    def train_model(self, model):
        torch.manual_seed(0)
        print("I train the model")
        #torch.manual_seed(0)
        batch_size = 500
        num_epoch = 10

        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        file = open('CNN_loss_and_accuracy_5filters.txt', 'w')
        for epoch in range(1, num_epoch + 1):
            for i in range(0, len(self.train_data_tensor), batch_size):
                X = self.train_data_tensor[i:i + batch_size]
                y = self.train_label_tensor[i:i + batch_size]

                y_pred = model(X)
                # print(y_pred)
                l = loss(y_pred, y)

                model.zero_grad()
                l.backward()
                optimizer.step()

            _, predicted = torch.max(y_pred.data, 1)
            accuracy = (predicted == y).sum()
            print("Epoch %d final minibatch had loss %.4f and accuracy %.4f" % (
            epoch, l.item(), 100 * accuracy / X.shape[0]))
            file.write("Epoch %d final minibatch had loss %.4f and accuracy %.4f\n" % (
            epoch, l.item(), 100 * accuracy / X.shape[0]))
        file.close()
        return model

    def test_model(self, model):
        total_correct = 0
        total_images = 0
        confusion_matrix = np.zeros([10, 10], int)
        with torch.no_grad():
            test_output = model(self.test_data_tensor)
            _, predicted = torch.max(test_output.data, 1)
            total_images += self.test_label_tensor.size(0)
            total_correct += (predicted == self.test_label_tensor).sum().item()
            for i, l in enumerate(self.test_label_tensor):
                confusion_matrix[l.item(), predicted[i].item()] += 1

        model_accuracy = total_correct / total_images * 100
        print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))

    def define_model(self):

        filter_size = 5

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=filter_size // 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 5, padding=filter_size // 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, 5, padding=filter_size // 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(2048,1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
        return model

    def activation_maximization(self, class_index, model):
        torch.manual_seed(0)
        print("I maximize the input image")
        def f(x):
            mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            mask[class_index] = 1
            return model(x) * torch.tensor(mask)

        # x = torch.zeros((1,3, 32,32),requires_grad=True)
        x = torch.rand((1, 3, 32, 32), requires_grad=True)

        for i in range(10000):
            y = f(x)
            # l = loss(y, tensor_train_label[0:500])
            y.backward(torch.ones(1, 10))
            x.data += 0.02 * x.grad.data
            x.grad.data.zero_()
        ximage = x.detach().numpy()

        print(self.classes[class_index])
        fig, ax = plt.subplots()
        ax.imshow(ximage[0][0], cmap='gray')
        # stdscaler= StandardScaler().fit(ximage)
        # ximage = stdscaler.transform(ximage)
        # ximage = Image(ximage).convert('LA')
        #ximage = np.column_stack([n1,n2,n3])
        #ximage = ximage.reshape(3, 32, 32).transpose(1,2,0)
        #ax.imshow(ximage.reshape(3, 32, 32), cmap='gray')
        fig.savefig(path+self.classes[class_index]+'filter5.png')

    def main(self):
        self.preprocess_data()
        '''if not os.path.exists('/model'):
            os.makedirs('/model', 0o777)
            model = self.define_model()
            model_trained = self.train_model(model)
            torch.save(model_trained, '/model/model.pth')'''
        model = self.define_model()
        model_trained = self.train_model(model)

        #trained_model = torch.load('/model/model.pth')
        self.test_model(model_trained)
        for class_index in range(10):
            self.activation_maximization(class_index, model_trained)


class DecisionTree:
    def __init__(self):
        cifar_dec = Cifar()
        self.train_set = cifar_dec.getXTrain()
        self.train_label = cifar_dec.getYTrain()
        self.test_set = cifar_dec.getXTest()
        self.test_label = cifar_dec.getYTest()

    def data_scale(self):
        scaler = StandardScaler().fit(self.train_set)
        self.train_set = scaler.transform(self.train_set)
        self.test_set = scaler.transform(self.test_set)

    def train_decision_tree(self):
        dec_tree = DecisionTreeClassifier(random_state=0, max_depth=10000, min_samples_split=100).fit(self.train_set,self.train_label)
        return dec_tree

    def run_dec_tree_classifier(self):
        self.data_scale()

        print(plot_tree(self.train_decision_tree()))
        print(self.train_decision_tree().score(self.test_set, self.test_label))
        '''dot_data = tree.export_graphviz(clf, out_file=None)

        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Show graph
        Image(graph.create_png())'''



cnn = CNN()
cnn.main()

#dec_tree = DecisionTree()
#dec_tree.run_dec_tree_classifier()