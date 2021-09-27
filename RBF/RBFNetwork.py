from __future__ import division
import random
import math
from Pattern import Pattern
from Data import Data
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Test import Test

class RBFNetwork:
    def __init__(self, no_of_input, no_of_hidden, no_of_output, data, test): #*****
        self.no_of_input = no_of_input
        self.no_of_hidden = no_of_hidden
        self.no_of_output = no_of_output
        self.data = data
        self.input = np.zeros(self.no_of_input)
        self.centroid = np.zeros((self.no_of_hidden, self.no_of_input))
        self.sigma = np.zeros(self.no_of_hidden)
        self.hidden_output = np.zeros(self.no_of_hidden)
        self.hidden_to_output_weight = np.zeros((self.no_of_hidden, self.no_of_output))
        self.output = np.zeros(self.no_of_output)
        self.output_bias = np.zeros(self.no_of_output)
        self.actual_target_values = []
        self.total = 0
        self.learningRate = 0.0262
        self.setup_center()
        self.setup_sigma_spread_radius()
        self.set_up_hidden_to_ouput_weight()
        self.set_up_output_bias()
        self.test = test   # *****

    def setup_center(self):
        """Setup center using clustering ,for now just randomize between 0 and 1"""
        # print("Setup center")
        for i in range(self.no_of_hidden):
            self.centroid[i] = np.random.uniform(0, 1, self.no_of_input)

    def setup_sigma_spread_radius(self):
        # print("Setup Sigma spread radius")
        for i in range(self.no_of_hidden):
            center = self.centroid[i]
            self.sigma[i] = self.set_up_sigma_for_center(center)
            # print("Sigma i",i, self.sigma[i])

    def set_up_sigma_for_center(self, center):
        # print("Get sigma for center")
        p = self.no_of_hidden / 3
        sigma = 0
        distances = [0 for i in range(self.no_of_hidden)]
        for i in range(self.no_of_hidden):
            distances[i] = self.euclidean_distance(center, self.centroid[i])
            # print("Distance ", i, distances[i])
        sum = 0
        for i in range(int(p)):
            nearest = self.get_smallest_index(distances)
            distances[nearest] = float("inf")

            neighbour_centroid = self.centroid[nearest]
            for j in range(len(neighbour_centroid)):
                sum += (center[j] - neighbour_centroid[j]) ** 2

        sigma = sum / p
        sigma = math.sqrt(sigma)
        #return random.uniform(0, 1) * 6
        return sigma

    @staticmethod
    def euclidean_distance( x, y):
        return np.linalg.norm(x-y)

    @staticmethod
    def get_smallest_index( distances):
        min_index = 0
        for i in range(len(distances)):
            if (distances[min_index] > distances[i]):
                min_index = i
        return min_index

    def set_up_hidden_to_ouput_weight(self):
        #print("Setup hidden to output weight")
        self.hidden_to_output_weight = np.random.uniform(0, 1, (self.no_of_hidden, self.no_of_output))

        #print("Hiden to output weight ", self.hidden_to_output_weight)

    def set_up_output_bias(self):
        print("Setup output bias")
        self.output_bias = np.random.uniform(0, 1, self.no_of_output)

    # train n iteration
    def train(self, n):
        for i in range(n):
            error = self.pass_one_epoch()
            print("Iteration ", i, " Error ", error)
        """starts my part here"""
        weight_final=np.zeros((self.no_of_output,self.no_of_hidden))
        bias_final=np.zeros(self.no_of_output)
        centroids_final=np.zeros((self.no_of_hidden,self.no_of_input))
        sigma_final=np.zeros(self.no_of_hidden)
        """ends my part here"""
        weight_final=self.hidden_to_output_weight
        bias_final=self.output_bias
        centroids_final=self.centroid
        sigma_final=self.sigma
        return error,weight_final,bias_final,centroids_final,sigma_final

    # Train an epoch and return total MSE
    def pass_one_epoch(self):
        # print("Pass one epoch")
        all_error = 0
        all_index = []
        for i in range(len(self.data.patterns)):
            all_index.append(i)
        # print("All index ",all_index)

        for i in range(len(self.data.patterns)):
            random_index = (int)(random.uniform(0, 1) * len(all_index))
            # print("Random index ",random_index, " Len ", len(all_index))
            """Get a random pattern to train"""
            pattern = self.data.patterns[random_index]
            del all_index[random_index]

            input = pattern.input
            self.actual_target_values = pattern.output
            self.pass_input_to_network(input)

            error = self.get_error_for_pattern()
            all_error += error
            self.gradient_descent()

        all_error = all_error / (len(self.data.patterns))
        #return all_error,weight_final,bias_final,centroids_final,sigma_final
        return all_error
    def pass_input_to_network(self, input):
        self.input = input
        self.pass_to_hidden_node()
        self.pass_to_output_node()

    def pass_to_hidden_node(self):
        # print("Pass to hidden node")
        self.hidden_output = np.zeros(self.no_of_hidden)
        for i in range(len(self.hidden_output)):
            euclid_distance = self.euclidean_distance(self.input, self.centroid[i]) ** 2
            self.hidden_output[i] = math.exp(- (euclid_distance / (2 * self.sigma[i] * self.sigma[i])))

            # print("Hdiden node output ",self.hidden_output)

    def pass_to_output_node(self):
        # print("Pass to output node")
        self.output = [0 for i in range(self.no_of_output)]
        total = 0
        for i in range(self.no_of_output):
            output_value = 0
            for j in range(self.no_of_hidden):
                self.output[i] += self.hidden_to_output_weight[j][i] * self.hidden_output[j]

    # Compute error for the pattern
    def get_error_for_pattern(self):
        error = 0
        for i in range(len(self.output)):
            error += (self.actual_target_values[i] - self.output[i]) ** 2
        return error

    # Weight update by gradient descent algorithm
    def gradient_descent(self):
        # compute the error of output layer
        self.mean_error = 0
        self.error_of_output_layer = [0 for i in range(self.no_of_output)]
        for i in range(self.no_of_output):
            self.error_of_output_layer[i] = (float)(self.actual_target_values[i] - self.output[i])
            e = (float)(self.actual_target_values[i] - self.output[i]) ** 2 * 0.5
            self.mean_error += e

        # Adjust hidden to output weight
        for o in range(self.no_of_output):
            for h in range(self.no_of_hidden):
                delta_weight = self.learningRate * self.error_of_output_layer[o] * self.hidden_output[h]
                self.hidden_to_output_weight[h][o] += delta_weight

        # For bias
        for o in range(self.no_of_output):
            delta_bias = self.learningRate * self.error_of_output_layer[o]
            self.output_bias[o] += delta_bias

        # Adjust center , input to hidden weight
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)((self.input[i] - self.centroid[j][i]) / math.pow(self.sigma[j], 2))
                delta_weight = (float)(self.learningRate * self.hidden_output[j] * second_part * summ)
                self.centroid[j][i] += delta_weight

        # Adjust sigma and spread radius
        for i in range(self.no_of_input):
            for j in range(self.no_of_hidden):
                summ = 0
                for p in range(self.no_of_output):
                    summ += self.hidden_to_output_weight[j][p] * (self.actual_target_values[p] - self.output[p])

                second_part = (float)(
                    (math.pow((self.input[i] - self.centroid[j][i]), 2)) / math.pow(self.sigma[j], 3));
                delta_weight = (float)(0.1 * self.learningRate * self.hidden_output[j] * second_part * summ);
                self.sigma[j] += delta_weight

        """My part starts here"""
        return self.mean_error,self.hidden_to_output_weight,self.output_bias,self.centroid,self.sigma

        """My part ends here"""
        #return self.mean_error

    def get_accuracy_for_training(self):
        correct = 0
        for i in range(len(self.data.patterns)):
            pattern = self.data.patterns[i]
            self.pass_input_to_network(pattern.input)
            n_output = self.output
            act_output = pattern.output
            a_neuron = act_output
            n_neuron = self.do_some_rounding(n_output)
            print("output",n_neuron,"label",a_neuron)
            if n_neuron == a_neuron:
                correct += 1
        accuracy = (float)(correct / len(self.data.patterns)) * 100
        return accuracy

    """my test begins"""
    def get_accuracy_for_testing(self):
        for i in range(len(self.test)):
            test_pattern = self.test[i]
            self.pass_input_to_network(test_pattern.input)
            pred_output = self.output
            pred_output = do_some_rounding(pred_output)
            print("pred_output",pred_output)
        return pred_output
    """my test ends"""

    def do_some_rounding(self,output):
        if output[0]>0:
            round=1
        else:
            round=-1
        return round

def do_some_rounding(output):
    if output[0]>0:
        round=1
    else:
        round=-1
    return round

"""Create training data """
dataArr=loadmat('../Data/data_train.mat')
labelArr=loadmat('../Data/label_train.mat')
data_name=list(dataArr.keys())[-1]
label_name=list(labelArr.keys())[-1]
dataArr=dataArr[data_name]
labelArr=labelArr[label_name]
dataArr=np.array(dataArr)
labelArr=np.array(labelArr)
MyPatterns=[]
createVar=locals()
for i in range(330):
    createVar['p'+str(i)]=Pattern(i,dataArr[i,:],labelArr[i,:])
    MyPatterns.append(createVar['p'+str(i)])
"""create test data"""
testArr=loadmat('../Data/data_test.mat')
test_name=list(testArr.keys())[-1]
testArr=testArr[test_name]
testArr=np.array(testArr)
testPatterns=[]
testVar=locals()
for i in range(21):
    testVar['t'+str(i)]=Test(i,testArr[i])
    testPatterns.append(testVar['t'+str(i)])
"""training set"""
patterns=MyPatterns
data = Data(patterns)
test = testPatterns
rbf = RBFNetwork(33, 30, 1, data, test)
mse,we,bi,ce,si = rbf.train(100)
accuracy = rbf.get_accuracy_for_training()
print("Total accuracy is ", accuracy)
print("Last MSE ",mse)
"""testing set"""
pred_output=rbf.get_accuracy_for_testing()
