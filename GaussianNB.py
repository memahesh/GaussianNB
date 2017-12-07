import numpy as np
import pandas as pd





# Separating data into different classes
def SeparateByClass(data):
    # Separating the data into different classes based on last column value
    c1 = data.iloc[np.where(data.iloc[:,-1]==0)]
    c2 = data.iloc[np.where(data.iloc[:,-1]==1)]
    class_data = {
        "c1":c1,
        "c2":c2
    }
    return class_data

# Function to get the mean and standard deviation of all features of a particular class
def summaries(data):
    means = np.mean(data.iloc[:,1:-1])
    std = np.std(data.iloc[:, 1:-1])
    summary = {}
    summary["means"] = means
    summary["stds"] = std
    return summary

# Calculating class-wise probabilities
def ClassProbabilities(test_data, class_data):
    summary = {}
    # class_data.keys() gives first "c2" and then "c1". Because dictionary orders its keys alphabetically
    for item in class_data.keys():
        summary[item] = summaries(class_data[item])
    class_prob = np.zeros(shape=(test_data.shape[0],2))
    for i in range(test_data.shape[0]):
        probability =[1000]
        for item in summary.keys():
            # Gaussian Distribution Formula for Probability
            exponent = np.exp(-(test_data.iloc[i, 1:-1] - summary[item]["means"]) ** 2 / (2 * (summary[item]["stds"])**2))
            prob = (1.0 / (summary[item]["stds"] * np.sqrt(2 * 3.14))) * exponent
            prob = np.prod(prob)
            probability.append(prob)
        del probability[0]
        class_prob[i] = probability
    return class_prob

# Calculating Accuracy
def accuracy(class_prob, test_data):
    results = test_data.iloc[:,-1]
    pred_classes = []
    for i in class_prob:
        if i[0]>i[1]:
            pred = 1
        else:
            pred = 0
        pred_classes.append(pred)
    wrong_perc = (np.sum(abs(pred_classes-results))/float(test_data.shape[0]))*100
    acc = 100 - wrong_perc
    return acc

# Loading file
data = pd.read_csv('diabetes.csv', header=None)
# Removing any missing data rows in given dataset
data = data.dropna(how='any')
data = data.reset_index()
# Normalizing the dataset given
data.iloc[:,:-1] -= np.mean(data.iloc[:,:-1])
data.iloc[:,:-1] /= np.std(data.iloc[:,:-1])
# Splitting into training and test data
test_data = data.sample(frac=0.2)
data = data.drop(test_data.index)
# Separating data into different classes
class_data = SeparateByClass(data)
# Getting class-wise probabilities
class_prob = ClassProbabilities(test_data, class_data)
# Printing Accuracy
print "Accuracy is ", accuracy(class_prob, test_data)