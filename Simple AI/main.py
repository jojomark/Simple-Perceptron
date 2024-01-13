"""importing pandas, random and numpy"""
import pandas as pd
import random
import numpy as np

"""The perceptron class"""
class Perceptron():
    def __init__(self, num_features, learning_rate):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_features) * 10
        self.bias = random.random()*1000

    """Basically a sign step function """
    def sign(self, x):
        return 1 if x >= 0 else 0

    """The perceptron prediction, which has two use cases:
    1: You want to predict something with the perceptron. Returns 0 or 1
    2: You want to predict something with the perceptron, then check if the prediction is correct. Returns the error"""
    def predict(self, inputs, classification=None):
        activation = np.dot(inputs, self.weights) + self.bias
        if classification is not None:
            return classification - self.sign(activation) #Returns 1 or -1 if incorrect, 0 if correct
        else:
            return self.sign(activation)
                
    """The perceptron training, which has two use cases:
    1: You want to train until you have a 99% success rate 
    2: You want to train a certain number of times. """
    def train(self, inputs, classification, epochs=None): 
        if epochs:
            for _ in range(epochs):
                for x, y in zip(inputs, classification):
                    error = self.predict(x, y)
                    if error:
                        self.weights += error * self.learning_rate * x
                        self.bias += error * self.learning_rate
        else:
            mistakes = len(classification)
            one_percent = len(classification)/100
            while mistakes > one_percent:
                mistakes = 0
                for x, y in zip(inputs, classification):
                    error = self.predict(x, y)
                    if error:
                        self.weights += error * self.learning_rate * x
                        self.bias += error * self.learning_rate
                        mistakes += 1
            with open("weights.txt", "a") as f:
                f.write(f"Weights: {self.weights}\nBias: {self.bias}")
                

                    
if __name__ == "__main__":
    df = pd.read_excel("X.xlsx", sheet_name=0, header=None)
    df.rename(columns={0:"Height", 1:"Weight"}, inplace=True)

    df2 = pd.read_excel("Y.xlsx", sheet_name=0, header=None)


    X = df.to_numpy()
    Y = df2.to_numpy()

    p = Perceptron(2, 0.01)
    p.train(X, Y)

    mistakes = 0
    for x, y in zip(X, Y):
        prediction = p.predict(x)
        if prediction != y:
            mistakes += 1
    print(mistakes)
        
   
