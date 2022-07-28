from typing import List, Optional
import numpy as np

class NN():
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int) -> None:
        self.mean, self.std_dev = 0.0, 1.0
        self.num_inputs, self.num_hidden, self.num_outputs = num_inputs, num_hidden, num_outputs
        self.hidden_weights, self.output_weights = self.randomWeights()

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X is (num_inputs,) size array
        # print("X: ", X.shape)
        # Add bias term
        b0 = np.hstack((X, [1]))
        # print("b0: ", b0.shape)
        # Hidden layer
        f1 = b0.dot(self.hidden_weights)
        # print("f1: ", f1.shape)
        a1 = self.activation(f1)
        # print("a1: ", a1.shape)
        b1 = np.hstack((a1, [1]))
        # print("b1: ", b1.shape)
        # Output layer
        f2 = b1.dot(self.output_weights)
        # print("f2: ", f2.shape)
        return self.activation(f2)

    def getWeights(self) -> List[np.ndarray]:
        return [self.hidden_weights, self.output_weights]

    def setWeights(self, weights: List[np.ndarray]):
        # Check dimensions
        if weights[0].shape != self.hidden_weights.shape or weights[1].shape != self.output_weights.shape:
            raise Exception("Weights are being set incorrectly in setWeights().\n"\
                            "Dimensions for new weights and network weights do not match!\n"\
                            "New weights dim != Network weights dim\n"\
                            f"[{weights[0].shape},{weights[1].shape}] != [{self.hidden_weights.shape},{self.output_weights.shape}]")
        # Set weights
        self.hidden_weights, self.output_weights = weights
        return None

    def activation(self, arr: np.ndarray) -> np.ndarray:
        return np.tanh(arr)

    def randomWeights(self):
        return [
            np.random.normal(self.mean, self.std_dev, size=(self.num_inputs+1, self.num_hidden)),
            np.random.normal(self.mean, self.std_dev, size=(self.num_hidden+1, self.num_outputs))
        ]

    def shape(self):
        return (self.num_inputs, self.num_hidden, self.num_outputs)

def createNNfromWeights(weights: List[np.ndarray]):
    num_inputs = weights[0].shape[0]-1
    num_hidden = weights[0].shape[1]
    num_outputs = weights[1].shape[1]
    net = NN(num_inputs, num_hidden, num_outputs)
    net.setWeights(weights)
    return net

if __name__ == "__main__":
    nn = NN(num_inputs=4, num_hidden=10, num_outputs=2)
    Y = nn.forward(np.array([1,2,3,4]))
    print(Y)
