from typing import List, Optional, Union, Tuple
import numpy as np

class NN():
    def __init__(self, num_inputs: int, num_hidden: Union[int, List[int]], num_outputs: int) -> None:
        self.mean, self.std_dev = 0.0, 1.0
        if type(num_hidden) == int:
            num_hidden = [num_hidden]
        self.num_inputs, self.num_hidden, self.num_outputs = num_inputs, num_hidden, num_outputs
        # Number of nodes in each layer
        self.shape = tuple([self.num_inputs]+self.num_hidden+[self.num_outputs])
        # Number of layers
        self.num_layers = len(self.shape)-1
        self.weights = self.randomWeights()
        self.weights_shape = calculateWeightShape(self.weights)
        self.total_weights = calculateWeightSize(self.weights)

    def randomWeights(self)->List[np.ndarray]:
        weights = []
        for num_inputs, num_outputs in zip(self.shape[:-1], self.shape[1:]):
            weights.append(np.random.normal(self.mean, self.std_dev, size=(num_inputs+1, num_outputs)))
        return weights

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Input layer is not activated.
        # We treat it as an activated layer so that we don't activate it.
        # (you wouldn't activate an already activated layer)
        a = X
        # Feed forward through each layer of hidden units and the last layer of output units
        for layer_ind in range(self.num_layers):
            # Add bias term
            b = np.hstack((a, [1]))
            # Feedforward through the weights w. summations
            f = b.dot(self.weights[layer_ind])
            # Activate the summations
            a = self.activation(f)
        return a

    def getWeights(self) -> List[np.ndarray]:
        return self.weights.copy()

    def setWeights(self, weights: List[np.ndarray]):
        # Check dimensions
        dimensions_incorrect = []
        for new_layer, old_layer in zip(weights, self.weights):
            dimensions_incorrect.append(new_layer.shape != old_layer.shape)
        if any(dimensions_incorrect):
            raise Exception("Weights are being set incorrectly in setWeights().\n"\
                            "Dimensions for new weights and network weights do not match!\n"\
                            "New weights dim != Network weights dim\n"\
                            +str([w.shape for w in weights])+" != "+str([w.shape for w in self.weights]))
        # Set weights
        self.weights = weights
        return None

    def activation(self, arr: np.ndarray) -> np.ndarray:
        return np.tanh(arr)

    def shape(self):
        return (self.num_inputs, self.num_hidden, self.num_outputs)

def createNNfromWeights(weights: List[np.ndarray]):
    weight_shape = calculateWeightShape(weights)
    num_inputs = weight_shape[0][0]-1
    num_hidden = [d[0]-1 for d in weight_shape[1:]]
    num_outputs = weight_shape[-1][1]
    net = NN(num_inputs, num_hidden, num_outputs)
    net.setWeights(weights)
    return net

def calculateWeightShape(weights: List[np.ndarray])->Tuple[Tuple[int]]:
    return tuple([w.shape for w in weights])

def calculateWeightSize(weights: List[np.ndarray])->Tuple[Tuple[int]]:
    return sum([w.size for w in weights])

if __name__ == "__main__":
    # 4 inputs, 10 hidden, 2 outputs
    # Weights should be shape (5, 10), (11, 2)
    nn = NN(num_inputs=4, num_hidden=10, num_outputs=2)
    Y = nn.forward(np.array([1,2,3,4]))
    print(Y)
