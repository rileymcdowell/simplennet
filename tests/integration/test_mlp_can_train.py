from __future__ import print_function
import numpy as np

from simplenet import get_mlp_network_regressor, get_mlp_network_classifier 
from simplenet.trainers import MlpTrainer 

def test_can_execute_forward_pass():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    output = net.activate(np.array([0.35, 0.9]))
        
    assert np.allclose(0.69, output, atol=0.01)

def test_can_execute_single_backpropagation_pass():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This article is linked from the wikipedia backpropagation page as a resource for
    an in-depth explanation of the backpropagation algorithm (as of 9-13-15).

    This article does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    trainer.train_on_data(input[np.newaxis,:], desired_output[np.newaxis,:]) 

    assert np.allclose(net.layers[0].weights, np.array([[0.10, 0.80], [0.40, 0.59]]), atol=0.01)
    assert np.allclose(net.layers[1].weights, np.array([0.27, 0.87]), atol=0.01)
        

def test_backpropagation_decreases_error():
    """
    Following example on 'https://www4.rgu.ac.uk/files/chapter3 - bp.pdf'
    This freely available book chapter is linked from the wikipedia backpropagation 
    page as a resource for an in-depth explanation of the backpropagation 
    algorithm (as of 9-13-15).

    This chapter does not include neuron bias, so we will turn it off for this test.
    """
    net = get_mlp_network_classifier(2, 2, 1, apply_bias=False)
    net.layers[0].weights = np.array([[0.1, 0.8],[0.4, 0.6]])
    net.layers[1].weights = np.array([[0.3, 0.9]])

    input = np.array([0.35, 0.9])
    desired_output = np.array([0.5])
    trainer = MlpTrainer(net)

    original_error = desired_output - net.activate(input)
    trainer.train_on_data(input[np.newaxis,:], desired_output[np.newaxis,:]) 
    new_error = desired_output - net.activate(input)

    assert original_error < new_error
    assert np.allclose(original_error, -0.19, atol=0.01)
    assert np.allclose(new_error, -0.18, atol=0.01)

def test_mlp_classifier_can_approximate_xor_function():
    """
    This is the hello world of using sigmoid neuron networks
    for classification. We should be able to nail this one.
    """
    x = np.array([[1,1], [0,1], [1,0], [0,0]])
    y = np.array([[0],[1],[1],[0]])

    hidden_neurons = 2 

    mlp_network = get_mlp_network_classifier(2, hidden_neurons, 1, apply_bias=False)
    mlp_trainer = MlpTrainer(mlp_network, learning_rate=1.0)

    predicted_out = mlp_network.activate_many(x)
    
    # Cycle through the 4 points, training on each one individually many times.
    for _ in range(20000):
        original_error = y - mlp_network.activate_many(x)
        for i in range(4):
            mlp_trainer.train_on_datum(x[i], y[i])
        new_error = np.sum((y - mlp_network.activate_many(x))**2)
        if new_error < 0.001:
            break

    print(new_error)
    predicted_out = mlp_network.activate_many(x)
    # Should be no problem getting them all right with high certainty.
    assert np.allclose(y, predicted_out, rtol=0.0, atol=1e-1) 
    
