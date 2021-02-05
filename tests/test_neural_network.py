from neural_network import NeuralNetwork, MUTATION_CHANCE
import numpy as np

def test_make_nn():
    hidden_layers = [1,2,3,4,5,6,5,100]
    my_nn = NeuralNetwork(input_layer_size=8, output_layer_size=6, hidden_layers=hidden_layers)


    expected_shapes = {"hidden_layer_0": (8,1),
                       "hidden_layer_1": (1, 2),
                       "hidden_layer_2": (2, 3),
                       "hidden_layer_3": (3, 4),
                       "hidden_layer_4": (4, 5),
                       "hidden_layer_5": (5, 6),
                       "hidden_layer_6": (6, 5),
                       "hidden_layer_7": (5, 100),
                       "output_layer": (100, 6),
                       "input_layer":(0, 8)
                      }
    for layer in my_nn.brain:
        assert np.all(my_nn.brain[layer]["biases"]) == False
        assert my_nn.brain[layer]["biases"].size == expected_shapes[layer][1]
    for layer in my_nn.brain:
        if layer == "input_layer":
            continue
        assert my_nn.brain[layer]["weights"].shape == expected_shapes[layer]

def test_mutate():
    # make a nn then mutate and check that they're not the same
    np.random.seed(1)
    hidden_layers = [10, 20]
    unmutated = NeuralNetwork(input_layer_size=1, output_layer_size=5, hidden_layers=hidden_layers)
    np.random.seed(1)
    mutated = NeuralNetwork(input_layer_size=1, output_layer_size=5, hidden_layers=hidden_layers)
    mutated.mutate()
    num_differences = 0
    total_num_weights = 10 + 10 * 20 + 20 * 5
    for layer1, layer2 in zip(unmutated.brain, mutated.brain):
        try:
            for weight_list1, weight_list2 in zip(unmutated.brain[layer1]["weights"], mutated.brain[layer2]["weights"]):
                for weight1, weight2 in zip(weight_list1, weight_list2):
                    if weight1 != weight2:
                        num_differences += 1
        except KeyError:
            pass
    actual_mutation_percentage = num_differences / total_num_weights
    assert actual_mutation_percentage / MUTATION_CHANCE > 0.7
