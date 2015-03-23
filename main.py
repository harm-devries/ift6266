######################
# Model construction #
######################

from theano import tensor

from blocks.bricks import Rectifier, MLP, Softmax
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.conv import ConvolutionalLayer, ConvolutionalSequence
from blocks.bricks.conv import MaxPooling, ConvolutionalActivation, Flattener
from blocks.filter import VariableFilter
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.roles import WEIGHT, FILTER, INPUT
from blocks.graph import ComputationGraph, apply_dropout
from speech_project.datasets.schemes import SequentialShuffledScheme

import numpy
import logging
logging.basicConfig()


x = tensor.tensor4('features')
y = tensor.lmatrix('targets')

# Convolutional layers

conv_layers = [ConvolutionalLayer(Rectifier().apply, (3, 3), 16, (2, 2), name='l1'),
               ConvolutionalLayer(Rectifier().apply, (3, 3), 32, (2, 2), name='l2'),
               ConvolutionalLayer(Rectifier().apply, (3, 3), 64, (2, 2), name='l3'),
               ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l4'),
               ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l5'),
               ConvolutionalLayer(Rectifier().apply, (3, 3), 128, (2, 2), name='l6')]
convnet = ConvolutionalSequence(conv_layers, num_channels=3,
                                image_size=(260, 260),
                                weights_init=Uniform(0, 0.2),
                                biases_init=Constant(0.0), 
                                tied_biases=False)
convnet.initialize()
output_dim = numpy.prod(convnet.get_dim('output'))
print output_dim
# Fully connected layers

features = Flattener().apply(convnet.apply(x))

mlp = MLP(activations=[Rectifier(), Rectifier(), None],
          dims=[output_dim, 256, 128, 2], weights_init=Uniform(0, 0.2),
          biases_init=Constant(0.0))
mlp.initialize()
y_hat = mlp.apply(features)

# Numerically stable softmax
#cost = Softmax().categorical_cross_entropy(y, y_hat)
#cost.name = 'nll'
y = y.flatten()
misclass = MisclassificationRate().apply(y, y_hat)
misclass.name = 'error_rate'

cost = Softmax().categorical_cross_entropy(y, y_hat)
# z = y_hat - y_hat.max(axis=1).dimshuffle(0, 'x')
# log_prob = z - tensor.log(tensor.exp(z).sum(axis=1).dimshuffle(0, 'x'))
# flat_log_prob = log_prob.flatten()
# range_ = tensor.arange(y.shape[0])
# flat_indices = y.flatten() + range_ * 2
# log_prob_of = flat_log_prob[flat_indices].reshape(y.shape, ndim=2)
# cost = -log_prob_of.mean()
cost.name = 'nll'

cg = ComputationGraph([cost])

inputs = VariableFilter(roles=[INPUT], bricks=mlp.linear_transformations[:2])(cg.variables)
cg_dropout = apply_dropout(cg, inputs, 0.5)

weights = VariableFilter(roles=[FILTER, WEIGHT])(cg.variables)                     
l2_regularization = 1e-3 * sum([(W**2).sum() for W in weights])

cost_l2 = cg_dropout.outputs[0] + l2_regularization
cost_l2.name = 'nll_l2_dropout'

# Print sizes to check
print("Representation sizes:")
for layer in convnet.layers:
    print(layer.get_dim('input_'))

############
# Training #
############

from blocks.dump import load_parameter_values
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter
from blocks.algorithms import GradientDescent, Momentum
from blocks.extensions import Printing, SimpleExtension
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint, LoadFromDump
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.model import Model

from dataset import DogsVsCats
from streams import RandomPatch
from extensions import MyLearningRateSchedule, EarlyStoppingDump
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

batch_size = 128
rng = numpy.random.RandomState(1)
training_stream = DataStream(DogsVsCats('train'),
  iteration_scheme=ShuffledScheme(20000, batch_size))
training_stream = RandomPatch(training_stream, 270, (260, 260))

valid_stream = DataStream(DogsVsCats('valid'),
                iteration_scheme=ShuffledScheme(2500, batch_size))
valid_stream = RandomPatch(valid_stream, 270, (260, 260))

model = Model(cost_l2)
algorithm = GradientDescent(cost=cost_l2, params=model.parameters, step_rule=Momentum(learning_rate=1e-2,
                                                          momentum=0.9))

main_loop = MainLoop(
    model=model, data_stream=training_stream, algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=150),
        TrainingDataMonitoring(
            [cost, misclass],
            prefix='train',
            after_epoch=True),
        DataStreamMonitoring(
            [cost, misclass],
            valid_stream,
            prefix='valid'),
        Checkpoint('cats_vs_dogs.pkl', after_epoch=True),
        EarlyStoppingDump('/home/user/Documents/ift6266', 'valid_error_rate'),
        Printing()
    ]
)
main_loop.run()
