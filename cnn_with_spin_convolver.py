import numpy as np
import random
from tensorflow import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


dataLoader = keras.datasets.mnist
(features, labels), (testFeatures, testLabels) = dataLoader.load_data()
onehot_labels = np.zeros((labels.shape[0], 10))
onehot_labels[np.arange(labels.shape[0]), labels] = 1
labels = onehot_labels
features = features/255
testFeatures = testFeatures/255
onehot_testLabels = np.zeros((testLabels.shape[0], 10))
onehot_testLabels[np.arange(testLabels.shape[0]), testLabels] = 1
testLabels = onehot_testLabels


# The parameters
BatchLength = 4
Size = [28*28, 1, 1]
length_of_kernel = [9, 1]
NumIteration = 2001
LearningRate = 1e-4
NumClasses = 10
EvalFreq = 1000
NumKernels = [8, 16]


def saw_convolution(signal_with_paddings, w_with_paddings, range_num):
    matrix_of_w = [w_with_paddings]
    shift_step = 1
    exp_attenuation = 0.9999  # a (attenuation parameter)
    vector_of_scalars = [1.0]
    for i in range(range_num-1):
        w_with_paddings = pow(exp_attenuation, i + 1) * tf.roll(w_with_paddings, shift_step, axis=0)
        matrix_of_w.append(w_with_paddings)
        vector_of_scalars.append(pow(exp_attenuation, i+1))
    matrix_of_w = tf.concat(matrix_of_w, 1)
    matrix_of_w = tf.expand_dims(matrix_of_w, 0)
    matrix_of_w = tf.tile(matrix_of_w, [BatchLength, 1, 1, 1])
    vector_of_scalars = tf.convert_to_tensor(vector_of_scalars)
    vector_of_scalars = tf.expand_dims(vector_of_scalars, 1)
    vector_of_scalars = tf.expand_dims(vector_of_scalars, 2)
    vector_of_scalars = tf.expand_dims(vector_of_scalars, 0)
    vector_of_scalars = tf.tile(vector_of_scalars, [BatchLength, 1, int(signal_with_paddings.shape[1]), 1])
    vector_of_scalars = tf.transpose(vector_of_scalars, perm=[0, 2, 1, 3])
    matrix_of_signal = tf.tile(signal_with_paddings, [1, 1, range_num, 1])
    matrix_of_signal = tf.multiply(matrix_of_signal, vector_of_scalars)
    saw_res = []
    for ker_num in range(int(matrix_of_w.shape[3])):
        saw_res_tmp = tf.multiply(matrix_of_signal[:, :, :, :], tf.expand_dims(matrix_of_w[:, :, :, ker_num], -1))
        saw_res_tmp = tf.math.tanh(saw_res_tmp)
        saw_res.append(tf.expand_dims(tf.reduce_sum(saw_res_tmp, axis=-1), -1))
    saw_res = tf.concat(saw_res, -1)
    saw_res = tf.reduce_sum(saw_res, axis=1)
    saw_res = tf.math.tanh(saw_res)
    saw_res = tf.expand_dims(saw_res, 2)
    return saw_res


def time_kernel_convolution(signal, out_channels):
    w = tf.get_variable('W', [length_of_kernel[0], int(Size[-1]), out_channels])
    shift_step = 1
    range_num = int((int(signal.shape[1]) + length_of_kernel[0] - 1)/shift_step)
    paddings_of_signal = tf.constant([[0, 0], [length_of_kernel[0]-1, length_of_kernel[0]-1], [0, 0], [0, 0]])
    paddings_of_w = tf.constant([[0, int(signal.shape[1])+length_of_kernel[0]-2], [0, 0], [0, 0]])
    signal_with_paddings = tf.pad(signal, paddings_of_signal, "CONSTANT")
    w_with_paddings = tf.pad(w, paddings_of_w, "CONSTANT")
    out = saw_convolution(signal_with_paddings, w_with_paddings, range_num)
    # Norm = tf.layers.batch_normalization(out, training=True)
    return out


tf.reset_default_graph()
InputData = tf.placeholder(tf.float32, [None] + Size)
OneHotLabels = tf.placeholder(tf.int32, [None, NumClasses])

input_data = InputData
CurrentFilters = Size[-1]

# a loop which creates all layers
LayerNum = 0
for N in range(len(NumKernels)):
    with tf.variable_scope('conv'+str(N)):
        input_data = time_kernel_convolution(input_data, NumKernels[N])
with tf.variable_scope('FC'):
    CurrentShape = input_data.get_shape()
    FeatureLength = int(CurrentShape[1]*CurrentShape[2]*CurrentShape[3])  # *CurrentShape[3]
    FC = tf.reshape(input_data, [-1, FeatureLength])
    W = tf.get_variable('W', [FeatureLength, NumClasses])
    FC = tf.matmul(FC, W)
    Bias = tf.get_variable('Bias', [NumClasses])
    FC = tf.add(FC, Bias)


with tf.name_scope('loss'):
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=OneHotLabels, logits=FC))

with tf.name_scope('optimizer'):
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):	  
    CorrectPredictions = tf.equal(tf.argmax(FC, 1), tf.argmax(OneHotLabels, 1))
    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))


Init = tf.global_variables_initializer()
with tf.Session() as Sess:
    Sess.run(Init)
    Step = 1
    while Step < NumIteration:
        UsedInBatch = random.sample(range(features.shape[0]), BatchLength)
        batch_xs = features[UsedInBatch, :]
        batch_ys = labels[UsedInBatch, :]
        batch_xs = np.reshape(batch_xs, [BatchLength]+Size)
        _, Acc, L = Sess.run([Optimizer, Accuracy, Loss], feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
        if (Step % 100) == 0:
            print("Iteration: " + str(Step))
            print("Accuracy: " + str(Acc))
            print("Loss: " + str(L))
        if (Step % EvalFreq) == 0:
            SumAcc = 0.0
            for i in range(0, testFeatures.shape[0]):
                batch_xs = testFeatures[i, :]
                batch_ys = testLabels[i, :]
                batch_xs = np.reshape(batch_xs, [1]+Size)
                batch_ys = np.reshape(batch_ys, [1, NumClasses])
                a = Sess.run(Accuracy, feed_dict={InputData: batch_xs, OneHotLabels: batch_ys})
                SumAcc += a
            print("Independent Test set: "+str(float(SumAcc)/testFeatures.shape[0]))
        Step += 1
