# A demo to predict the target sequence of PPMI data.
# Data points have same sequence length within a batch.
# Inci M. Baytas
# Michigan State University
# February, 2017
import tensorflow as tf
import numpy as np
import random

import h5py

from T_LSTM import T_LSTM

Data = []
Time = []
Target = []
with h5py.File("Latest_PPMI_data.mat") as f:
    for column in f['Data']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Data.append(row_data)
    for column in f['Time']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Time.append(row_data)
    for column in f['Target']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Target.append(row_data)

cell_len = len(Data[0])

def generate_batches(data, time, target, cell_len):
    index = random.randint(0,cell_len-2)
    batch_data = np.transpose(data[0][index])
    batch_time = np.transpose(time[0][index])
    batch_target = np.transpose(target[0][index])
    return batch_data, batch_time, batch_target


# set learning parameters
learning_rate = 1e-3
training_iters = 5000
batch_size = 10
display_step = 10

# set network parameters
input_dim = np.size(Data[0][0],0)
hidden_dim = 512
output_dim = np.size(Target[0][0],0)

lstm = T_LSTM(input_dim, output_dim, hidden_dim)

loss = lstm.get_loss()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    count = 0
    while count * batch_size < training_iters:
        x, t, l = generate_batches(Data, Time, Target, cell_len)
        _, c = sess.run([optimizer, loss], feed_dict={lstm.input: x, lstm.target: l, lstm.time: t})
        if count % display_step == 0:
            print "Iter:", '%04d' % ((count + 1) * batch_size), "batch loss=", "{:.9f}".format(
                c)
        count += 1


    # Test

    print("Test error: ", (sess.run(lstm.get_loss(), feed_dict={lstm.input: np.transpose(Data[0][cell_len-1]), lstm.target: np.transpose(Target[0][cell_len-1]), lstm.time: np.transpose(Time[0][cell_len-1])})))



