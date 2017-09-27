import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import h5py

from T_LSTM_AE import T_LSTM_AE

# Toy data

Data = []
Time = []
Assignments = []
Target = []
with h5py.File("Clustering_Data_1D.mat") as f:#
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
    for column in f['Assign']:
        row_data = []
        for row_number in range(len(column)):
            row_data.append(f[column[row_number]][:])
    Assignments.append(row_data)

cell_len = len(Data[0])


def generate_batches(data, time, assign, index):
    batch_data = np.transpose(data[0][index])
    batch_time = np.transpose(time[0][index])
    batch_assign = np.transpose(assign[0][index])
    return batch_data, batch_time, batch_assign


# set learning parameters
learning_rate = 1e-3
ae_iters = 500
display_step = 5

# set network parameters
input_dim = np.size(Data[0][0],0)
hidden_dim = 32
output_dim = input_dim

lstm_ae = T_LSTM_AE(input_dim, output_dim, hidden_dim)
loss_ae = lstm_ae.get_reconstruction_loss()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_ae) 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    Loss = np.zeros(ae_iters)
    for i in range(ae_iters):
        Ll = 0
        for j in range(cell_len):
            x, t, a = generate_batches(Data, Time, Assignments, j)
            _, L = sess.run([optimizer, loss_ae], feed_dict={lstm_ae.input: x, lstm_ae.time: t})
            Ll += L
        Loss[i] = Ll / cell_len

    plt.plot(Loss, label='T-LSTM')
    plt.xlabel('Number of Epochs', fontsize=14)
    plt.ylabel('Objective', fontsize=14)
    plt.show()















