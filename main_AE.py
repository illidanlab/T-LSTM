# A toy example for clustering with 2-layer TLSTM auto-encoder
# Inci M. Baytas, 2017
# How to run: Directly run the main file: python main_AE.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import h5py

from T_LSTM_AE import T_LSTM_AE

# A synthetic data

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
ae_iters = 2000

# set network parameters
input_dim = np.size(Data[0][0],0)
hidden_dim = 8
hidden_dim2 = 2
hidden_dim3 = 8
output_dim = hidden_dim
output_dim2 = hidden_dim2
output_dim3 = input_dim

lstm_ae = T_LSTM_AE(input_dim, output_dim, output_dim2, output_dim3, hidden_dim, hidden_dim2, hidden_dim3)

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
        print('Loss: %f' %(Loss[i]))

    assign_truth = []
    data_reps = []
    for c in range(cell_len):
        data = np.transpose(Data[0][c])
        time = np.transpose(Time[0][c])
        assign = np.transpose(Assignments[0][c])
        reps, cells = sess.run(lstm_ae.get_representation(), feed_dict={lstm_ae.input: data, lstm_ae.time: time})
        if c == 0:
            data_reps = reps
            assign_truth = assign
        else:
            data_reps = np.concatenate((data_reps, reps))
            assign_truth = np.concatenate((assign_truth, assign))

######## Clustering ##########################
kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++').fit(data_reps)
centroid_values = kmeans.cluster_centers_

plt.figure(1)
plt.scatter(data_reps[:, 0], data_reps[:, 1], c=assign_truth, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=35)
plt.title('TLSTM')
plt.show()
