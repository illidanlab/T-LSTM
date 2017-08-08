# Time-Aware LSTM
# main function for supervised task
# An example dataset is shared but the original synthetic dataset
# can be accessed from http://www.emrbots.org/.
# Inci M. Baytas, 2017
#
# How to run: Give the correct path to the data
# Data is a list where each element is a 3 dimensional matrix which contains same length sequences.
# Instead of zero padding, same length sequences are put in the same batch.
# Example: L is the list containing all the batches with a length of N.
#          L[0].shape gives [number of samples x sequence length x dimensionality]
# Call python main.py 50 1028 512 2 0.6
# For instance; number_epochs: 50, hidden_dim:1028, fully_connected dim: 512, output_dim: 2, training dropout prob: 0.6

import tensorflow as tf
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import sys
import math



from TLSTM import T_LSTM

# To generate test/training splits.
def generate_split(input, target, elapse_time, cell_len):
    Train_data = []
    Test_data = []
    Train_labels = []
    Test_labels = []
    Train_time = []
    Test_time = []
    for i in range(cell_len):
        data = input[0][i]
        label = target[0][i]
        time = elapse_time[0][i]
        data, label, time = shuffle(data, label, time)
        N = np.size(data,0)
        if N > 1:
           split = int(math.floor(N * 0.7))
           Train_data.append(data[1:split])
           Test_data.append(data[split:N])
           Train_labels.append(label[1:split])
           Test_labels.append(label[split:N])
           Train_time.append(time[1:split])
           Test_time.append(time[split:N])
        else:
           Train_data.append(data)
           Train_labels.append(label)
           Train_time.append(time)
    return Train_data, Train_time, Train_labels, Test_data, Test_time, Test_labels


def main(argv):

    S1 = 'Synt_EHR2.mat'
    m = sio.loadmat(S1)
    General_Patient = m['General_Patient']
    General_Elapsed = m['General_Elapsed']
    General_Labels = m['General_Labels']

    cell_len = len(General_Patient[0])
    data_train_batches,elapsed_train_batches,labels_train_batches,data_test_batches,elapsed_test_batches,labels_test_batches = generate_split(General_Patient, General_Labels, General_Elapsed, cell_len)


    number_train_batches = len(data_train_batches)
    number_test_batches = len(data_test_batches)

    print("Train and test data is loaded!")


    # set learning parameters
    learning_rate = 1e-3
    training_epochs = int(sys.argv[1])

    # set network parameters
    input_dim = data_train_batches[0].shape[2]
    hidden_dim = int(sys.argv[2])
    fc_dim = int(sys.argv[3])
    output_dim = int(sys.argv[4]) # Binary labels
    train_dropout_prob = float(sys.argv[5])


    lstm = T_LSTM(input_dim, output_dim, hidden_dim, fc_dim)

    cross_entropy, y_pred, y, logits, labels= lstm.get_cost_acc()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy) # RMSPropOptimizer


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        Cost = np.zeros(training_epochs)
        for epoch in range(training_epochs):#
            # Loop over all batches
            total_cost = 0
            for i in range(number_train_batches):#
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], elapsed_train_batches[i]
                batch_ts = np.reshape(batch_ts, [batch_ts.shape[0],batch_ts.shape[2]])
                sess.run(optimizer, feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys, lstm.time: batch_ts, lstm.keep_prob:train_dropout_prob})
                cost, y_train, y_pred_train, logits_train, labels_train = sess.run([cross_entropy, y, y_pred, logits, labels], feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys, lstm.time: batch_ts, lstm.keep_prob:train_dropout_prob})
                total_cost += cost

            Cost[epoch] = total_cost / number_train_batches

        print("Optimization is over!")

        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []
        for i in range(number_train_batches):  #
            batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], elapsed_train_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0],batch_ts.shape[2]])
            cost, y_train, y_pred_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,lstm.time: batch_ts,lstm.keep_prob: train_dropout_prob})
            if i>0:
               Y_true = np.concatenate([Y_true, y_train], 0)
               Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
               Labels = np.concatenate([Labels, labels_train], 0)
               Logits = np.concatenate([Logits, logits_train], 0)
            else:
               Y_true = y_train
               Y_pred = y_pred_train
               Labels = labels_train
               Logits = logits_train

        total_acc = accuracy_score(Y_true, Y_pred)
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        print("Train Accuracy = {:.3f}".format(total_acc))
        print("Train AUC = {:.3f}".format(total_auc))


        test_dropout_prob = 1.0
        Y_true = []
        Y_pred = []
        Logits = []
        Labels = []
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], labels_test_batches[i], elapsed_test_batches[i]
            batch_ts = np.reshape(batch_ts, [batch_ts.shape[0],batch_ts.shape[2]])
            c_test, y_pred_test, y_test, logits_test, labels_test  = sess.run(lstm.get_cost_acc(), feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys, lstm.time: batch_ts, lstm.keep_prob: test_dropout_prob})
            if i>0:
               Y_true = np.concatenate([Y_true, y_test], 0)
               Y_pred = np.concatenate([Y_pred, y_pred_test], 0)
               Labels = np.concatenate([Labels, labels_test], 0)
               Logits = np.concatenate([Logits, logits_test], 0)
            else:
               Y_true = y_test
               Y_pred = y_pred_test
               Labels = labels_test
               Logits = logits_test
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_acc = accuracy_score(Y_true, Y_pred)
        print("Test Accuracy = {:.3f}".format(total_acc))
        print("Test AUC = {:.3f}".format(total_auc))

if __name__ == "__main__":
   main(sys.argv[1:])
