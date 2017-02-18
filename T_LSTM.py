# Time Aware LSTM
# Inci M. Baytas
# Computer Science-Michigan State University
# February, 2017
import tensorflow as tf
import math


class T_LSTM(object):
    def init_weights(self, input_dim, output_dim, name=None, std=1.0):
        return tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=std / math.sqrt(input_dim)), name=name)

    def init_bias(self, output_dim, name=None):
        return tf.Variable(tf.zeros([output_dim]), name=name)

    def __init__(self, input_dim, output_dim, hidden_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Wi = self.init_weights(input_dim, hidden_dim, name='Input_Hidden_weight')
        self.Ui = self.init_weights(hidden_dim, hidden_dim, name='Input_State_weight')
        self.bi = self.init_bias(hidden_dim, name='Input_Hidden_bias')

        self.Wf = self.init_weights(input_dim, hidden_dim, name='Forget_Hidden_weight')
        self.Uf = self.init_weights(hidden_dim, hidden_dim, name='Forget_State_weight')
        self.bf = self.init_bias(hidden_dim, name='Forget_Hidden_bias')

        self.Wog = self.init_weights(input_dim, hidden_dim, name='Output_Hidden_weight')
        self.Uog = self.init_weights(hidden_dim, hidden_dim, name='Output_State_weight')
        self.bog = self.init_bias(hidden_dim, name='Output_Hidden_bias')

        self.Wc = self.init_weights(input_dim, hidden_dim, name='Cell_Hidden_weight')
        self.Uc = self.init_weights(hidden_dim, hidden_dim, name='Cell_State_weight')
        self.bc = self.init_bias(hidden_dim, name='Cell_Hidden_bias')

        self.W_decomp = self.init_weights(hidden_dim, hidden_dim, name='Input_Hidden_weight')
        self.b_decomp = self.init_bias(hidden_dim, name='Input_Hidden_bias_enc')

        self.Wo = self.init_weights(hidden_dim, output_dim, name='Output_Layer_weight')
        self.bo = self.init_bias(output_dim, name='Output_Layer_bias')

        # [batch size x seq length x input dim]
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])
        self.target = tf.placeholder('float', shape=[None, None, self.output_dim])

        self.time = tf.placeholder('float', [None, None])


    def T_LSTM_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unpack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0,1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0,0], [batch_size,1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t, self.hidden_dim)

        # Decompose the previous cell if there is a elapse time
        prev_cell_decompose = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - prev_cell_decompose + tf.mul(T, prev_cell_decompose)

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf + self.bf))
        # f = tf.mul(T, f)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.pack([current_hidden_state, Ct])


    def get_states(self): # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_) #scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time) # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32) #np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        ini_state_cell = tf.pack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0],tf.shape(scan_time)[1],1])
        concat_input = tf.concat(2, [scan_time, scan_input]) # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.T_LSTM_Unit, concat_input, initializer=ini_state_cell, name='encoder_states')
        all_states = packed_hidden_states[:, 0, :, :]
        # all_encoder_cells = packed_hidden_states[:, 1, :, :]
        return all_states

    def get_output(self, state):
        output = tf.matmul(state, self.Wo) + self.bo
        # output = tf.nn.softmax(tf.nn.relu(tf.matmul(state, self.Wo) + self.bo))
        return output


    def get_outputs(self): # Returns the output of only the last time step
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        all_outputs_ = tf.transpose(all_outputs, perm=[2, 0, 1])
        all_outputs = tf.transpose(all_outputs_)
        return all_outputs

    def get_loss(self):
        outputs = self.get_outputs()
        loss = tf.reduce_mean(tf.square(self.target - outputs))
        # loss = tf.reduce_sum(tf.square(self.input - outputs))
        return loss

    def map_elapse_time(self, t, dim):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        # T = tf.div(c1, tf.add(t , c1), name='Log_elapse_time')

        Ones = tf.ones([1, dim], dtype=tf.float32)

        T = tf.matmul(T, Ones)

        return T



