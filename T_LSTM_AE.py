import tensorflow as tf
import math

class T_LSTM_AE(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        #tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name,shape=[input_dim, output_dim],initializer=tf.random_normal_initializer(0.0, std),regularizer = reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name,shape=[output_dim],initializer=tf.constant_initializer(0.0))

    def __init__(self, input_dim, output_dim, hidden_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Wi_enc = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight_enc')
        self.Ui_enc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_enc')
        self.bi_enc = self.init_bias(self.hidden_dim, name='Input_Hidden_bias_enc')

        self.Wf_enc = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight_enc')
        self.Uf_enc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_enc')
        self.bf_enc = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias_enc')

        self.Wog_enc = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight_enc')
        self.Uog_enc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_enc')
        self.bog_enc = self.init_bias(self.hidden_dim, name='Output_Hidden_bias_enc')

        self.Wc_enc = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight_enc')
        self.Uc_enc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_enc')
        self.bc_enc = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias_enc')

        self.W_decomp_enc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomp_Hidden_weight_enc')
        self.b_decomp_enc = self.init_bias(self.hidden_dim, name='Decomp_Hidden_bias_enc')

        self.Wi_dec = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight_dec')
        self.Ui_dec = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight_dec')
        self.bi_dec = self.init_bias(self.hidden_dim, name='Input_Hidden_bias_dec')

        self.Wf_dec = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight_dec')
        self.Uf_dec = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight_dec')
        self.bf_dec = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias_dec')

        self.Wog_dec = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight_dec')
        self.Uog_dec = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight_dec')
        self.bog_dec = self.init_bias(self.hidden_dim, name='Output_Hidden_bias_dec')

        self.Wc_dec = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight_dec')
        self.Uc_dec = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight_dec')
        self.bc_dec = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias_dec')

        self.W_decomp_dec = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomp_Hidden_weight_dec')
        self.b_decomp_dec = self.init_bias(self.hidden_dim, name='Decomp_Hidden_bias_dec')

        self.Wo = self.init_weights(self.hidden_dim, self.output_dim, name='Output_Layer_weight')
        self.bo = self.init_bias(self.output_dim, name='Output_Layer_bias')

        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])
        self.time = tf.placeholder('float', [None, None])

    def T_LSTM_Encoder_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0,1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0,0], [batch_size,1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t, self.hidden_dim)

        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp_enc) + self.b_decomp_enc)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis


        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi_enc) + tf.matmul(prev_hidden_state, self.Ui_enc) + self.bi_enc)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf_enc) + tf.matmul(prev_hidden_state, self.Uf_enc) + self.bf_enc)


        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog_enc) + tf.matmul(prev_hidden_state, self.Uog_enc) + self.bog_enc)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc_enc) + tf.matmul(prev_hidden_state, self.Uc_enc) + self.bc_enc)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def T_LSTM_Decoder_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t, self.hidden_dim)

        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp_dec) + self.b_decomp_dec)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis


        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi_dec) + tf.matmul(prev_hidden_state, self.Ui_dec) + self.bi_dec)

        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf_dec) + tf.matmul(prev_hidden_state, self.Uf_dec) + self.bf_dec)
        # f = tf.mul(T, f)

        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog_dec) + tf.matmul(prev_hidden_state, self.Uog_dec) + self.bog_dec)

        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc_dec) + tf.matmul(prev_hidden_state, self.Uc_dec) + self.bc_dec)

        # Current Memory cell
        Ct = f * prev_cell + i * C

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_encoder_states(self): # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_) #scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time) # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32) #np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0],tf.shape(scan_time)[1],1])
        concat_input = tf.concat([scan_time, scan_input],2) # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.T_LSTM_Encoder_Unit, concat_input, initializer=ini_state_cell, name='encoder_states')
        all_encoder_states = packed_hidden_states[:, 0, :, :]
        all_encoder_cells = packed_hidden_states[:, 1, :, :]
        return all_encoder_states, all_encoder_cells

    def get_representation(self):
        all_encoder_states, all_encoder_cells = self.get_encoder_states()
        # We need the last hidden state of the encoder
        representation = tf.reverse(all_encoder_states, [0])[0,:,:]#tf.reverse(all_encoder_states, [True, False, False])[0, :, :]
        decoder_ini_cell = tf.reverse(all_encoder_cells, [0])[0,:,:]#tf.reverse(all_encoder_cells, [True, False, False])[0, :, :]
        return representation, decoder_ini_cell

    def get_decoder_states(self):
        batch_size = tf.shape(self.input)[0]
        seq_length = tf.shape(self.input)[1]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input_ = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        z = tf.zeros([1, batch_size, self.input_dim], dtype=tf.float32)
        scan_input = tf.concat([scan_input_,z],0)
        scan_input = tf.slice(scan_input, [1,0,0],[seq_length ,batch_size, self.input_dim])
        scan_input = tf.reverse(scan_input, [0])#tf.reverse(scan_input, [True, False, False])
        scan_time_ = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        z2 = tf.zeros([1, batch_size], dtype=tf.float32)
        scan_time = tf.concat([scan_time_, z2],0)
        scan_time = tf.slice(scan_time, [1,0],[seq_length ,batch_size])
        scan_time = tf.reverse(scan_time, [0])#tf.reverse(scan_time, [True, False])
        initial_hidden, initial_cell = self.get_representation()
        ini_state_cell = tf.stack([initial_hidden, initial_cell])
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input],2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.T_LSTM_Decoder_Unit, concat_input, initializer=ini_state_cell, name='decoder_states')
        all_decoder_states = packed_hidden_states[:, 0, :, :]
        return all_decoder_states


    def get_output(self, state):
        output = tf.matmul(state, self.Wo) + self.bo
        # output = tf.nn.softmax(tf.nn.relu(tf.matmul(state, self.Wo) + self.bo))
        return output

    def get_decoder_outputs(self): # Returns the output of only the last time step
        all_decoder_states = self.get_decoder_states()
        all_outputs = tf.map_fn(self.get_output, all_decoder_states)
        reversed_outputs = tf.reverse(all_outputs, [0])#tf.reverse(all_outputs, [True, False, False])
        outputs_ = tf.transpose(reversed_outputs, perm=[2, 0, 1])
        outputs = tf.transpose(outputs_)
        return outputs

    def get_reconstruction_loss(self):
        outputs = self.get_decoder_outputs()
        # loss = tf.reduce_sum(tf.square(self.target - outputs))
        loss = tf.reduce_sum(tf.square(self.input - outputs))
        return loss


    def map_elapse_time(self, t, dim):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        # T = tf.div(c1, tf.add(t , c1), name='Log_elapse_time')

        Ones = tf.ones([1, dim], dtype=tf.float32)

        T = tf.matmul(T, Ones)

        return T



