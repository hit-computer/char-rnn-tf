#coding:utf-8
import tensorflow as tf
import sys,time
import numpy as np
import cPickle, os
import random

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

model_path = './Model' #the path of model that need to save or load
save_time = 40 #load save_time saved models
is_sample = True #true means using sample, if not using max
is_beams = True #whether or not using beam search
beam_size = 2 #size of beam search
start_word = u'诚' #the first Chinese character of generated text
len_of_generation = 100 #The number of characters by generated

char_to_idx, idx_to_char = cPickle.load(open(model_path+'.voc', 'r'))

class Config(object):
    def __init__(self):
        self.init_scale = 0.04
        self.learning_rate = 0.001
        self.max_grad_norm = 15
        self.num_layers = 3
        self.num_steps = 25 # number of steps to unroll the RNN for
        self.hidden_size = 1000 # size of hidden layer of neurons
        self.iteration = 50
        self.save_freq = 5 #The step (counted by the number of iterations) at which the model is saved to hard disk.
        self.keep_prob = 0.5
        self.batch_size = 32
        self.vocab_size = 0
        
class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        self.lr = config.learning_rate

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #声明输入变量x, y

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size]) #size是wordembedding的维度
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)#返回一个tensor，shape是(batch_size, num_steps, size)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state) #inputs[:, time_step, :]的shape是(batch_size, size)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        """
        outpus是一个list，n*(batch_size, hidden_size)，tf.concat(1, outputs)返回一个矩阵(batch_size, n*hidden_size)
        reshape(..., [-1, size])
        """
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b #logits应该是(batch_size*time_step, vocab_size)，顺序是第一段的第一个词，第二个词，...，然后是第二段的第一个词，...
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._logits = logits

        if not is_training:
            self._prob = tf.nn.softmax(logits)
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op
        
        
def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1,1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                         {m.input_data: x,
                          m.initial_state: state})
    return prob, _state
    
def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        config = cPickle.load(open(model_path+'.fig', 'r'))
        config.batch_size = 1
        config.num_steps = 1
        
        start_idx = char_to_idx[start_word]

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model(is_training=False, config=config)

        #tf.global_variables_initializer().run()
        
        model_saver = tf.train.Saver()
        print 'model loading ...'
        model_saver.restore(session, model_path+'-%d'%save_time)
        print 'Done!'
        
        if not is_beams:
            _state = mtest.initial_state.eval()
            test_data = np.int32([start_idx])
            prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
            gen_res = [start_word]
            if is_sample:
                gen = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                gen = gen[0]
            else:
                gen = np.argmax(prob.reshape(-1))
            test_data = np.int32(gen)
            gen_res.append(idx_to_char[gen])
            for i in range(len_of_generation-1):
                prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                if is_sample:
                    gen = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                    gen = gen[0]
                else:
                    gen = np.argmax(prob.reshape(-1))
                test_data = np.int32(gen)
                gen_res.append(idx_to_char[gen])
            print 'Generated Result: ',''.join(gen_res)
        else:
            _state = mtest.initial_state.eval()
            beams = [(0.0, [idx_to_char[start_idx]], idx_to_char[start_idx])]
            test_data = np.int32([start_idx])
            prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
            y1 = np.log(1e-20 + prob.reshape(-1))
            if is_sample:
                top_indices = np.random.choice(config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
            else:
                top_indices = np.argsort(-y1)
            b = beams[0]
            beam_candidates = []
            for i in xrange(beam_size):
                wordix = top_indices[i]
                beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
            beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
            for xy in range(len_of_generation-1):
                beam_candidates = []
                for b in beams:
                    test_data = np.int32(b[2])
                    prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), b[3])
                    y1 = np.log(1e-20 + prob.reshape(-1))
                    if is_sample:
                        top_indices = np.random.choice(config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                    else:
                        top_indices = np.argsort(-y1)
                    for i in xrange(beam_size):
                        wordix = top_indices[i]
                        beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
                beam_candidates.sort(key = lambda x:x[0], reverse = True) # decreasing order
                beams = beam_candidates[:beam_size] # truncate to get new beams
            
            print 'Generated Result: ',''.join(beams[0][1])
            
if __name__ == "__main__":
    tf.app.run()