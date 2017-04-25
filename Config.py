#coding:utf-8

class Config(object):
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 15
    num_layers = 3
    num_steps = 30 # number of steps to unroll the RNN for
    hidden_size = 800 # size of hidden layer of neurons
    iteration = 30
    save_freq = 5 #The step (counted by the number of iterations) at which the model is saved to hard disk.
    keep_prob = 0.5
    batch_size = 128
    model_path = './Model' #the path of model that need to save or load
    
    #parameters for generation
    save_time = 20 #load save_time saved models
    is_sample = True #true means using sample, if not using max
    is_beams = True #whether or not using beam search
    beam_size = 2 #size of beam search
    len_of_generation = 100 #The number of characters by generated
    start_sentence = u'那是因为我看到了另一个自己的悲伤' #the seed sentence to generate text