# char-rnn-tf
本程序用于自动生成一段中文文本（具体生成文本的内容和形式取决于训练语料）。模型基本思想和karpathy的[char-rnn](https://github.com/karpathy/char-rnn)程序一致，利用循环神经网络(RNN)在大规模语料上训练一个language model，然后利用训练好的language model去自动生成一段文本。相比于theano版本的[char-rnn](https://github.com/hit-computer/char-rnn)模型，本模型采用了多层RNN而不是单层（tensorflow中实现一个多层RNN简直太方便了），同时还加入beam-sample的生成策略。本程序代码参考了tensorflow官方给出的一个language model程序[ptb_word_lm.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py)。

模型结构如下图所示

![model](model.jpg?raw=true "model")

========================================================
##运行说明
####模型训练过程
在命令行中输入：

    python train.py [训练语料]

注意：训练语料为文本文件，请采用utf-8编码。

train.py文件中模型参数说明：
- model_path：模型保存路径
- （以下参数在Config类中设定）
- init_scale：参数使用均匀分布进行初始化，该值为均匀分布的上下界
- learning_rate：学习率
- max_grad_norm：对梯度进行规范化，参见tensorflow的clip_by_global_norm函数 
- num_layers：RNN的层级数目
- num_steps：RNN展开的步骤数（每次训练多少个字符）
- hidden_size：神经网络隐含层的维度
- save_freq：每迭代多少次保存一次模型，同时进行一次生成
- keep_prob：dropout的概率
- batch_size：min-batch的大小

####模型测试过程（文本生成过程）
在进行文本生成时，有两种策略：max和sample策略。本程序同时支持这两种策略，以及beam-search解码。

在命令行中输入：

    python generate.py
    
generate.py文件中模型参数说明：
- model_path：模型保存路径
- is_sample：是否采用sample策略，设置为False是采用max策略
- is_beams：是否采用beam-search进行解码，设置为False时不采用（相当于beam_size=1）
- beam_size：beam-search的窗口大小
- start_word：期望生成文本的开始符（第一个字）
- len_of_generation：期望生成文本的长度（包括多少个字）