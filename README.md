# char-rnn-tf
本程序用于自动生成一段中文文本（训练语料是英文时也可用于生成英文文本），具体生成文本的内容和形式取决于训练语料。模型基本思想和karpathy的[char-rnn](https://github.com/karpathy/char-rnn)程序一致，利用循环神经网络(RNN)在大规模语料上训练一个language model，然后利用训练好的language model去自动生成一段文本。相比于theano版本的[char-rnn](https://github.com/hit-computer/char-rnn)模型，本模型采用了多层RNN而不是单层（tensorflow中实现一个多层RNN简直太方便了），同时还支持max、sample和beam-search多种生成策略。本程序代码参考了tensorflow官方给出的一个language model程序[ptb_word_lm.py](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)。

模型结构如下图所示

![model](model.jpg?raw=true "model")


## 运行说明
本代码是在Python 2 / TensorFlow 0.12版本下编写的，要在1.0版本下运行需要修改Model.py文件的三个地方：把所有的`tf.nn.rnn_cell`都改成`tf.contrib.rnn`，39行的`tf.concat(1, outputs)`改成`tf.concat(outputs, 1)`，以及50行的`tf.nn.seq2seq.sequence_loss_by_example`改成`tf.contrib.legacy_seq2seq.sequence_loss_by_example`。

#### 模型参数设置（在Config.py文件中设置）：
- init_scale：参数使用均匀分布进行初始化，该值为均匀分布的上下界
- learning_rate：学习率
- max_grad_norm：对梯度进行规范化（gradient clipping） 
- num_layers：RNN的层级数目
- num_steps：RNN展开的步骤数（每次训练多少个字符）
- hidden_size：神经网络隐含层的维度
- iteration：模型总共迭代次数
- save_freq：每迭代多少次保存一次模型，同时进行一次生成
- keep_prob：dropout的概率
- batch_size：min-batch的大小
- model_path：模型保存路径
- 以下是generate过程需要设置的参数
- save_time: 载入第save_time次保存的模型
- is_sample：是否采用sample策略，设置为False是采用max策略
- is_beams：是否采用beam-search进行解码，设置为False时不采用（相当于beam_size=1）
- beam_size：beam-search的窗口大小
- len_of_generation：期望生成文本的长度（包括多少个字）
- start_sentence：期望生成文本的开始部分（可以是一个句子，一个词，也可以仅仅只是一个字）

#### 模型训练过程
在命令行中输入：

    python train.py [训练语料]

注意：训练语料为文本文件，请采用utf-8编码。

#### 模型测试过程（文本生成过程）
在进行文本生成时，有两种策略：max和sample策略。本程序同时支持这两种策略，以及beam-search解码。（在此感谢 [@fukuball](https://github.com/fukuball)对generate模型的扩展，使得generate.py程序能够以一个句子作为文本生成的开始）

在命令行中输入：

    python generate.py


## 实验
训练语料为初高中优秀作文，语料规模为31.70MB，共包含11,264,367个字符，总共迭代次数为40次。
#### 不同策略下的生成结果
第一个字我们选用“诚”，生成文本长度为100，在生成每个字时有两种策略（max和sample），同时在进行全局解码时可以采用beam-search或不采用，这样就有4种生成策略，以下是各种生成策略的结果：

策略1：max策略，不使用beam-search

>诚信，不是一个人的生命，而是一种人生的价值。  
>我们的生活是一个美丽的梦，我们的梦想是我们的梦想，我们的梦想是我们的梦想，我们的梦想是我们的梦想，我们的梦想是我们的梦想，我们的梦想是我们的梦想，我们的梦  

策略2：max策略，使用beam-search，beam_size=5

>诚信。  
>一个人的生命是有限的，但是，我们每个人都有自己的价值。  
>我们的生活中，我们应该有一颗感恩的心，我们的人生才会更加美好，更加美好，更加美好。  
>人生就像一场旅行，我们每一个人都有一个属于自己的  

策略3：sample策略，不使用beam-search

>诚信，锲而不舍，坚韧反应，庸碌于世界世界面对。  
>正是最好的成功。每次想起自己的成绩来，自然不能解释可事。或许在自己考上好时还是的。无论哪是天，只要你收留这个目标呢！你就是来手，把人类分别，厚重地向他那  

策略4：sample策略，使用beam-search，beam_size=2

>诚然无私，那是因为我看到了另一个自己的悲伤，是因为他得到了世界上最真挚的情感。  
>生活是一张美丽的咖啡，温暖的人生，是最美丽的一页。  
>人生如梦，让我们追求不到的梦想。不管是人生中的挫折还是痛苦与痛苦，  


**实验结果分析：**

策略1中采用max生成每个字，并且不使用beam-search进行全局解码，这种策略生成的结果很差的，随着句子长度的增加会出现重复的现象。而在策略2中，使用了beam-search进行全局解码后，结果有所提升，但局部文字仍有重复现象。策略3中，采用sample策略生成每个字，没有使用beam-search，由于sample引入了随机性，所以很好的解决文字重复出现的现象，但随机带来的弊端就是会出现局部语句不连贯的现象。策略4，相当于把max生成句子连贯和sample生成句子具有随机性两个优势进行了结合，即不会出现重复现象又能保证句子连贯。另外一方面，相比于max策略（策略1、2）中只要给定了开始字符生成的文本一定是固定的，策略4（以及策略3）具有一定的随机性，每次生成的文本都是不一样（增添生成文本的多样性），可以从生成的多个候选中选出一个最优的（有人的做法是再训练一个Ranking模型，对sample得到的多个候选进行排序，返回最好的结果或者top-n）。
