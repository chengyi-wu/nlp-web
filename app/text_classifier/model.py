# coding: utf-8
from __future__ import print_function
import os, sys
import tensorflow.contrib.keras as kr
import tensorflow as tf
import numpy as np
from collections import Counter
import time
from datetime import timedelta
import csv
import random

if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(native_content(content)))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [native_content(_.strip()) for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

class TCNNConfig(object):
    embedding_dim = 64  # 词向量维度
    seq_length = 4000  # 序列长度
    num_classes = 0  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 2  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果

class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class CnnModel(object):
    def __init__(self, vocab_dir, categories, config = None):
        self.vocab_dir = vocab_dir
        self.categories = categories
        self.cat_to_id = dict(zip(categories, range(len(categories))))
        self.words, self.word_to_id = read_vocab(self.vocab_dir)
        self.config = config
        if self.config is None:
            self.config = TCNNConfig()
        self.config.num_classes = len(categories)
        
        tf.reset_default_graph()
        self.model = TextCNN(self.config)
        c = tf.ConfigProto()
        c.gpu_options.allow_growth = True # Stop GPU from OOM
        self.session = tf.Session(config=c)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load(self, save_path):
        '''
        save_path: path to the best_validation
        '''
        print("Restore from %s" % save_path)
        self.save_path = save_path
        self.saver.restore(sess=self.session, save_path=save_path)
        
    def predict(self, content):
        '''
        content: full text to be predicted
        '''
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.model.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]

    @staticmethod
    def train(vocab_dir, categories, save_dir, train_dir, val_dir, config = None, full = False, num_epochs = 1):
        '''
        This is not supposed to be called by REST APIs

        vocab_dir: path to the vocab_dir
        categories: categories
        save_dir: parent dir to best_validation
        train_dir: path to the training file
        val_dir: path to the val file
        config: TCNNConfig
        full: whether to train from last time
        num_epochs: number of epochs
        '''
        save_path = os.path.join(save_dir, 'best_validation')
        if config is None:
            config = TCNNConfig()
        if full:
            print("Build vocab %s" % vocab_dir)
            build_vocab(train_dir, vocab_dir, config.vocab_size)

        cnnModel = CnnModel(vocab_dir, categories, config)
        saver = cnnModel.saver
        session = cnnModel.session
        cat_to_id = cnnModel.cat_to_id
        word_to_id = cnnModel.word_to_id
        words = cnnModel.words
        config = cnnModel.config

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if not full:
            cnnModel.load(save_path)

        model = cnnModel.model # <= TextCNN

        def feed_data(x_batch, y_batch, keep_prob):
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.keep_prob: keep_prob
            }
            return feed_dict

        def evaluate(sess, x_, y_):
            """评估在某一数据上的准确率和损失"""
            data_len = len(x_)
            batch_eval = batch_iter(x_, y_, 128)
            total_loss = 0.0
            total_acc = 0.0
            for x_batch, y_batch in batch_eval:
                batch_len = len(x_batch)
                feed_dict = feed_data(x_batch, y_batch, 1.0)
                loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                total_loss += loss * batch_len
                total_acc += acc * batch_len

            return total_loss / data_len, total_acc / data_len

        print("Loading training and validation data...")
        # 载入训练集与验证集
        start_time = time.time()
        x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
        x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

        flag = False
        for epoch in range(num_epochs):
            print('Epoch: %d / %d' % (epoch + 1, num_epochs))
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

                if total_batch % config.print_per_batch == 0:
                    # 每多少轮次输出在训练集和验证集上的性能
                    feed_dict[model.keep_prob] = 1.0
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=save_path)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                        + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break  # 跳出循环
            if flag:  # 同上
                break

def writetofile(rows, filename):
    '''
    rows: json from ES
    filename: loation where the file is stored as csv
    '''
    with open_file(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow('sentiment', 'businessline', 'tag', 'content') # header
        for row in rows:
            content = row['content']
            sentiment = row['sentiment']
            business = row['businessline']
            tag = row['tag']
            writer.writerow([sentiment, business, tag, content])

def split_file(filename, base_dir, categories):
    '''
    filename: writetofile(filename)
    base_dir: base_dir to store the trainging files
    categories = {
        'sentiment' : ['正面', '负面', '中立'],
    }
    '''
    for cat in categories:
        generate_training_files(filename, base_dir, cat, categories[cat])

def generate_training_files(filename, base_dir, category_name, categories, split_ratio=0.2):
    '''
    base_dir = /static/ => becomes /static/category_name/train.txt + /static/category_name/val.txt
    categories = ['正面', '负面', '中立']
    category_name = 'sentiment'
    split_ratio=0.2 : random.randint(1,10) <= 10 * (1 - split_ratio) goes to train
    '''
    base_dir = os.path.join(base_dir, category_name)
    train_file = os.path.join(base_dir, 'train.txt')
    val_file = os.path.join(base_dir, 'val.txt')

    train_file = open_file(train_file, 'w')
    val_file = open_file(val_file, 'w')

    x = 10 - int(10 * split_ratio)

    with open_file(filename, 'r') as f:
        reader = csv.reader(f)
        header = True
        pos = -1
        for row in reader:
            if header:
                for i, c in enumerate(row):
                    if c == category_name:
                        pos = i
            else:
                if random.randint(1, 10) <= x:
                    train_file.write('%s\t%s\n' % (row[pos], row[-1]))
                else:
                    val_file.write('%s\t%s\n' % (row[pos], row[-1]))
    train_file.close()
    val_file.close()

class PSRModel(object):
    def __init__(self):
        self.safety_5 = ['死亡', '杀人', '绑架', '抢救无效', '衰竭', '被害人']
        self.safety_4 = ['故意伤害', '强奸', '性侵', '恶性案件', '自杀']
        self.safety_3 = ['抢救', '殴打', '猥亵', '较大财产损失', '生命危险', '救治', '车祸', '重伤', '重大事故']
        self.safety_2 = ['受伤', '肢体冲突', '人身骚扰', '事故', '损伤', '追尾', '纠纷', '刮蹭', '交通事故']
        self.safety_1 = []

        self.platform_5 = ['宕机', '瘫痪', '产品漏洞', '系统漏洞']
        self.platform_4 = ['定价不明', '价格不明', '定价不合理', '价格不合理', '制度不合理', '规则不合理', '管理混乱', '收入体系不合理']
        self.platform_3 = ['交通事故', '保险不赔', '人身安全', '财产受威胁']
        self.platform_2 = ['物品遗失', '物品丢失', '包丢失', '包遗失', '手机遗失', '手机丢失', '无法找回']
        self.platform_1 = []

    def identify_safety_level(self, content):
        nums = []
        freqs = []

        nums.append(0)
        freqs.append(0)

        num, freq = self.find_total_matching_times(content, self.safety_2)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.safety_3)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.safety_4)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.safety_5)
        nums.append(num)
        freqs.append(freq)

        platform_level = nums.index(max(nums)) + 1

        print('NUM:', nums, 'Max index=', nums.index(max(nums)), 'Max num=', max(nums))
        print('FREQ: ', freqs, 'Max index=', freqs.index(max(freqs)), 'Max freq=', max(freqs))

        return platform_level

    def identify_platform_level(self, content):
        nums = []
        freqs = []

        nums.append(0)
        freqs.append(0)

        num, freq = self.find_total_matching_times(content, self.platform_2)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.platform_3)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.platform_4)
        nums.append(num)
        freqs.append(freq)

        num, freq = self.find_total_matching_times(content, self.platform_5)
        nums.append(num)
        freqs.append(freq)

        safety_level = nums.index(max(nums)) + 1

        print('NUM:', nums, 'Max index=', nums.index(max(nums)), 'Max num=', max(nums))
        print('FREQ: ', freqs, 'Max index=', freqs.index(max(freqs)), 'Max freq=', max(freqs))

        return safety_level


    def find_total_matching_times(self, content, keywords):
        total_times = 0
        for keyword in keywords:
            times = self.find_matching_times(content, keyword)
            total_times = total_times + times
        frequency = total_times / len(keywords)
        return total_times, frequency


    def find_matching_times(self, content, keyword):
        i = 0
        count = 0
        start = time.time()
        while i <= (len(content)-len(keyword)):
            j = 0
            while content[i] == keyword[j]:
                i = i + 1
                j = j + 1
                if j == len(keyword):
                    break
                elif j == len(keyword)-1:
                    count = count + 1
            else:
                i = i+1
                j = 0
        #print(count)
        #print(time.time()-start)
        return count

    def identify_propagation_level(self, media, duplicates, reads):
        if media == 5 or duplicates >= 50 or reads >= 100 * 1000:
            return 5

        if media == -1:
            return -1

        if media == 4 or duplicates >= 20 or reads >= 10 * 1000:
            return 4
        
        if media == 3 or duplicates >= 10 or reads >= 1000:
            return 3
        
        if media == 2 or duplicates >= 1 or reads >= 100:
            return 2
        
        return 1

    def identify_severity(self, P, S, R):
        '''
        P：{-1:未知, 1 - 4 } 
        S: {0:非负面，1 - 4 } 
        R: {0:非负面, 1 - 4 } 
        PSR : {-1:未知, 0:非负面, 1 - 125}
        Severity: { -1:未知, 0：非负面, 1 - 4 : 4 = A(非常严重)} 任意一个为5 = 4，其他有未知 = -1

        '''
        severity = psr = P * S * R

        if 1 <= psr <= 10:
            severity = 1
        elif 10 < psr <= 30:
            severity = 2
        elif 30 < psr <=60:
            severity = 3
        elif 60 < psr <= 150:
            severity = 4

        if P == 5 or S == 5 or R == 5:
            severity = 4
        elif P == -1:
            severity = -1
        
        return severity