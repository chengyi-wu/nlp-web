from model import CnnModel

CnnModel.train(vocab_dir = '/Users/chengyiwu/GitHub/nlp/vocab.txt'
, categories = ['正面', '负面', '中立']
, save_dir = '/Users/chengyiwu/GitHub/nlp/sentiment/textcnn'
, train_dir = '/Users/chengyiwu/GitHub/nlp2/text-classification-cnn-rnn-sentiment/data/cnews/cnews.train.txt'
, val_dir = '/Users/chengyiwu/GitHub/nlp2/text-classification-cnn-rnn-sentiment/data/cnews/cnews.val.txt'
, config = None, full = True, num_epochs = 1)

CnnModel.train(vocab_dir = '/Users/chengyiwu/GitHub/nlp/vocab.txt'
, categories = ['正面', '负面', '中立']
, save_dir = '/Users/chengyiwu/GitHub/nlp/sentiment/textcnn'
, train_dir = '/Users/chengyiwu/GitHub/nlp2/text-classification-cnn-rnn-sentiment/data/cnews/cnews.train.txt'
, val_dir = '/Users/chengyiwu/GitHub/nlp2/text-classification-cnn-rnn-sentiment/data/cnews/cnews.val.txt'
, config = None, full = False, num_epochs = 1)