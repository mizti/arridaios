import pandas as pd
import numpy as np
import os
import six
import chainer
import chainer.functions as F
import chainer.links as L
import zipfile
from chainer.links import caffe
from chainer import link, Chain, optimizers, Variable
from PIL import Image


class ImageData():
    def __init__(self, img_dir, meta_data):
        self.img_dir = img_dir

        assert meta_data.endswith('.tsv')
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.index = np.array(self.meta_data.index)
        self.split = False

    def shuffle(self):
        assert self.split == True
        self.train_index = np.random.permutation(self.train_index)

    def split_train_val(self, train_size):
        self.train_index = np.random.choice(self.index, train_size, replace=False)
        self.val_index = np.array([i for i in self.index if i not in self.train_index])
        self.split = True

    def generate_minibatch(self, batchsize, img_size = 224, mode = None):
        i = 0
        if mode == 'train':
            assert self.split == True
            meta_data = self.meta_data.ix[self.train_index]
            index = self.train_index

        elif mode == 'val':
            assert self.split == True
            meta_data = self.meta_data.ix[self.val_index]
            index = self.val_index

        else:
            meta_data = self.meta_data
            index = self.index

        while i < len(index):
            data = meta_data.iloc[i:i + batchsize]
            images = []
            for f in list(data['file_name']):
                image = Image.open(os.path.join(self.img_dir, f))
                image = image.resize((img_size, img_size))
                images.append(np.array(image))
            images = np.array(images)
            images = images.transpose((0,3,1,2))
            images = images.astype(np.float32)

            if 'category_id' in data.columns:
                labels = np.array(list(data['category_id']))
                labels = labels.astype(np.int32)
                yield images, labels
            else:
                yield images
            i += batchsize

class Alex(Chain):
    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 25),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride = 2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        y = self.fc8(h)

        return y

class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)

        return self.loss

def train_val(train_data, classifier, optimizer, num_train = 9000, epochs = 10, batchsize = 30, gpu = True):
    # split data to train and val
    train_data.split_train_val(num_train)

    for epoch in range(epochs):
        # train
        classifier.predictor.train = True
        num_samples = 0
        train_cum_loss = 0
        train_cum_acc = 0
        for data in train_data.generate_minibatch(batchsize, mode = 'train'):
            num_samples += len(data[0])
            #print num_samples, '/', len(train_data.train_index), '(epoch:%s)'%(epoch+1)
            optimizer.zero_grads()
            x, y = Variable(data[0]), Variable(data[1])
            if gpu:
                x.to_gpu()
                y.to_gpu()
            loss = classifier(x, y)

            train_cum_acc += classifier.accuracy.data*batchsize
            #print 'train_accuracy:', train_cum_acc/num_samples
            train_cum_loss += classifier.loss.data*batchsize
            #print 'train_loss:', train_cum_loss/num_samples

            loss.backward()    # back propagation
            optimizer.update() # update parameters

        train_accuracy = train_cum_acc/num_samples
        train_loss = train_cum_loss/num_samples

        # validation
        classifier.predictor.train = False
        num_samples = 0
        val_cum_loss = 0
        val_cum_acc = 0
        for data in train_data.generate_minibatch(batchsize, mode = 'val'):
            num_samples += len(data[0])
            #print num_samples, '/', len(train_data.val_index), '(epoch:%s)'%(epoch+1)
            x, y = Variable(data[0]), Variable(data[1])
            if gpu:
                x.to_gpu()
                y.to_gpu()
            loss = classifier(x, y)

            val_cum_acc += classifier.accuracy.data*batchsize
            #print 'val_accuracy:', val_cum_acc/num_samples
            val_cum_loss += classifier.loss.data*batchsize
            #print 'val_loss:', val_cum_loss/num_samples

        val_accuracy = val_cum_acc/num_samples
        val_loss = val_cum_loss/num_samples

        print('-----------------', 'epoch:', epoch+1, '-----------------')
        print('train_accuracy:', train_accuracy, 'train_loss:', train_loss)
        print('val_accuracy:', val_accuracy, 'val_loss:', val_loss)
        print('\n')

        # shuffle train data
        train_data.shuffle()

    return classifier, optimizer

def download_model(path, model_name):
    if model_name == 'alexnet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
        name = 'bvlc_alexnet.caffemodel'
    elif model_name == 'caffenet':
        url = 'http://dl.caffe.berkeleyvision.org/' \
              'bvlc_reference_caffenet.caffemodel'
        name = 'bvlc_reference_caffenet.caffemodel'
    elif model_name == 'googlenet':
        url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
        name = 'bvlc_googlenet.caffemodel'
    elif model_name == 'resnet':
        url = 'http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip'
        name = 'models.zip'
    else:
        raise RuntimeError('Invalid model type. Choose from '
                           'alexnet, caffenet, googlenet and resnet.')

    if os.path.isfile(path + '/' + name):
        print('passed!')
        pass
    else:
        print('Downloading model file...')
        six.moves.urllib.request.urlretrieve(url, path + '/' + name)
        print('Download completed')
        if model_name == 'resnet':
            print('extracting file..')
            with zipfile.ZipFile(path + '/' + name, 'r') as zf:
                zf.extractall('.')
        print('Done.')
    return path + '/' + name

def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    print('--------')
    print(src)
    print(dst)
    for child in src.children():
        print('==============')
        print(child.name)
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                print('a=' + str(a))
                print('b=' + str(b))
                print(a[0])
                print(b[0])
                print(a[1].data.shape)
                print(b[1].data.shape)
                if a[0] != b[0]:
                    print("unmatch1")
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    print("unmatch2")
                    match = False
                    break
            if not match:
                print ('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print ('Copy %s' % child.name)

def predict(test_data, classifier, batchsize = 5, gpu = True):
    if gpu:
        classifier.predictor.to_gpu()
    else:
        classifier.predictor.to_cpu()
    classifier.predictor.train = False
    num_samples = 0
    predictions = np.zeros((len(test_data.index),25))
    for data in test_data.generate_minibatch(batchsize):
        num_samples += len(data)
        #print num_samples, '/', len(test_data.index)
        x = Variable(data)
        if gpu:
            x.to_gpu()
        yhat = classifier.predictor(x)
        yhat = F.softmax(yhat)
        yhat.to_cpu()
        predictions[num_samples-len(data):num_samples,:] = yhat.data

    return predictions

alex = Alex()

model_path = download_model('data', 'alexnet')  # モデルパラメータのダウンロード
print(model_path)
func = caffe.CaffeFunction(model_path)

copy_model(func, alex)                  # モデルパラメータのコピー
#alex.to_gpu()                           # gpuを使う場合

classifier = Classifier(alex)
print('exit!')
exit()
optimizer = optimizers.MomentumSGD(lr=0.0005)  # パラメータの学習方法は慣性項付きの確率的勾配法で, 学習率は0.0005に設定.
optimizer.setup(classifier)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))  # l2正則化

#train_data = ImageData('clf_train_images', 'clf_train_master.tsv')  # 学習データの読み込み
#test_data = ImageData('clf_test_images', 'clf_test.tsv')            # 評価データの読み込み
train_data = ImageData('data', 'clf_train_master.tsv')  # 学習データの読み込み
test_data = ImageData('data', 'clf_test.tsv')            # 評価データの読み込み

classifier, optimizer = train_val(train_data, classifier, optimizer)               # 学習＋検証
p = predict(test_data, classifier)                                                 # 予測値(確率値)の出力
pd.DataFrame(p.argmax(axis=1)).to_csv('sample_submit.csv', header=None)            # 予測結果を応募用ファイルとして出力
