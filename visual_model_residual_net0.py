import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import tensorflow as tf
# import data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

# data setting
y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)
train_data = np.asarray(x_train, dtype=np.float32)
train_labels =np.asarray(y_train, dtype=np.float32)
eval_data = np.asarray(x_test, dtype=np.float32)
eval_labels = np.asarray(y_test, dtype=np.float32)
# one hot encoding
y_train = tflearn.data_utils.to_categorical(y_train, 100)
y_test = tflearn.data_utils.to_categorical(y_test, 100)
# image argumentation
image_preprocessing = tflearn.ImagePreprocessing()
image_preprocessing.add_featurewise_zero_center(per_channel=True)
image_argumentation = tflearn.ImageAugmentation()
image_argumentation.add_random_flip_leftright()
image_argumentation.add_random_crop([32, 32], padding=4)
# build 56 layers residual layers
n = 9
net = tflearn.input_data(shape=[None, 32, 32, 3],
                          data_preprocessing=image_preprocessing,
                          data_augmentation=image_argumentation)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'elu')
image_embedding = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(image_embedding, 100, activation='softmax')
optimizer = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=optimizer,
                         loss='categorical_crossentropy')

# define model
model = tflearn.DNN(net, checkpoint_path='model_resnet_cifar100',
                    max_checkpoints=10, tensorboard_verbose=3)
# model training
model.fit(train_data, y_train, n_epoch=100, validation_set=(eval_data, y_test),
          snapshot_epoch=False, snapshot_step=500,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnet_cifar100')
# save model
model.save('resnet_cifar100_model.tflearn')

# reload pretrain model
model.load('resnet_cifar100_model.tflearn')
# Define model with removing softmax layer
embedding_out = tflearn.DNN(image_embedding, session=model.session)


# retrieve image vector
y_embedding_test = np.zeros((10000,1,64))
y_embedding_train = np.zeros((50000,1,64))
for i in range(10000):
   a = eval_data[i].reshape(1,32,32,3)
   y_embedding_test[i] = embedding_out.predict(a)
   if i%1000==0:
      print(i)
for i in range(50000):
   a = train_data[i].reshape(1,32,32,3)
   y_embedding_train[i] = embedding_out.predict(a)
   if i%1000==0:
      print(i)

# save image embedding as .npy file
y_embedding_train_output = np.reshape(y_embedding_train,(50000*1*64))
np.save('y_embedding_train0.npy',y_embedding_train_output )
y_embedding_test_output = np.reshape(y_embedding_test,(10000*1*64))
print(y_embedding_test_output.shape)
np.save('y_embedding_test0.npy',y_embedding_test_output )

