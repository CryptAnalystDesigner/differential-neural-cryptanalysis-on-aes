from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, Conv1D,Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pickle import dump
import tensorflow as tf
from AES_128_batch import generate_train_data
import numpy as np
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

bs = 1000
wdir = './3_round_good_trained_nets/'

if(not os.path.exists(wdir)):
  os.makedirs(wdir)

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

word_size=None

#make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=16, num_outputs=1, d0 = 2048, d1=64, d2=64, word_size=word_size, ks=3, depth=5, reg_param=0.0001, final_activation='sigmoid'):

    inp = Input(shape=(int(num_blocks * word_size),))
    rs = Reshape((int(num_blocks), word_size))(inp)
    perm = Permute((2, 1))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    # conv02 = Conv1D(num_filters, kernel_size=3, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # conv03 = Conv1D(num_filters, kernel_size=5, padding='same',
    #                 kernel_regularizer=l2(reg_param))(perm)
    # c2 = concatenate([conv01, conv02, conv03], axis=-1)
    
    c2 = conv01
    
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters*1, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*1, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        # if ks < 7:
        #     ks += 2

    flatten = Flatten()(shortcut)
    # dense0 = Dense(d0, kernel_regularizer=l2(reg_param))(flatten)
    # dense0 = BatchNormalization()(dense0)
    # dense0 = Activation('relu')(dense0)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flatten)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)


def train_distinguisher(num_epochs,data_index=None,nr=7, depth=1,word_size=16*8,flag=True):
    
    print("num_rounds = ", nr)

    strategy = tf.distribute.MirroredStrategy(
        devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3","/gpu:4"])
    # print('Number of devices: %d' % strategy.num_replicas_in_sync)  # 输出设备数量
    batch_size = bs * strategy.num_replicas_in_sync

    with strategy.scope():
        net = make_resnet(depth=depth, reg_param=10**-5,word_size=word_size)
        # net.summary()
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    
    # flag=True表示训练区分器的数据有MC
    X, Y = generate_train_data(10**7, nr,data_index=data_index,flag=flag)
    X_eval, Y_eval = generate_train_data(10**6, nr,data_index=data_index,flag=flag)
 
    
    src = wdir+'AES'+str(word_size)+'_data_index_'+str(data_index)+'_best_model_'+str(nr)+'r_depth'+str(depth)+"_num_epochs"+str(num_epochs)
    check = make_checkpoint(src+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr,check])
     
    dump(h.history, open(wdir+'AES'+str(word_size)+'_data_index_'+str(data_index)+'_hist'+str(nr)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    
    # 重命名文件
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src +'.h5' , dst+'.h5')
    


if __name__ == "__main__":
    
    # 现在只用了2个字节
    # word_size = 2*8
    # rounds=[3]
    # # index为选取的密文bytes位置
    # for index in range(8):
    #     print("index = ",index)
    #     for r in rounds:
    #         train_distinguisher(num_epochs=20,data_index=index,nr=r, depth=5,word_size=word_size,flag=False)

    # 使用全部密文
    # train_distinguisher(num_epochs=20,nr=3, depth=5,word_size=16*8,flag=True)
    word_size = 2*8
    train_distinguisher(num_epochs=20,data_index=[0,13],nr=3, depth=5,word_size=word_size,flag=False)
    # 随意2个字节
    
    # 用4个字节
    # word_size = 4*8
    # rounds=[3]
    # # index为选取的密文bytes位置
    # # index = 0
    # for index in range(4):
    #     print("index = ",index)
    #     for r in rounds:
    #         train_distinguisher(num_epochs=20,data_index=index,nr=r, depth=5,word_size=word_size,flag=False)
    
    # 抽1个字节
    # word_size = 1*8
    # rounds=[2]
    # # index为选取的密文bytes位置
    # # index = 0
    # for index in range(16):
    #     print("index = ",index)
    #     for r in rounds:
    #         train_distinguisher(num_epochs=20,data_index=index,nr=r, depth=5,word_size=word_size,flag=True)