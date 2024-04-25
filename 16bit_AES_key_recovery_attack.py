import numpy as np
from AES_128_batch import *
from os import urandom
from tensorflow.keras.models import load_model
import time
from copy import deepcopy
# 多核CPU 环境
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def attack_with_one_nd(n, cts, last_subkey=None, kg_bit_num=None, nd=None, c=None,data_index = None, key_index=None):
    
    ct0,ct1 = cts[0], cts[1]
    kg_guess_num = 2**(int(kg_bit_num/2))
    # kg_guess_num = 2**6

    sur_kg = []
    sur_kg_scores = []

    # print("True_subkey")
    # print(last_subkey)
    
    for kg2 in range(kg_guess_num):
        for kg1 in range(kg_guess_num):
    
            guess_key = last_subkey
            # 根据状态的位置，修改的密钥位要对应状态的0和1位
            # 2轮区分器[0,1]bytes对应的3轮密钥位置时[0,13]
            guess_key[0,int(key_index[0]%4),int(key_index[0]/4)] = kg1;guess_key[0,int(key_index[1]%4),int(key_index[1]/4)] = kg2

            guess_subkey = np.tile(guess_key, (n,1,1))
            # 加密代码这里转置了
            ct0_state = ct0.reshape(n,4,4).transpose((0, 2, 1))
            ct0_state = add_round_key(ct0_state, guess_subkey)
            ct0_state = inv_shift_rows(ct0_state)
            ct0_state = inv_sub_bytes(ct0_state) 
            ct0_state = ct0_state.transpose((0, 2, 1)).reshape(-1, 16)

            ct1_state = ct1.reshape(n,4,4).transpose((0, 2, 1))
            ct1_state = add_round_key(ct1_state, guess_subkey)
            ct1_state = inv_shift_rows(ct1_state)
            ct1_state = inv_sub_bytes(ct1_state)
            ct1_state = ct1_state.transpose((0, 2, 1)).reshape(-1, 16)

            # 抽取2个bytes数据
            byte_num = 2;i = data_index
            ct0_state = ct0_state[:,i*byte_num:(i+1)*byte_num];
            ct1_state = ct1_state[:,i*byte_num:(i+1)*byte_num];

            raw_x = np.hstack((ct0_state, ct1_state))
            x = convert_to_binary(raw_x,byte_num)
            z = nd.predict(x,batch_size=100)
            z = z.flatten()
            # print("z.shape=",z.shape)
            z = np.log2(z/(1-z))
            z = np.mean(z)
            # print("z = ",z)
            
            if z > c:
                sur_kg.append(kg1+(kg2<<8))
                # print("guess_key = ",guess_key)
                print("z = ",z)
                sur_kg_scores.append(z)
                
                # return sur_kg, sur_kg_scores
                
            
    return sur_kg, sur_kg_scores


def select_top_k_candidates(sur_kg, kg_scores, k=10):
    num = len(sur_kg)
    # print("len(sur_kg) = ",len(sur_kg))
    tp = deepcopy(kg_scores)
    tp.sort(reverse=True)
    if num > k:
        base = tp[k]
    else:
        base = tp[num-1]
    print("base=",base)
    filtered_subkey = []
    for i in range(num):
        if kg_scores[i] >= base:
            # print("kg_scores[i] = ",kg_scores[i])
            filtered_subkey.append(sur_kg[i])
    return filtered_subkey



def attack_with_dual_NDs(t=100, n=None, nr=None, nds=None, c=None,data_index=None, key_index=None):

    for i in range(t):

        start = time.time()

        print('Test:', i)
        # 随机生成密钥
        key = np.frombuffer(urandom(16), dtype=np.uint8).reshape(-1, 16)
        ks = key_expansion(key, nr)
        num = 0
        while 1:
            if num > 2**0:
                num = -1
                break
            
            # 生成明文和密文
            pt0 = np.frombuffer(urandom(16*n), dtype=np.uint8).reshape(-1, 16)
            pt1 = pt0.copy();pt1[:,0] = pt1[:,0] ^ diff
            # 密钥恢复的密文不带MC
            ct0 = aes_encrypt(pt0, key, nr,flag=False);ct1 = aes_encrypt(pt1, key, nr,flag=False)
            
            last_subkey=deepcopy(ks[-1])
            sur_kg, kg_scores = attack_with_one_nd(n,[ct0,ct1],last_subkey,kg_bit_num=16, nd=nds, c=c,data_index=data_index,key_index=key_index)
            num += 1

            # print('\r {} plaintext structures generated\n'.format(num), end='')
            if len(sur_kg) == 0:
                continue
            else:
                print(' ')
                print(len(sur_kg), ' subkeys survive')
                break
        if num == -1:
            print(' ')
            print('this trial fails.')
            # print('{} plaintext structures are generated.'.format(data_num))
            print('the time consumption is ', time.time() - start)
            continue
        # select the best candidate of the first stage
        kg = select_top_k_candidates(sur_kg, kg_scores)
        print("len(kg)=",len(kg))
        sur_kg = kg

        end = time.time()

        print('the final surviving subkey guesses are ', sur_kg)
        # compare returned keys and true subkey
        sk = ks[-1]
        num = len(sur_kg)
        for i in range(num):
            # 2轮区分器[0,1]bytes对应的3轮密钥位置时[0,13]
            print('difference between surviving kg and sk is ', bin((sk[0,int(key_index[0]%4),int(key_index[0]/4)]) ^ int(sur_kg[i] % 2**8)),bin((sk[0,int(key_index[1]%4),int(key_index[1]/4)]) ^ int(sur_kg[i] >> 8)))
        # print('{} plaintext structures are generated.'.format(data_num))
        print('the time consumption is ', end - start)


# if __name__ == '__main__':
    
# nd是区分器
for i in range(8):
    print("index = ",i)
    file_and_key_index = [
        ["AES16_data_index_0_best_model_2r_depth5_num_epochs20_acc_0.9980679750442505.h5",[0,13]],
        ["AES16_data_index_1_best_model_2r_depth5_num_epochs20_acc_0.9980459809303284.h5",[10,7]],
        ["AES16_data_index_2_best_model_2r_depth5_num_epochs20_acc_0.9979869723320007.h5",[4,1]],
        ["AES16_data_index_3_best_model_2r_depth5_num_epochs20_acc_0.9981039762496948.h5",[14,11]],
        ["AES16_data_index_4_best_model_2r_depth5_num_epochs20_acc_0.9980419874191284.h5",[8,5]],
        ["AES16_data_index_5_best_model_2r_depth5_num_epochs20_acc_0.9980940222740173.h5",[2,15]],
        ["AES16_data_index_6_best_model_2r_depth5_num_epochs20_acc_0.9980159997940063.h5",[12,9]],
        ["AES16_data_index_7_best_model_2r_depth5_num_epochs20_acc_0.9980049729347229.h5",[6,3]]
    ]
    
    nd = load_model(file_and_key_index[i][0])

    diff=0x80
    # t是实验次数，用来估计成功率
    attack_with_dual_NDs(t=1, n=3, nr=3, nds=nd, c=7, data_index=i, key_index = file_and_key_index[i][1])

