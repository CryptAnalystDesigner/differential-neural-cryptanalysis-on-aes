# Author: @YaoYiran
import numpy as np
from os import urandom
import time
import cProfile
import pstats

# S-box used in SubBytes step
s_box = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

# Inverse S-box used in InvSubBytes step
inv_s_box = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
])

# Round constant for key expansion
r_con = np.array([
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36
])

def sub_bytes(state):
    return s_box[state]

def inv_sub_bytes(state):
    return inv_s_box[state]

def shift_rows(state):
    # Row 1 shifted by 1 (to the right)
    state[:, 1, :] = state[:, 1, [1, 2, 3, 0]]

    # Row 2 shifted by 2 (to the right)
    state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]

    # Row 3 shifted by 3 (to the right)
    state[:, 3, :] = state[:, 3, [3, 0, 1, 2]]

    return state

def inv_shift_rows(state):
    # Row 1 shifted by 1
    state[:, 1, :] = state[:, 1, [3, 0, 1, 2]]

    # Row 2 shifted by 2
    state[:, 2, :] = state[:, 2, [2, 3, 0, 1]]

    # Row 3 shifted by 3
    state[:, 3, :] = state[:, 3, [1, 2, 3, 0]]

    return state

# Precompute the lookup tables
galois_lookup = np.zeros((256, 256), dtype=np.uint8)

for a in range(256):
    for b in range(256):
        p = 0
        x = a
        y = b
        for counter in range(8):
            if y & 1:
                p ^= x
            hi_bit_set = x & 0x80
            x <<= 1
            if hi_bit_set:
                x ^= 0x11b
            y >>= 1
        galois_lookup[a, b] = p % 256

# Modify galois_mult to use the lookup table
def galois_mult(a, b):
    return galois_lookup[a, b]

# Modify galois_matrix_mult to use the new function
def galois_matrix_mult(mat, state):
    tmp = np.zeros_like(state, dtype=np.uint8)
    
    for i in range(4):
        for k in range(4):
            tmp[:, i, :] ^= galois_lookup[mat[i, k], state[:, k, :]]
    
    return tmp

def mix_columns(state):
    mix = np.array([
        [2, 3, 1, 1],
        [1, 2, 3, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2]
    ])
    return galois_matrix_mult(mix, state)

def inv_mix_columns(state):
    inv_mix = np.array([
        [14, 11, 13, 9],
        [9, 14, 11, 13],
        [13, 9, 14, 11],
        [11, 13, 9, 14]
    ])
    return galois_matrix_mult(inv_mix, state)

def add_round_key(state, round_key):
    return np.bitwise_xor(state, round_key)

def key_expansion(key, nr):
    n = key.shape[0]
    key = key.reshape(n, 4, 4)
    w = np.zeros((n, 4 * (nr + 1), 4), dtype=np.uint8)
    w[:, :4] = key

    for i in range(4, 4 * (nr + 1)):
        temp = w[:, i-1].copy()
        if i % 4 == 0:
            temp = np.roll(temp, -1, axis=1)
            temp = s_box[temp]
            temp[:, 0] ^= r_con[i//4]
        w[:, i] = w[:, i-4] ^ temp

    return w.reshape((n, nr + 1, 4, 4)).transpose((1, 0, 3, 2))

def aes_encrypt(plain_text, key, nr=10,flag=False):
    n = plain_text.shape[0]
    # 把数据转成一个矩阵
    state = plain_text.reshape(n, 4, 4).transpose((0, 2, 1))
    round_keys = key_expansion(key, nr)

    state = add_round_key(state, round_keys[0])

    for i in range(1, nr):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[i])

    state = sub_bytes(state)
    state = shift_rows(state)
    # 注意原来的2轮区分器是用了带MC的数据训练的
    if flag:
        state = mix_columns(state)
    state = add_round_key(state, round_keys[nr])

    return state.transpose(0, 2, 1).reshape(-1, 16)

def aes_decrypt(cipher_text, key, nr=10,flag=False):
    n = cipher_text.shape[0]
    state = cipher_text.reshape(n, 4, 4).transpose((0, 2, 1))
    # 密钥也转置了
    round_keys = key_expansion(key, nr)
    
    state = add_round_key(state, round_keys[nr])
    if flag:
        state = inv_mix_columns(state)
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)

    for i in range(nr-1, 0, -1):
        state = add_round_key(state, round_keys[i])
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)

    state = add_round_key(state, round_keys[0])

    return state.transpose(0, 2, 1).reshape(-1, 16)

def convert_to_binary(arr,byte_num=16):
    # print(arr.shape[0])
    X = np.zeros((byte_num*8*2, arr.shape[0]), dtype=np.uint8)
    for i in range(X.shape[0]):
        index = i // 8
        offset = 8 - (i % 8) - 1
        X[i] = (arr[:, index] >> offset) & 1
    return X.transpose()

# 差分分析数据
# 单差分
def generate_train_data(n, nr, diff=0x80,data_index=0,flag=False):

    # generate labels
    # half 0, half 1
    Y = np.frombuffer(urandom(n), dtype=np.uint8) & 1
    # generate plaintext
    # AES是16个bytes
    pt0 = np.frombuffer(urandom(16*n), dtype=np.uint8).reshape(-1, 16)
    # print("pt0.shape=",pt0.shape)
    pt1 = pt0.copy()
   
    pt1[:, 0] ^= diff
 
    pt1[Y==0] = np.frombuffer(urandom(16*np.sum(Y==0)), dtype=np.uint8).reshape(-1, 16)
    # generate keys
    keys = np.frombuffer(urandom(16*n), dtype=np.uint8).reshape(-1, 16)
    # generate ciphertext

    ct0 = aes_encrypt(pt0, keys, nr,flag) # (n, 16)
    ct1 = aes_encrypt(pt1, keys, nr,flag) # (n, 16)
    # print(ct0)
    # 全部密文数据
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct)
 

    # 抽取一列数据
    # byte_num = 4
    # # 抽取第i列
    # i = data_index
    # ct0 = ct0[:,i*byte_num:(i+1)*byte_num];ct1 = ct1[:,i*byte_num:(i+1)*byte_num];
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct,byte_num)
    
    # 抽取一行数据
    # byte_num = 4    
    # i = data_index
    # ct0 = ct0[:,[j for j in range(16)][i:i+13:byte_num]];ct1 = ct1[:,[j for j in range(16)][i:i+13:byte_num]];
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct,byte_num)
    
    # 抽取某些bytes
    # byte_num = 4
    # ct0 = ct0[:,[0,1,2,3]];ct1 = ct1[:,[0,1,2,3]];
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct,byte_num)
    
    # 抽取一个byte
    # i = data_index
    # ct0 = ct0[:,i:i+1];ct1 = ct1[:,i:i+1];
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct,1)
    
    # 抽取2个bytes数据
    # byte_num = 2
    # # 抽取第i列
    # i = data_index
    # ct0 = ct0[:,i*byte_num:(i+1)*byte_num];ct1 = ct1[:,i*byte_num:(i+1)*byte_num];
    # ct = np.hstack((ct0, ct1))
    # X = convert_to_binary(ct,byte_num)
    
    # 抽取不连续的2个bytes数据
    byte_num = 2
    # 抽取第i列
    ct0 = ct0[:,data_index];ct1 = ct1[:,data_index];
    ct = np.hstack((ct0, ct1))
    X = convert_to_binary(ct,byte_num)
    
    return X, Y




if __name__ == "__main__":
    start = time.time()
    # n = 10**3
    X, Y = generate_train_data(10**6, 3,data_index=0,flag=False)
    print("X shape = ", X.shape)
    end = time.time()
    print(end - start)
