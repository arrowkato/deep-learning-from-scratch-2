# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Softmax


class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1)  #.repeat(T, axis=1)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)  #.repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape
        # dsはsを逆伝播させるときの行列
        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr  # アダマール積の逆伝播
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        # 下から順にforwardするので、Attention Weight, Weight Sumの順
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        # 結果をあとから参照するため
        self.attention_weight = a
        return out

    def backward(self, dout):
        # 上から順にbackwardするので、Weight Sum, Attention Weightの順
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        '''
        hs_enc: encoderから渡される値-> hs
        hs_dec: 下(LSTM)から順伝播される値
        '''
        # T:seq2seqの単語数 e.g. 5: <eos> I am a cat
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            # hs_encはすべてのAttentionレイヤで同じものを使う
            # hs_decつまり、LSTMからの値は、同じ時間TのLSTMから伝播してくる値を使う
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        # Affineレイヤに順伝播させる値を返す
        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        # doutと同じ行と列で、入っている値は適当な行列を作る
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            # 時刻:Tのときの dout からdhs_encとdhs_decの逆伝播を求める　図8-20参照
            dhs, dh = layer.backward(dout[:, t, :])
            # dhs_enc自体は、T個あるAttentionレイヤすべてから逆伝播した値
            dhs_enc += dhs
            # 時刻:Tのときのdoutから逆伝播した値をdhs_decとして受け取る
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
