# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention


class AttentionEncoder(Encoder):
    '''
    親クラスのEncoderの実装は、ch07/seq2seq.py参照
    実装の解説は7.3.1節
    '''
    def forward(self, xs):
        '''
        前章の Encoder クラスの forward() メソッドは、
        LSTM レイヤの最後の隠れ状態ベクトルだけを返しました。
        それに対して、今回はすべての隠れ状態を返します。
        '''
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        return hs  # 親クラスだと return hs[:, -1, :]

    def backward(self, dhs):
        '''
        wae
        '''
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        # TimeAttensionレイヤ追加 ch07/seq2seq.pyとの違い
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        # TimeLSTMレイヤとAffineの間に、TimeAttensionレイヤを追加
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        # 図8-21参照
        # Affineレイヤの入力は下記の2つレイヤの結果なので、concatenateで行列を結合
        # c: TimeAttensionの結果
        # dec_hs: TimeLSTMの結果
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        # dout: affine -> TimeAttension and TimeLSTM
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        # dc: TimeAffine -> TimeAttension
        # ddec_hs0: -> TimeAffine -> Time LSTM
        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        # denc_hs: TimeAttension -> Encoder
        # ddec_hs1: TimeAttension -> TimeLSTM
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        # TimeAffine and TimeAttension -> TimeLSTM
        # 下手に展開しない方がわかりやすいかも
        # dout = self.lstm.backward(ddec_hs0 + ddec_hs1)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)

        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        '''
        文章の生成。ch07/seq2seq.generate()との違いは、
        TimeAttensionレイヤが加わっただけ。
        '''
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        # EncoderではなくAttentionEncoderを使用
        self.encoder = AttentionEncoder(*args)
        # DecoderではなくAttensionDecoderを使用
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    # forward(), backward(), generate()はSeq2seqクラスを継承
