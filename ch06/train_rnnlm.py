# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm
###########################################################
# ch05/train.pyと共通する箇所
# - ミニバッチを “シーケンシャル” に作り、
# - モデルの順伝播と逆伝播を呼び、
# - オプティマイザで重みを更新し、
# - パープレキシティを評価する
###########################################################

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNNの隠れ状態ベクトルの要素数
time_size = 35  # RNNを展開するサイズ
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 1. 勾配クリッピングを適用して学習
# RnnlmTrainer クラスを使ってモデルの学習を行います。
# RnnlmTrainer クラスの fit() メソッドは、モデルの勾配を求め、
# モデルのパラメータを更新します。
# このとき、引数の max_grad を指定することで、勾配クリッピング(本文6.1.4節参照)が適用されます。
#
trainer.fit(
    xs,
    ts,
    max_epoch,
    batch_size,
    time_size,
    max_grad,
    # 今回はデータサイズが大きいので エポックごとの評価ではなく、
    # 20 イテレーションごとに評価
    eval_interval=20)
trainer.plot(ylim=(0, 500))

# 2. テストデータで評価
# 学習が終わった後にテストデータを使用してパープレキシティを評価
# 評価時、モデルの状態(LSTM の隠れ状態と記憶セル)をリセットして評価を行う。
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# 3. パラメータの保存
# 学習後のパラメータを外部ファイルに保存します。
# 7章で、学習後の重みパラメータを使って文章生成をする際に使用します。
model.save_params()
