import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import DynamicPositionEmbedding, SelfAttentionBlock
from utils.utils1 import get_reward1

class ChordConditionedMelodyTransformer(nn.Module):
    def __init__(self, num_pitch=89, frame_per_bar=16, num_bars=8,
                 chord_emb_size=128, pitch_emb_size=128, hidden_dim=128,
                 key_dim=128, value_dim=128, num_layers=6, num_heads=4,
                 input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0):
        super(ChordConditionedMelodyTransformer, self).__init__()

        self.max_len = frame_per_bar * num_bars
        self.frame_per_bar = frame_per_bar
        self.num_chords = 12
        self.num_pitch = num_pitch
        self.num_rhythm = 3

        # self.rhythm_emb_size = chord_emb
        self.chord_emb_size = chord_emb_size
        self.rhythm_emb_size = pitch_emb_size // 8 #32
        self.pitch_emb_size = pitch_emb_size #256
        self.chord_hidden = 7 * (pitch_emb_size // 32)  # 2 * chord_hidden + rhythm_emb = rhythm_hidden     chord_hidden=56
        self.rhythm_hidden = 9 * (pitch_emb_size // 16)  # 2 * chord_hidden + rhythm_hidden = pitch_emb     rhythm_hidden=144
        self.hidden_dim = hidden_dim  #hidden_dim=512

        self.rhythm1_hidden=3 * self.pitch_emb_size + self.rhythm_emb_size
        self.hidden1_dim= 7 * self.pitch_emb_size+self.rhythm_emb_size

        # embedding layer
        self.chord_emb = nn.Parameter(torch.randn(self.num_chords, self.chord_emb_size,
                                                  dtype=torch.float, requires_grad=True))
        self.rhythm_emb = nn.Embedding(self.num_rhythm, self.rhythm_emb_size)
        self.pitch_emb = nn.Embedding(self.num_pitch, self.pitch_emb_size)

        self.rhythm1_emb = nn.Embedding(self.num_rhythm, self.rhythm_emb_size)
        self.pitch1_emb = nn.Embedding(self.num_pitch, self.pitch_emb_size)
        #self.pitch1_emb = nn.Parameter(torch.randn(48, self.pitch_emb_size,
        #                                          dtype=torch.float, requires_grad=True))

        lstm_input = self.chord_emb_size
        self.chord_lstm = nn.LSTM(lstm_input, self.chord_hidden, num_layers=1,
                                  batch_first=True, bidirectional=True)
        self.rhythm_pos_enc = DynamicPositionEmbedding(self.rhythm_hidden, self.max_len)
        self.pos_encoding = DynamicPositionEmbedding(self.hidden_dim, self.max_len)
        self.rhythm1_pos_enc = DynamicPositionEmbedding(self.rhythm1_hidden, self.max_len)
        self.pos1_encoding = DynamicPositionEmbedding(self.hidden1_dim, self.max_len)

        # embedding dropout
        self.emb_dropout = nn.Dropout(input_dropout)

        # Decoding layers
        rhythm_params = (
            2 * self.chord_hidden + self.rhythm_emb_size,
            self.rhythm_hidden,
            key_dim // 4,
            value_dim // 4,
            num_heads,
            self.max_len,
            False,  # include succeeding elements' positional embedding also
            layer_dropout,
            attention_dropout
        )
        self.rhythm_decoder = nn.ModuleList([
            SelfAttentionBlock(*rhythm_params) for _ in range(num_layers)
        ])

        pitch_params = (
            2 * self.pitch_emb_size,
            self.hidden_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            False,  # preceding only
            layer_dropout,
            attention_dropout
        )
        self.pitch_decoder = nn.ModuleList([
            SelfAttentionBlock(*pitch_params) for _ in range(num_layers)
        ])



        rhythm1_params = (
            3 * self.pitch_emb_size + self.rhythm_emb_size,
            self.rhythm1_hidden,
            key_dim // 4,
            value_dim // 4,
            num_heads,
            self.max_len,
            False,  # include succeeding elements' positional embedding also
            layer_dropout,
            attention_dropout
        )
        self.rhythm1_decoder = nn.ModuleList([
            SelfAttentionBlock(*rhythm1_params) for _ in range(num_layers)
        ])

        pitch1_params = (
            7 * self.pitch_emb_size+self.rhythm_emb_size,
            self.hidden1_dim,
            key_dim,
            value_dim,
            num_heads,
            self.max_len,
            True,  # preceding only
            layer_dropout,
            attention_dropout
        )
        self.pitch1_decoder = nn.ModuleList([
            SelfAttentionBlock(*pitch1_params) for _ in range(num_layers)
        ])
        # output layer
        self.rhythm_outlayer = nn.Linear(self.rhythm_hidden, self.num_rhythm)
        self.pitch_outlayer = nn.Linear(self.hidden_dim, self.num_pitch)
        self.rhythm1_outlayer = nn.Linear(self.rhythm1_hidden, self.num_rhythm)
        self.pitch1_outlayer = nn.Linear(self.hidden1_dim, self.num_pitch)
        #self.pitch1_outlayer = nn.Linear(self.hidden1_dim, self.num_pitch-2)
        #self.pitch1_outlayer1=nn.Linear(self.num_pitch-2,num_pitch-2)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax=nn.Softmax(dim=-1)

    def init_lstm_hidden(self, batch_size):
        h0 = Variable(torch.zeros(2, batch_size, self.chord_hidden))
        c0 = Variable(torch.zeros(2, batch_size, self.chord_hidden))
        return (h0, c0)

    # rhythm : time_len + 1   (input & target)
    # pitch : time_len      (input only)
    # chord : time_len + 1  (input & target)
    def forward(self, rhythm, pitch, chord, rhythm1, pitch1, attention_map=False, rhythm_only=False, ):
        # chord_hidden : time_len   (target timestep)
        chord_hidden = self.chord_forward(chord)

        rhythm_dec_result = self.rhythm_forward(rhythm[:, :-1], chord_hidden, attention_map, masking=True)
        rhythm_out = self.rhythm_outlayer(rhythm_dec_result['output'])
        #rhythm_out = self.log_softmax(rhythm_out)
        result = {'rhythm': rhythm_out}

        if not rhythm_only:
            rhythm_enc_result = self.rhythm_forward(rhythm[:, 1:], chord_hidden, attention_map, masking=False)
            rhythm_emb = rhythm_enc_result['output']
            #pitch (16,129)
            pitch_emb = self.pitch_emb(pitch[:, :-1]) #(16,128,256)(batchsize,seq_length,embbeding_size)
            emb = torch.cat([pitch_emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
            emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
            pitch_output = self.pitch_forward(emb, attention_map)

            output = self.pitch_outlayer(pitch_output['output'])
            #output = self.log_softmax(output)
            result['pitch'] =output #(16,128,50)

            #将pitch送入encoder模型 生成pitch的hidden
            pitch_emb = self.pitch_emb(pitch[:, 1:])
            emb = torch.cat([pitch_emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
            emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
            pitch_enc_result = self.pitch_forward(emb, attention_map, masking=False)

            pitch_hidd = pitch_enc_result['output']
            #将rhythm1编码
            rhythm1_emb = self.rhythm1_emb(rhythm1[:, :-1])
            rhythm1_emb = torch.cat([rhythm1_emb, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
            rhythm1_emb /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
#将处理好的数据（包含rhythm1，chord信息，第一个rhythm信息，第一个pitch信息）放入decoder模型生成rhythm1
            rhythm1_dec_result = self.rhythm1_forward(rhythm1_emb, attention_map, masking=True)
            rhythm_out = self.rhythm1_outlayer(rhythm1_dec_result['output'])
            #rhythm_out = self.log_softmax(rhythm_out)
            result['rhythm1'] = rhythm_out

            rhythm1_emb = self.rhythm1_emb(rhythm1[:, 1:])
            rhythm1_emb = torch.cat([rhythm1_emb, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
            rhythm1_emb /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
            rhythm1_enc_result=self.rhythm1_forward(rhythm1_emb,attention_map,masking=False)
            rhythm1_hidd=rhythm1_enc_result['output']
            pitch1_emb=self.pitch1_emb(pitch1[:, :-1])
            #pitch1_emb = torch.matmul(pitch1[:, :-1].float(), self.pitch1_emb) #(16,128,48)*(48,256)=(16,128,256)
            pitch1_emb=torch.cat([pitch1_emb,chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd,rhythm1_hidd], -1)
            pitch1_emb /= torch.sqrt(torch.tensor(self.hidden1_dim, dtype=torch.float))
            pitch1_output=self.pitch1_forward(pitch1_emb,attention_map)
            output = self.pitch1_outlayer(pitch1_output['output'])
            # output = self.pitch1_outlayer(pitch1_output['output']) #推测(16,128,48)
            # output = self.pitch1_outlayer1(output) #推测(16,128,48)
            # output = self.sigmoid(output) #推测(16,128,48)
            #output = self.log_softmax(output)
            #output= self.softmax(output)
            result['pitch1'] = output


            if attention_map:
                result['weights_rdec'] = rhythm_dec_result['weights']
                result['weights_renc'] = rhythm_enc_result['weights']
                result['weights_pdec'] = pitch_output['weights']
                result['weights_penc'] = pitch_enc_result['weights']
                result['weights_r1dec'] = rhythm1_dec_result['weights']
                result['weights_r1enc'] = rhythm1_enc_result['weights']
                result['weights_p1dec'] = pitch1_output['weights']

        return result

    def chord_forward(self, chord):
        size = chord.size() #(16,129,12)
        chord_emb = torch.matmul(chord.float(), self.chord_emb) #(16,129,12)*(12,128)=(16,129,128)

        h0, c0 = self.init_lstm_hidden(size[0])
        self.chord_lstm.flatten_parameters()
        chord_out, _ = self.chord_lstm(chord_emb, (h0.to(chord.device), c0.to(chord.device)))
        chord_for = chord_out[:, 1:, :self.chord_hidden]
        chord_back = chord_out[:, 1:, self.chord_hidden:]
        return chord_for, chord_back

    def rhythm_forward(self, rhythm, chord_hidden, attention_map=False, masking=True):
        rhythm_emb = self.rhythm_emb(rhythm)
        rhythm_emb = torch.cat([rhythm_emb, chord_hidden[0], chord_hidden[1]], -1)
        rhythm_emb /= torch.sqrt(torch.tensor(self.rhythm_hidden, dtype=torch.float))
        rhythm_emb = self.rhythm_pos_enc(rhythm_emb)
        rhythm_emb = self.emb_dropout(rhythm_emb)

        weights = []
        for _, layer in enumerate(self.rhythm_decoder):
            result = layer(rhythm_emb, attention_map, masking)
            rhythm_emb = result['output']
            if attention_map:
                weights.append(result['weight'])

        result = {'output': rhythm_emb}
        if attention_map:
            result['weights'] = weights

        return result

    def pitch_forward(self, pitch_emb, attention_map=False, masking=True):
        emb = self.pos_encoding(pitch_emb)
        emb = self.emb_dropout(emb)

        # pitch model
        pitch_weights = []
        for _, layer in enumerate(self.pitch_decoder):
            pitch_result = layer(emb, attention_map, masking)
            emb = pitch_result['output']
            if attention_map:
                pitch_weights.append(pitch_result['weight'])



        result = {'output': emb}
        if attention_map:
            result['weights'] = pitch_weights

        return result

    def rhythm1_forward(self, rhythm1_emb, attention_map=False, masking=True):
        emb = self.rhythm1_pos_enc(rhythm1_emb)
        emb = self.emb_dropout(emb)

        # rhythm1 model
        rhythm1_weights = []
        for _, layer in enumerate(self.rhythm1_decoder):
            rhythm1_result = layer(emb, attention_map, masking)
            emb = rhythm1_result['output']
            if attention_map:
                rhythm1_weights.append(rhythm1_result['weight'])



        result = {'output': emb}
        if attention_map:
            result['weights'] = rhythm1_weights

        return result

    def pitch1_forward(self, pitch_emb, attention_map=False, masking=True):
        emb = self.pos1_encoding(pitch_emb)
        emb = self.emb_dropout(emb)

        # pitch model
        pitch_weights = []
        for _, layer in enumerate(self.pitch1_decoder):
            pitch_result = layer(emb, attention_map, masking)
            emb = pitch_result['output']
            if attention_map:
                pitch_weights.append(pitch_result['weight'])

        # output layer
        # output = self.pitch_outlayer(emb)
        # output = self.log_softmax(output)

        result = {'output': emb}
        if attention_map:
            result['weights'] = pitch_weights

        return result

    def sampling(self, prime_rhythm, prime_pitch, prime_rhythm1, prime_pitch1, chord, topk=None, attention_map=False):
        chord_hidden = self.chord_forward(chord)
        # ----------------------------------------rhythmの生成開始------------------------------------------------------------
        # batch_size * prime_len * num_outputs
        batch_size = prime_pitch.size(0)
        pad_length = self.max_len - prime_rhythm.size(1)
        rhythm_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(prime_rhythm.device)
        rhythm_result = torch.cat([prime_rhythm, rhythm_pad], dim=1)

        # sampling phase
        for i in range(prime_rhythm.size(1), self.max_len):
            rhythm_dec_result = self.rhythm_forward(rhythm_result, chord_hidden, attention_map, masking=True)
            rhythm_out = self.rhythm_outlayer(rhythm_dec_result['output'])
            rhythm_out = self.log_softmax(rhythm_out) #(5,128,3)
            if topk is None:
                idx = torch.argmax(rhythm_out[:, i - 1, :], dim=1)
            else:
                top3_probs, top3_idxs = torch.topk(rhythm_out[:, i - 1, :], 3, dim=-1)
                idx = torch.gather(top3_idxs, 1, torch.multinomial(F.softmax(top3_probs, dim=-1), 1)).squeeze()
            rhythm_result[:, i] = idx
# ----------------------------------------rhythmの生成完了------------------------------------------------------------
# ----------------------------------------Pitchの生成開始------------------------------------------------------------
        rhythm_dict = self.rhythm_forward(rhythm_result, chord_hidden, attention_map, masking=True)
        rhythm_out = self.rhythm_outlayer(rhythm_dict['output'])
        rhythm_out = self.log_softmax(rhythm_out)
        idx = torch.argmax(rhythm_out[:, -1, :], dim=1)
        rhythm_temp = torch.cat([rhythm_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        rhythm_enc_dict = self.rhythm_forward(rhythm_temp, chord_hidden, attention_map, masking=False)
        rhythm_emb = rhythm_enc_dict['output']

        pad_length = self.max_len - prime_pitch.size(1)
        pitch_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(prime_pitch.device)
        pitch_pad *= (self.num_pitch - 1)
        pitch_result = torch.cat([prime_pitch, pitch_pad], dim=1)
        for i in range(prime_pitch.size(1), self.max_len):
            pitch_emb = self.pitch_emb(pitch_result)
            emb = torch.cat([pitch_emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
            emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
            pitch_dict= self.pitch_forward(emb, attention_map)
            pitch_dict['output'] = self.pitch_outlayer(pitch_dict['output'])
            pitch_dict['output'] = self.log_softmax(pitch_dict['output'])
            if topk is None:
                idx = torch.argmax(pitch_dict['output'][:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(pitch_dict['output'][:, i - 1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            pitch_result[:, i] = idx
# ----------------------------------------pitchの生成完了------------------------------------------------------------
# ----------------------------------------rhythm1の生成開始------------------------------------------------------------
        emb=self.pitch_emb(pitch_result)
        emb = torch.cat([emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        pitch_dict = self.pitch_forward(emb, attention_map, masking=True)
        output = self.pitch_outlayer(pitch_dict['output'])
        output = self.log_softmax(output)
        idx = torch.argmax(output[:, -1, :], dim=1)
        pitch_temp = torch.cat([pitch_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        pitch_temp = self.pitch_emb(pitch_temp)
        emb = torch.cat([pitch_temp, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        pitch_enc_dict = self.pitch_forward(emb, attention_map, masking=False)
        pitch_hidd = pitch_enc_dict['output']


        pad_length=self.max_len-prime_rhythm1.size(1)
        rhythm1_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(prime_rhythm1.device)
        rhythm1_result = torch.cat([prime_rhythm1, rhythm1_pad], dim=1)
        for i in range(prime_rhythm1.size(1), self.max_len):
            rhythm1_result1 = self.rhythm1_emb(rhythm1_result)

            rhythm1_result1 = torch.cat([rhythm1_result1, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
            rhythm1_result1/= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
            rhythm1_dec_result = self.rhythm1_forward(rhythm1_result1, attention_map, masking=True)
            rhythm1_out = self.rhythm1_outlayer(rhythm1_dec_result['output'])
            rhythm1_out = self.log_softmax(rhythm1_out)
            if topk is None:
                idx = torch.argmax(rhythm1_out[:, i - 1, :], dim=1)
            else:
                top3_probs, top3_idxs = torch.topk(rhythm1_out[:, i - 1, :], 3, dim=-1)
                idx = torch.gather(top3_idxs, 1, torch.multinomial(F.softmax(top3_probs, dim=-1), 1)).squeeze()
            rhythm1_result[:, i] = idx
# ----------------------------------------rhythm1の生成完了------------------------------------------------------------
# ----------------------------------------Pitch1の生成開始------------------------------------------------------------
        rhythm1_result1 = self.rhythm1_emb(rhythm1_result)
        rhythm1_result1 = torch.cat([rhythm1_result1, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
        rhythm1_result1 /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
        rhythm1_dict = self.rhythm1_forward(rhythm1_result1, attention_map, masking=True)
        rhythm1_out = self.rhythm1_outlayer(rhythm1_dict['output'])
        rhythm1_out = self.log_softmax(rhythm1_out)
        idx = torch.argmax(rhythm1_out[:, -1, :], dim=1)
        rhythm1_temp = torch.cat([rhythm1_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        rhythm1_temp = self.rhythm1_emb(rhythm1_temp)
        rhythm1_temp = torch.cat([rhythm1_temp, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
        rhythm1_temp /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
        rhythm1_enc_dict = self.rhythm1_forward(rhythm1_temp, attention_map, masking=False)
        rhythm1_emb = rhythm1_enc_dict['output']



        pad_length = self.max_len - prime_pitch1.size(1)
        pitch1_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(prime_pitch.device)
        pitch1_pad *= (self.num_pitch - 1)
        pitch1_result = torch.cat([prime_pitch1, pitch1_pad], dim=1)
        for i in range(prime_pitch1.size(1), self.max_len):
            pitch1_emb = self.pitch1_emb(pitch1_result)
            emb = torch.cat([pitch1_emb,chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd,rhythm1_emb], -1)
            emb /= torch.sqrt(torch.tensor(self.hidden1_dim, dtype=torch.float))
            pitch1_dict = self.pitch1_forward(emb, attention_map)
#--------------------------------------多音生成代码start----------------------------------------
            # pitch1_dict['output'] = self.pitch1_outlayer(pitch1_dict['output']) #[5,128,48][batchsize,seq_len,pitch_num]
            # pitch1_dict['output'] = self.pitch1_outlayer1(pitch1_dict['output'])
            # pitch1_dict['output'] = self.sigmoid(pitch1_dict['output']) #[5,128,48]
            # # if topk is None:
            # #     idx = torch.argmax(pitch1_dict['output'][:, i - 1, :], dim=1)
            # # else:
            # #     topk_probs, topk_idxs = torch.topk(pitch1_dict['output'][:, i - 1, :], topk, dim=-1)
            # #     idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            # pitch1_result[:, i,:] = pitch1_dict['output'][:,i,:]
#----------------------------------------多音生成代码end--------------------------------------
#----------------------------------------单音生成代码start-----------------------------------
            pitch1_dict['output'] = self.pitch1_outlayer(pitch1_dict['output'])
            pitch1_dict['output'] = self.log_softmax(pitch1_dict['output'])
            if topk is None:
                idx = torch.argmax(pitch1_dict['output'][:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(pitch1_dict['output'][:, i - 1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            pitch1_result[:, i] =idx
# ----------------------------------------单音生成代码end-----------------------------------

        result = {'rhythm': rhythm_result,
                  'pitch': pitch_result,
                  'rhythm1':rhythm1_result,
                  'pitch1':pitch1_result
                  }
        if attention_map:
            result['weights_rdec'] = rhythm_dict['weights']
            result['weights_renc'] = rhythm_enc_dict['weights']
            result['weights_pitch'] = pitch_dict['weights']
        return result

    def getrhythm_pitch(self, prime_rhythm, prime_pitch, prime_rhythm1, prime_pitch1, chord, topk=None, attention_map=False):
        chord_hidden = self.chord_forward(chord)
        # ----------------------------------------rhythmの生成開始------------------------------------------------------------
        # batch_size * prime_len * num_outputs
        batch_size = prime_pitch.size(0)
        pad_length = self.max_len - prime_rhythm.size(1)
        rhythm_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(prime_rhythm.device)
        rhythm_result = torch.cat([prime_rhythm, rhythm_pad], dim=1)

        # sampling phase
        for i in range(prime_rhythm.size(1), self.max_len):
            rhythm_dec_result = self.rhythm_forward(rhythm_result, chord_hidden, attention_map, masking=True)
            rhythm_out = self.rhythm_outlayer(rhythm_dec_result['output'])
            rhythm_out = self.log_softmax(rhythm_out)
            if topk is None:
                idx = torch.argmax(rhythm_out[:, i - 1, :], dim=1)
            else:
                top3_probs, top3_idxs = torch.topk(rhythm_out[:, i - 1, :], 3, dim=-1)
                idx = torch.gather(top3_idxs, 1, torch.multinomial(F.softmax(top3_probs, dim=-1), 1)).squeeze()
            rhythm_result[:, i] = idx
        # ----------------------------------------rhythmの生成完了------------------------------------------------------------
        # ----------------------------------------Pitchの生成開始------------------------------------------------------------
        rhythm_dict = self.rhythm_forward(rhythm_result, chord_hidden, attention_map, masking=True)
        rhythm_out = self.rhythm_outlayer(rhythm_dict['output'])
        rhythm_out = self.log_softmax(rhythm_out)
        idx = torch.argmax(rhythm_out[:, -1, :], dim=1)
        rhythm_temp = torch.cat([rhythm_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        rhythm_enc_dict = self.rhythm_forward(rhythm_temp, chord_hidden, attention_map, masking=False)
        rhythm_emb = rhythm_enc_dict['output']

        pad_length = self.max_len - prime_pitch.size(1)
        pitch_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(prime_pitch.device)
        pitch_pad *= (self.num_pitch - 1)
        pitch_result = torch.cat([prime_pitch, pitch_pad], dim=1)
        for i in range(prime_pitch.size(1), self.max_len):
            pitch_emb = self.pitch_emb(pitch_result)
            emb = torch.cat([pitch_emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
            emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
            pitch_dict = self.pitch_forward(emb, attention_map)
            pitch_dict['output'] = self.pitch_outlayer(pitch_dict['output'])
            pitch_dict['output'] = self.log_softmax(pitch_dict['output'])
            if topk is None:
                idx = torch.argmax(pitch_dict['output'][:, i - 1, :], dim=1)
            else:
                topk_probs, topk_idxs = torch.topk(pitch_dict['output'][:, i - 1, :], topk, dim=-1)
                idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
            pitch_result[:, i] = idx
        # ----------------------------------------pitchの生成完了------------------------------------------------------------
        # ----------------------------------------rhythm1の生成開始------------------------------------------------------------
        emb = self.pitch_emb(pitch_result)
        emb = torch.cat([emb, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        pitch_dict = self.pitch_forward(emb, attention_map, masking=True)
        output = self.pitch_outlayer(pitch_dict['output'])
        output = self.log_softmax(output)
        idx = torch.argmax(output[:, -1, :], dim=1)
        pitch_temp = torch.cat([pitch_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        pitch_temp = self.pitch_emb(pitch_temp)
        emb = torch.cat([pitch_temp, chord_hidden[0], chord_hidden[1], rhythm_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float))
        pitch_enc_dict = self.pitch_forward(emb, attention_map, masking=False)
        pitch_hidd = pitch_enc_dict['output']

        pad_length = self.max_len - prime_rhythm1.size(1)
        rhythm1_pad = torch.zeros([batch_size, pad_length], dtype=torch.long).to(prime_rhythm1.device)
        rhythm1_result = torch.cat([prime_rhythm1, rhythm1_pad], dim=1)
        for i in range(prime_rhythm1.size(1), self.max_len):
            rhythm1_result1 = self.rhythm1_emb(rhythm1_result)

            rhythm1_result1 = torch.cat([rhythm1_result1, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
            rhythm1_result1 /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
            rhythm1_dec_result = self.rhythm1_forward(rhythm1_result1, attention_map, masking=True)
            rhythm1_out = self.rhythm1_outlayer(rhythm1_dec_result['output'])
            rhythm1_out = self.log_softmax(rhythm1_out)
            if topk is None:
                idx = torch.argmax(rhythm1_out[:, i - 1, :], dim=1)
            else:
                top3_probs, top3_idxs = torch.topk(rhythm1_out[:, i - 1, :], 3, dim=-1)
                idx = torch.gather(top3_idxs, 1, torch.multinomial(F.softmax(top3_probs, dim=-1), 1)).squeeze()
            rhythm1_result[:, i] = idx
        # ----------------------------------------rhythm1の生成完了------------------------------------------------------------
        # ----------------------------------------Pitch1の生成開始------------------------------------------------------------
        rhythm1_result1 = self.rhythm1_emb(rhythm1_result)
        rhythm1_result1 = torch.cat([rhythm1_result1, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
        rhythm1_result1 /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
        rhythm1_dict = self.rhythm1_forward(rhythm1_result1, attention_map, masking=True)
        rhythm1_out = self.rhythm1_outlayer(rhythm1_dict['output'])
        rhythm1_out = self.log_softmax(rhythm1_out)
        idx = torch.argmax(rhythm1_out[:, -1, :], dim=1)
        rhythm1_temp = torch.cat([rhythm1_result[:, 1:], idx.unsqueeze(-1)], dim=1)
        rhythm1_temp = self.rhythm1_emb(rhythm1_temp)
        rhythm1_temp = torch.cat([rhythm1_temp, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd], -1)
        rhythm1_temp /= torch.sqrt(torch.tensor(self.rhythm1_hidden, dtype=torch.float))
        rhythm1_enc_dict = self.rhythm1_forward(rhythm1_temp, attention_map, masking=False)
        rhythm1_emb = rhythm1_enc_dict['output']

        pad_length = self.max_len - prime_pitch1.size(1)
        pitch1_pad = torch.ones([batch_size, pad_length], dtype=torch.long).to(prime_pitch.device)
        pitch1_pad *= (self.num_pitch - 1)
        pitch1_result = torch.cat([prime_pitch1, pitch1_pad], dim=1)
        pitch1_emb = self.pitch1_emb(pitch1_result)
        emb = torch.cat([pitch1_emb, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd, rhythm1_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden1_dim, dtype=torch.float))
        i=prime_pitch1.size(1)
        return i,emb,pitch_result,pitch1_result,chord_hidden, rhythm_emb, pitch_hidd, rhythm1_emb
        # for i in range(prime_pitch1.size(1), self.max_len):
        #     pitch1_emb = self.pitch1_emb(pitch1_result)
        #     emb = torch.cat([pitch1_emb, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd, rhythm1_emb], -1)
        #     emb /= torch.sqrt(torch.tensor(self.hidden1_dim, dtype=torch.float))
        #     pitch1_dict = self.pitch1_forward(emb, attention_map)
        #     pitch1_dict['output'] = self.pitch1_outlayer(pitch1_dict['output'])
        #     pitch1_dict['output'] = self.log_softmax(pitch1_dict['output'])
        #     if topk is None:
        #         idx = torch.argmax(pitch1_dict['output'][:, i - 1, :], dim=1)
        #     else:
        #         topk_probs, topk_idxs = torch.topk(pitch1_dict['output'][:, i - 1, :], topk, dim=-1)
        #         idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
        #     pitch1_result[:, i] = idx
        #
        # result = {'rhythm': rhythm_result,
        #           'pitch': pitch_result,
        #           'rhythm1': rhythm1_result,
        #           'pitch1': pitch1_result
        #           }
        # if attention_map:
        #     result['weights_rdec'] = rhythm_dict['weights']
        #     result['weights_renc'] = rhythm_enc_dict['weights']
        #     result['weights_pitch'] = pitch_dict['weights']
        #return result

    
    def choose_action(self,state,i):
        #输入状态，输出action i代表生成到第几个（）
        topk=5
        attention_map=False
        pitch1_dict = self.pitch1_forward(state, attention_map)
        pitch1_dict['output'] = self.pitch1_outlayer(pitch1_dict['output'])
        pitch1_dict['output'] = self.log_softmax(pitch1_dict['output'])
        if topk is None:
            idx = torch.argmax(pitch1_dict['output'][:, i - 1, :], dim=1)
        else:
            topk_probs, topk_idxs = torch.topk(pitch1_dict['output'][:, i - 1, :], topk, dim=-1)
            idx = torch.gather(topk_idxs, 1, torch.multinomial(F.softmax(topk_probs, dim=-1), 1)).squeeze()
        return pitch1_dict,idx

    def step(self, idx,i,pitch1_result,chord_hidden, rhythm_emb, pitch_hidd, rhythm1_emb,topk=None, attention_map=False):
        pitch1_result[:, i] = idx
        pitch1_emb = self.pitch1_emb(pitch1_result)
        emb = torch.cat([pitch1_emb, chord_hidden[0], chord_hidden[1], rhythm_emb, pitch_hidd, rhythm1_emb], -1)
        emb /= torch.sqrt(torch.tensor(self.hidden1_dim, dtype=torch.float))

        return emb
    def ac_learning(self,logits):
        pass