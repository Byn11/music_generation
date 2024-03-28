import os
import datetime
import numpy as np
import pretty_midi
import matplotlib
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import logger
from tensorboardX import SummaryWriter


def get_optimizer(params, lr, config, name='adam'):
    name = name.lower()
    if name == 'sgd':
        optimizer = optim.sgd(params, lr=lr, **config[name])
    elif name == 'adam':
        optimizer = optim.Adam(params, lr=lr, **config[name])
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, **config[name])
    else:
        raise RuntimeError("%s is not available." % name)

    return optimizer


def make_save_dir(save_path, config):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)
    config.save(os.path.join(save_path, "hparams.yaml"))


def get_tfwriter(asset_path):
    now = datetime.datetime.now()
    folder = "run-%s" % now.strftime("%m%d-%H%M%S")
    writer = SummaryWriter(logdir=os.path.join(asset_path, 'tensorboard', folder))

    return writer


def print_result(losses, metrics):
    for name, val in losses.items():
        logger.info("%s: %.4f" % (name, val))
    for name, val in metrics.items():
        logger.info("%s: %.4f" % (name, val))


def tensorboard_logging_result(tf_writer, epoch, results):
    for tag, value in results.items():
        if 'img' in tag:
            tf_writer.add_image(tag, value, epoch)
        elif 'hist' in tag:
            tf_writer.add_histogram(tag, value, epoch)
        else:
            tf_writer.add_scalar(tag, value, epoch)


def pitch_to_midi(pitch, pitch1,chord, frame_per_bar=16, save_path=None, basis_note=60):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name='melody')
    instrument1 = pretty_midi.Instrument(program=1, name='melody1')
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second

    on_pitch = {}
    for t, idx in enumerate(pitch):
        if idx in range(48):
            if bool(on_pitch):
                note = pretty_midi.Note(start=on_pitch['time'],
                                        end=t * unit_time,
                                        pitch=on_pitch['pitch'],
                                        velocity=100)
                instrument.notes.append(note)
                on_pitch = {}
            on_pitch['pitch'] = basis_note + idx
            on_pitch['time'] = t * unit_time
        elif idx == 49 and bool(on_pitch):
            note = pretty_midi.Note(start=on_pitch['time'],
                                    end=t * unit_time,
                                    pitch=on_pitch['pitch'],
                                    velocity=100)
            instrument.notes.append(note)
            on_pitch = {}

    if bool(on_pitch):
        note = pretty_midi.Note(start=on_pitch['time'],
                                end=t * unit_time,
                                pitch=on_pitch['pitch'],
                                velocity=100)
        instrument.notes.append(note)
#------------------------------------------------------------------------------
    on_pitch1 = {}
    for t, idx in enumerate(pitch1):#(128,48)
        if idx in range(48):
            if bool(on_pitch1):
                note = pretty_midi.Note(start=on_pitch1['time'],
                                        end=t * unit_time,
                                        pitch=on_pitch1['pitch'],
                                        velocity=100)
                instrument1.notes.append(note)
                on_pitch1 = {}
            on_pitch1['pitch'] = basis_note + idx
            on_pitch1['time'] = t * unit_time
        elif idx == 49 and bool(on_pitch1):
            note = pretty_midi.Note(start=on_pitch1['time'],
                                    end=t * unit_time,
                                    pitch=on_pitch1['pitch'],
                                    velocity=100)
            instrument1.notes.append(note)
            on_pitch1 = {}

    if bool(on_pitch1):
        note = pretty_midi.Note(start=on_pitch1['time'],
                                end=t * unit_time,
                                pitch=on_pitch1['pitch'],
                                velocity=100)
        instrument1.notes.append(note)

    # for t in range(len(pitch1)):  # 128
    #     for i in range(len(pitch1[t])):  # 48 在t这个时间0音的遍历
    #         if pitch1[t][i] == [0]:
    #             j = t - 1
    #             if j == -1:
    #                 continue
    #             if pitch1[j][i] == [0]:
    #                 continue
    #
    #             while (j != -1 and pitch1[j][i] != [0]):
    #                 j -= 1
    #             note = pretty_midi.Note(start=(j + 1) * unit_time,
    #                                     end=(t + 1) * unit_time,
    #                                     pitch=i+basis_note,
    #                                     velocity=100)
    #             instrument1.notes.append(note)



    midi_data.instruments.append(instrument)
    midi_data.instruments.append(instrument1)
    midi_data.instruments.append(chord_to_instrument(chord, frame_per_bar=frame_per_bar))
    if save_path is not None:
        midi_data.write(save_path)

    return midi_data.instruments


def chord_to_instrument(chord_array, frame_per_bar=16):
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second
    instrument = pretty_midi.Instrument(program=0, name='chord')
    chord = chord_array[0]
    prev_t = 0
    for t in range(chord_array.shape[0]):
        if not (chord_array[t] == chord).all():
            chord_notes = chord.nonzero()[0]
            for pitch in chord_notes:
                note = pretty_midi.Note(start=prev_t * unit_time, end=t * unit_time, pitch=48 + pitch, velocity=70)
                instrument.notes.append(note)
            prev_t = t
            chord = chord_array[t]
    chord_notes = chord.nonzero()[0]
    for pitch in chord_notes:
        note = pretty_midi.Note(start=prev_t * unit_time, end=chord_array.shape[0] * unit_time, pitch=48 + pitch, velocity=70)
        instrument.notes.append(note)
    return instrument


def save_instruments_as_image(filename, instruments, frame_per_bar=16, num_bars=8):
    melody_inst = instruments[0]
    timelen = frame_per_bar * num_bars
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second

    piano_roll = melody_inst.get_piano_roll(fs=frame_per_second)
    if piano_roll.shape[1] < timelen:
        piano_roll = np.pad(piano_roll, ((0, 0), (0, timelen - piano_roll.shape[1])),
                            mode="constant", constant_values=0)
    for note in melody_inst.notes:
        note_len = note.end - note.start
        note.end = note.start + min(note_len, unit_time)
    onset_roll = melody_inst.get_piano_roll(fs=frame_per_second)
    if onset_roll.shape[1] < timelen:
        onset_roll = np.pad(onset_roll, ((0, 0), (0, timelen - onset_roll.shape[1])),
                            mode="constant", constant_values=0)

    if (num_bars) // 16 > 1:
        rows = (num_bars) // 16
        lowest_pitch = 36
        highest_pitch = 96
        C_labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    else:
        rows = 1
        lowest_pitch = 0
        highest_pitch = 128
        C_labels = ['C-1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fig = plt.figure(figsize=(8, 6))
    for row in range(rows):
        ax = fig.add_subplot(rows, 1, row + 1)
        ax.set_ylim([lowest_pitch, highest_pitch])
        ax.set_xticks(np.arange(frame_per_bar, timelen // rows, frame_per_bar))
        ax.set_xticklabels(np.arange(row*(timelen // rows) + 2 * frame_per_bar, (row+1)*(timelen // rows) + frame_per_bar, frame_per_bar))
        ax.set_yticks(np.arange(lowest_pitch, highest_pitch, 12))
        ax.set_yticklabels(C_labels)
        for C_idx in range(12 + lowest_pitch, highest_pitch, 12):
            ax.axhline(y=C_idx, color='b', linewidth=0.5)
        for i in range(num_bars // rows):
            ax.axvline(x=frame_per_bar*(i+1), color='r', linewidth=0.5)
        plot = plt.imshow((piano_roll + onset_roll)[:, row*(timelen // rows):(row+1)*(timelen // rows)] / 2, interpolation=None, cmap='gray_r')
    plt.savefig(filename)
    plt.close(fig)

root_note_list = [' C', 'C#', ' D', 'D#', ' E', ' F', 'F#', ' G', 'G#', ' A', 'A#', ' B']

def idx_list_to_symbol_list(idx_list):
    symbol_list = []
    for i, event_idx in enumerate(idx_list):
        if event_idx == 48:
            symbol = '%03d,  N' % (i + 1)
        elif event_idx == 49:
            symbol =  ''
        else:
            octave = event_idx // 12 + 3
            root_note = event_idx % 12
            symbol = '%03d,%s%d' % (i + 1, root_note_list[root_note], octave)
        symbol_list.append(symbol)
    return symbol_list


def chord_to_symbol_list(chord):
    symbol_list = []
    root_list = []
    for root in chord[0].nonzero()[0].tolist():
        root_list.append(root_note_list[root].replace(' ', ''))
    symbol = ','.join(root_list)
    symbol_list.append('001,'+symbol)
    for i in range(1, chord.shape[0]):
        if (chord[i] - chord[i-1]).tolist() == np.zeros(12).tolist():
            symbol = ''
        else:
            root_list = []
            for root in chord[i].nonzero()[0].tolist():
                root_list.append(root_note_list[root].replace(' ', ''))
            symbol = '-'.join(root_list)
            symbol = '%03d,%s' % (i + 1, symbol)
        symbol_list.append(symbol)
    return symbol_list

def rhythm_to_symbol_list(beat_list):
    symbol_list = []
    for i, beat in enumerate(beat_list):
        if beat == 2:
            symbol = '%03d, 2' % (i + 1)
        elif beat == 1:
            symbol = '1'
        else:
            symbol = ''
        symbol_list.append(symbol)
    return symbol_list

def pitch_to_symbol_list(pitch_list):
    symbol_list = []
    for i, pitch in enumerate(pitch_list):
        if pitch == 88:
            symbol = ''
        else:
            octave = (pitch + 20) // 12 - 2
            root_note = (pitch + 20) % 12
            symbol = '%03d,%s%d' % (i + 1, root_note_list[root_note], octave)
        symbol_list.append(symbol)
    return symbol_list


def chord_dict_to_array(chord_dict, max_len):
    chord = []
    next_t = max_len
    for t in sorted(chord_dict.keys(), reverse=True):
        chord_array = np.zeros(12)
        for note in chord_dict[t] % 12:
            chord_array[note] = 1
        chord_array = np.tile(chord_array, (next_t - t, 1))
        chord.append(chord_array)
        next_t = t
    if next_t != 0:
        chord.append(np.tile(np.zeros(12), (next_t, 1)))
    chord = np.concatenate(chord)[::-1]
    return chord

def chord_array_to_dict(chord_array):
    chord_dict = dict()
    chord = np.zeros(12)
    for t in range(chord_array.shape[0]):
        if not (chord_array[t] == chord).all():
            chord_dict[t] = chord_array[t].nonzero()[0]
            chord = chord_array[t]
    return chord_dict


def get_reward1(pitch_result,rhythm1,pitch1,chord,i):#pitch_result [16,128]
    # for ii in range(16):
    #     print(ii,rhythm1[ii])
    #     print(ii,pitch1[ii])

    r_list=np.zeros([50,16],dtype='float32')
    #reward_l={[3,4,8,9]:0.8,[5,7]:1,[0,1,2,6,10,11]:0}
    reward_l = {3: 0.8, 4: 0.8, 8: 0.8, 9: 0.8, 5: 1, 7: 1, 0: 0, 1: 0, 2: 0, 6: 0, 10: 0, 11: 0}
    reward_ll=np.array([0,0,0,0.8,0.8,1,0,1,0.8,0.8,0,0])

    if i>0:
        jisuan=np.array(pitch_result[:,i-0:i+1].cpu())
    else:
        jisuan=np.array(pitch_result[:,0:i+1].cpu())#(16,8)


    for a in range(48):
        temploss=np.zeros([16])
        #for b in range(len(jisuan[0])):
            # c=jisuan[:,b] #(16,1)
            # d=c-np.array([a for i in range(16)]).T
            # d=abs(d%12)
            # loss1=reward_ll[d]
            # loss1[np.where(c>47)]=0
            # temploss=temploss+loss1
        temploss[np.where(rhythm1[:,i].cpu()==0)]=0
        temploss[np.where(rhythm1[:,i].cpu()==1)]=0
        r_list[a] =temploss #(16,1)


    if i<8:
        upitch = pitch1[:, 0:i].cpu().numpy()
    if i>=8:
        upitch = pitch1[:, i - 8:i].cpu().numpy()
    last, last1 = [], np.zeros([16])
    for m in range(16):
        
        if len(upitch[0]) == 0:
            break
        last.append(list(np.where(upitch[m] < 48)))
        if len(last[m][0]) == 0:
            continue
        if i %4==3:
            if rhythm1[m,i]==2:
                z=np.nonzero(chord[m, i, :])
                z1=np.nonzero(chord[m, i, :])
                z1=torch.cat((z1,z+12))
                z1 = torch.cat((z1, z+24))
                z1 = torch.cat((z1, z +36)).cpu().numpy()
                r_list[z1,m]=1
        aaa = last[m][0]
        aa = max(last[m][0])
        max_v = upitch[m, max(last[m][0])]
        if max_v > 8:
            r_list[0:int(max_v - 8), m] = 0
        if max_v < 42:
            r_list[max_v + 8:50, m] = 0

    r_list[48]=np.zeros([16])
    r_list[49]=np.zeros([16])
    #hh=np.where(rhythm1[:,i]==0)
    #hhh=np.where(rhythm1[:,i]==1)
    #r_list[:][np.where(rhythm1[:, i] == 0)[0]]=0
    r_list[-1][np.where(rhythm1[:,i].cpu()==0)[0]]=1
    r_list[-2][np.where(rhythm1[:, i].cpu() == 1)] = 1
    for n in range(16):
        if np.sum(r_list[:,n])==0 or np.sum(r_list[:,n])==1:
            continue
        gh=np.sum(r_list[:,n])
        r_list[:,n]=r_list[:,n]/gh

    return r_list #[50,16]

def get_reward2(rhythm,pitch,chord, i):
    r_list = np.zeros([50, 16], dtype='float32')
    # for ii in range(128):
    #     print(ii,rhythm[:,ii])
    #     print(ii,chord[:,ii,:])
    for a in range(48):
        temploss=np.zeros([16])

        temploss[np.where(rhythm[:,i].cpu()==0)]=0
        temploss[np.where(rhythm[:,i].cpu()==1)]=0
        r_list[a] =temploss #(16,1)
    if i<8:
        upitch = pitch[:, 0:i].cpu().numpy()
    if i>=8:
        upitch = pitch[:, i - 8:i].cpu().numpy()
    last, last1 = [], np.zeros([16])
    for m in range(16):
        
        if len(upitch[0]) == 0:
            break
        last.append(list(np.where(upitch[m] < 48)))
        if len(last[m][0]) == 0:
            continue
        if i %4==3:
            if rhythm[m,i]==2:
                z=np.nonzero(chord[m, i, :])
                z1=np.nonzero(chord[m, i, :])
                z1=torch.cat((z1,z+12))
                z1 = torch.cat((z1, z+24))
                z1 = torch.cat((z1, z +36)).cpu().numpy()
                r_list[z1,m]=1
        aaa = last[m][0]
        aa = max(last[m][0])
        max_v = upitch[m, max(last[m][0])]
        if max_v > 8:
            r_list[0:int(max_v - 8), m] = 0
        if max_v < 42:
            r_list[max_v + 8:50, m] = 0

    r_list[48]=np.zeros([16])
    r_list[49]=np.zeros([16])
    #hh=np.where(rhythm[:,i]==0)
    #hhh=np.where(rhythm[:,i]==1)
    #r_list[:][np.where(rhythm1[:, i] == 0)[0]]=0
    r_list[-1][np.where(rhythm[:,i].cpu()==0)[0]]=1
    r_list[-2][np.where(rhythm[:, i].cpu() == 1)] = 1
    for n in range(16):
        if np.sum(r_list[:,n])==0 or np.sum(r_list[:,n])==1:
            continue
        gh=np.sum(r_list[:,n])
        r_list[:,n]=r_list[:,n]/gh
    return r_list
