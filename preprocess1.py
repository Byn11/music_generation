import glob
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from tqdm import tqdm
from scipy.sparse import csc_matrix


def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, pitch_range=48, shift=False,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):
    if shift:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_12keys' % (num_bars, frame_per_bar, pitch_range)
    else:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_ckey' % (num_bars, frame_per_bar, pitch_range)

    dir_name = os.path.join(root_dir, 'pkl_files_p', instance_folder)
    dir_name=dir_name.replace('\\', '/')
    os.makedirs(dir_name, exist_ok=True)

    instance_len = frame_per_bar * num_bars
    stride = int(instance_len / 2)
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    song_list = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*.mid')))
    song_list=[song.replace('\\', '/') for song in song_list]
    song_list=[song.split('/')[-1] for song in song_list]
    midi_files = sorted(glob.glob(os.path.join(root_dir, midi_dir, '*.mid')))
    midi_files=[midi.replace('\\', '/') for midi in midi_files]
    num_eval = int(len(song_list) * data_ratio[1])
    num_test = int(len(song_list) * data_ratio[2])
    random.seed(0)
    eval_test_cand = set([song.split('/')[-1] for song in song_list])
    eval_set = random.sample(eval_test_cand, num_eval)
    test_set = random.sample(eval_test_cand - set(eval_set), num_test)

    for midi_file in tqdm(midi_files, desc="Processing"):
        song_title = midi_file.split('/')[-1]
        filename = midi_file.split('/')[-1].split('.')[0]

        if song_title in eval_set:
            mode = 'eval'
        elif song_title in test_set:
            mode = 'test'
        else:
            mode = 'train'
        temp=os.path.join(dir_name, mode, song_title).replace('\\', '/')
        os.makedirs(temp, exist_ok=True)
        key_count = len(sorted(glob.glob(os.path.join(dir_name, mode, song_title, '*_+0_*.pkl')))) # in case of modulation

        if shift:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
        for k in pitch_shift:
            midi = pm.PrettyMIDI(midi_file)
            if len(midi.instruments) < 3:
                continue
            print(len(midi.instruments))
            on_midi = pm.PrettyMIDI(midi_file)
            off_midi = pm.PrettyMIDI(midi_file)
            note_instrument = midi.instruments[0]
            onset_instrument = on_midi.instruments[0]
            offset_instrument = off_midi.instruments[0]
            note_instrument1 = midi.instruments[1]
            onset_instrument1= on_midi.instruments[1]
            offset_instrument1 = off_midi.instruments[1]
            for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
                if k != 0:
                    note.pitch += k
                    onset_note.pitch += k
                    offset_note.pitch += k
                note_length = offset_note.end - offset_note.start
                onset_note.end = onset_note.start + min(note_length, unit_time)
                offset_note.end += unit_time
                offset_note.start = offset_note.end - min(note_length, unit_time)
            for note1, onset_note1, offset_note1 in zip(note_instrument1.notes, onset_instrument1.notes, offset_instrument1.notes):
                if k != 0:
                    note1.pitch += k
                    onset_note1.pitch += k
                    offset_note1.pitch += k
                note_length1 = offset_note1.end - offset_note1.start
                onset_note1.end = onset_note1.start + min(note_length1, unit_time)
                offset_note1.end += unit_time
                offset_note1.start = offset_note1.end - min(note_length1, unit_time)

            pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
            onset_roll = onset_instrument.get_piano_roll(fs=frame_per_second)
            offset_roll = offset_instrument.get_piano_roll(fs=frame_per_second)

            pianoroll1 = note_instrument1.get_piano_roll(fs=frame_per_second)
            onset_roll1 = onset_instrument1.get_piano_roll(fs=frame_per_second)
            offset_roll1 = offset_instrument1.get_piano_roll(fs=frame_per_second)

            chord_instrument = midi.instruments[2]
            timelen = min(pianoroll.shape[1], offset_roll.shape[1])
            for chord_note in chord_instrument.notes:
                if k != 0:
                    chord_note.pitch += k
                chord_note.end = chord_note.start + unit_time
            chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)

            pianoroll = pad_pianorolls(pianoroll, timelen)
            onset_roll = pad_pianorolls(onset_roll, timelen)
            offset_roll = pad_pianorolls(offset_roll, timelen)
            chord_onset = pad_pianorolls(chord_onset, timelen)

            pianoroll1 = pad_pianorolls(pianoroll1, timelen)
            onset_roll1 = pad_pianorolls(onset_roll1, timelen)
            offset_roll1 = pad_pianorolls(offset_roll1, timelen)

            pianoroll[pianoroll > 0] = 1
            onset_roll[onset_roll > 0] = 1
            offset_roll[offset_roll > 0] = 1
            chord_onset[chord_onset > 0] = 1

            pianoroll1[pianoroll1 > 0] = 1
            onset_roll1[onset_roll1 > 0] = 1
            offset_roll1[offset_roll1 > 0] = 1
            for i in range(0, timelen - (instance_len + 1), stride):
                pitch_list = []
                chord_list = []

                pitch_list1 = []

                pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
                onset_inst = onset_roll[:, i:(i+instance_len+1)]
                chord_inst = chord_onset[:, i:(i + instance_len + 1)]

                pianoroll_inst1 = pianoroll1[:, i:(i+instance_len+1)]
                onset_inst1 = onset_roll1[:, i:(i+instance_len+1)]
                if len(chord_inst.nonzero()[1]) < 3:
                     continue

                rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
                rhythm_idx = rhythm_idx.astype(int)

                rhythm_idx1 = np.minimum(np.sum(pianoroll_inst1.T, axis=1), 1) + np.minimum(np.sum(onset_inst1.T, axis=1), 1)
                rhythm_idx1 = rhythm_idx1.astype(int)

                # If more than 75% is not-playing, do not make instance
                if rhythm_idx.nonzero()[0].size < (instance_len // 4):
                    continue
                if rhythm_idx1.nonzero()[0].size < (instance_len // 4):
                    continue

                if pitch_range == 128:
                    base_note = 0
                else:
                    highest_note = max(onset_inst.T.nonzero()[1])
                    lowest_note = min(onset_inst.T.nonzero()[1])
                    base_note = 12 * (lowest_note // 12)
                    if highest_note - base_note >= pitch_range:
                        continue

                    highest_note1 = max(onset_inst1.T.nonzero()[1])
                    lowest_note1 = min(onset_inst1.T.nonzero()[1])
                    base_note1 = 12 * (lowest_note1 // 12)
                    if highest_note1 - base_note1 >= pitch_range:
                        continue

                prev_chord = np.zeros(12)
                cont_rest = 0
                prev_onset = 0
                for t in range(instance_len+1):

                    if t in onset_inst.T.nonzero()[0]:
                        pitch_list.append(onset_inst[:, t].T.nonzero()[0][-1] - base_note)#[0][0]为第一个值 取最高音应该是[0][-1]
                        if (t != onset_inst.T.nonzero()[0][-1]) and abs(onset_inst[:, t].T.nonzero()[0][-1] - base_note - prev_onset) > 12 and prev_onset!=0:
                            cont_rest = 30
                            break
                        else:
                            prev_onset = onset_inst[:, t].T.nonzero()[0][-1] - base_note
                            cont_rest = 0
                    elif rhythm_idx[t] == 1:
                        pitch_list.append(pitch_range)
                    elif rhythm_idx[t] == 0:
                        pitch_list.append(pitch_range + 1)
                        cont_rest += 1
                        if cont_rest >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx[t], onset_inst.T.nonzero())

                    if len(chord_inst[:, t].nonzero()[0]) != 0:
                        prev_chord = np.zeros(12)
                        for note in sorted(chord_inst[:, t].nonzero()[0][:] % 12):
                            prev_chord[note] = 1
                    chord_list.append(prev_chord)

                cont_rest1 = 0
                prev_onset1 = 0
                for t in range(instance_len + 1):
                    if t in onset_inst1.T.nonzero()[0]:
                        o = onset_inst1.T.nonzero()[0]

                        pitch_list1.append(onset_inst1[:, t].T.nonzero()[0][0] - base_note1)
                        if (t != onset_inst1.T.nonzero()[0][0]) and abs(
                                onset_inst1[:, t].T.nonzero()[0][0] - base_note1 - prev_onset1) > 24 and prev_onset1!=0:
                            cont_rest1 = 30
                            break
                        else:
                            prev_onset1 = onset_inst1[:, t].T.nonzero()[0][0] - base_note1
                            cont_rest1 = 0
                    elif rhythm_idx1[t] == 1:
                        pitch_list1.append(pitch_range)
                    elif rhythm_idx1[t] == 0:
                        pitch_list1.append(pitch_range + 1)
                        cont_rest1 += 1
                        if cont_rest1 >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx1[t], onset_inst1.T.nonzero())

                #pianoroll_inst1=pianoroll_inst1[base_note1:base_note1+48,:]


                if (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                    continue

                if (cont_rest1 >= 30) or (len(set(pitch_list1)) <= 5):
                    continue

                pitch_list = np.array(pitch_list)
                chord_result = csc_matrix(np.array(chord_list))

                pitch_list1 = np.array(pitch_list1)
                result = {'pitch': pitch_list,
                          'rhythm': rhythm_idx,
                          'chord': chord_result,
                          'pitch1': pitch_list1,#多音版本为pianoroll_inst1
                          'rhythm1': rhythm_idx1,
                          'chord_prime':np.array(chord_list)
                          }
                ps = ('%d' % k) if (k < 0) else ('+%d' % k)
                pkl_filename = os.path.join(dir_name, mode, song_title, '%s_%02d_%s_%02d.pkl' % (song_title, key_count, ps, i // stride))
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(result, f)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', type=str, default='C:/git_workspace/CMT-pytorch/dataset')
    # parser.add_argument('--midi_dir', type=str, default='Traditional')
    # parser.add_argument('--num_bars', type=int, default=8)
    # parser.add_argument('--frame_per_bar', type=int, default=16)
    # parser.add_argument('--pitch_range', type=int, default=48)
    # parser.add_argument('--shift', dest='shift', action='store_true')
    #
    # args = parser.parse_args()
    # root_dir = args.root_dir
    # midi_dir = args.midi_dir
    # num_bars = args.num_bars
    # frame_per_bar = args.frame_per_bar
    # pitch_range = args.pitch_range
    # shift = args.shift

    #make_instance_pkl_files(root_dir, midi_dir, num_bars, frame_per_bar, pitch_range, shift)
    make_instance_pkl_files('C:\\git_workspace\\CMT-pytorch\\dataset', 'ALTO', 8, 16, 48)