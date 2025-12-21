# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
 
import torch
import numpy as np

import soundfile as sf
from typing import Dict

from torch.utils.data import Dataset

def get_dtype(value: int) -> str:
    """Return the most suitable type for storing the
    value passed in parameter in memory.

    Parameters
    ----------
    value: int
        value whose type is best suited to storage in memory

    Returns
    -------
    str:
        numpy formatted type
        (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    """
    # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
    types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
    filtered_list = [
        (max_val, type) for max_val, type in types_list if max_val > abs(value)
    ]
    if not filtered_list:
        return "i8"  # signed long (64 bits)
    return filtered_list[0][1]

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def load_uem(uem_file: str) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split() for line in open(uem_file)]
    return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}
    
def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> None:
    init_posi = int(init_posi + 1)
    data_len = int(data_len - 1)
    cur_len = data_len - init_posi
    assert cur_len > size
    num_chunks = int((cur_len - size + step) / step)
    
    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size

def _collate_fn(batch, max_speakers_per_chunk=4) -> torch.Tensor:
    """
    Collate variable-length waveforms and frame labels.
    Pads both time-domain samples and frame-level labels, and returns a mask
    indicating valid frames for loss/metric masking.
    """
    xs, ys, names = [], [], []
    wav_lengths = []
    frame_lengths = []

    for x, y, name in batch:
        num_speakers = y.shape[-1]
        if num_speakers > max_speakers_per_chunk:
            # sort speakers in descending talkativeness order
            indices = np.argsort(-np.sum(y, axis=0), axis=0)
            # keep only the most talkative speakers
            y = y[:, indices[: max_speakers_per_chunk]]
        elif num_speakers < max_speakers_per_chunk:
            # create inactive speakers by zero padding
            y = np.pad(
                y,
                ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
                mode="constant",
            )
        # collect lengths before time padding
        wav_lengths.append(x.shape[-1])
        frame_lengths.append(y.shape[0])

        xs.append(x)
        ys.append(y)
        names.append(name)

    max_wav_len = max(wav_lengths)
    max_frame_len = max(frame_lengths)

    padded_xs = []
    padded_ys = []
    masks = []

    for x, y, flen in zip(xs, ys, frame_lengths):
        # pad waveform to max_wav_len
        if x.shape[-1] < max_wav_len:
            pad_width = ((0, 0), (0, max_wav_len - x.shape[-1]))
            x = np.pad(x, pad_width, mode="constant")
        # pad frame labels to max_frame_len
        if y.shape[0] < max_frame_len:
            y = np.pad(
                y,
                ((0, max_frame_len - y.shape[0]), (0, 0)),
                mode="constant",
            )
        padded_xs.append(x)
        padded_ys.append(y)

        mask = np.zeros((max_frame_len,), dtype=np.float32)
        mask[:flen] = 1.0
        masks.append(mask)

    return {
        'xs': torch.from_numpy(np.stack(padded_xs)).float(),
        'ts': torch.from_numpy(np.stack(padded_ys)),
        'mask': torch.from_numpy(np.stack(masks)),
        'names': names
    }        
        
        
class DiarizationDataset(Dataset):
    def __init__(
        self, 
        scp_file: str, 
        rttm_file: str,
        uem_file: str,
        model_num_frames: int,    # default: wavlm_base
        model_rf_duration: float,  # model.receptive_field.duration, seconds
        model_rf_step: float,  # model.receptive_field.step, seconds
        chunk_size: int = 5,  # seconds
        chunk_shift: int = 5, # seconds
        sample_rate: int = 16000,
        full_utterance: bool = False,
        max_sessions: int = None,   # limit number of recordings for quick debug
        max_chunks: int = None,     # limit number of generated chunks
    ): 
        self.chunk_indices = []
        
        self.sample_rate = sample_rate
        self.full_utterance = full_utterance
        
        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames
        
        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        # map session -> idx for O(1) lookup
        self.session_to_idx = {k: i for i, k in enumerate(self.rec_scp.keys())}

        # optional quick subset for debugging; 0 or None means no limit
        if (max_sessions is not None) and (max_sessions > 0):
            rec_items = list(self.reco2dur.items())[:max_sessions]
            self.reco2dur = {k: v for k, v in rec_items}
            self.rec_scp = {k: self.rec_scp[k] for k, _ in rec_items if k in self.rec_scp}
            self.session_to_idx = {k: i for i, k in enumerate(self.rec_scp.keys())}
        
        # parse RTTM once, store per-session annotations to avoid repeated scanning
        self.annotations_by_session = self.rttm2label(rttm_file)

        for rec, dur_info in self.reco2dur.items():
            if rec not in self.annotations_by_session:
                continue
            start_sec, end_sec = dur_info   
            try:
                # When full_utterance is True, use the whole recording as one chunk
                if (not self.full_utterance) and chunk_size > 0:
                    for st, ed in _gen_chunk_indices(
                            start_sec,
                            end_sec,
                            chunk_size,
                            chunk_shift
                    ):
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))      # seconds
                else:
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))
            except:
                print(f'Un-matched recording: {rec}')

            if (max_chunks is not None) and (max_chunks > 0) and len(self.chunk_indices) >= max_chunks:
                break

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        return self.session_to_idx[session]
            
    def rttm2label(self, rttm_file):
        '''
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        '''
        annotations_by_session = {}
        label_lists = {}
        with open(rttm_file, 'r') as file:
            for seg_idx, line in enumerate(file):   
                parts = line.split()
                session, start, dur = parts[1], parts[3], parts[4]

                if session not in self.session_to_idx:
                    # skip segments not in selected subset
                    continue

                start = float(start)
                end = start + float(dur)
                spk = parts[-2] if parts[-2] != "<NA>" else parts[-3]
                
                if session not in label_lists:
                    label_lists[session] = []
                if spk not in label_lists[session]:
                    label_lists[session].append(spk)
                label_idx = label_lists[session].index(spk)
                
                annotations_by_session.setdefault(session, []).append(
                    (
                        self.get_session_idx(session),
                        start,
                        end,
                        label_idx
                    )
                )
                
        segment_dtype = [
            (
                "session_idx",
                get_dtype(len(self.session_to_idx)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("label_idx", get_dtype(max((len(v) for v in label_lists.values()), default=1))),
        ]
        
        for session, segs in annotations_by_session.items():
            annotations_by_session[session] = np.array(segs, dtype=segment_dtype)
        
        return annotations_by_session

    def extract_wavforms(self, path, start, end, num_channels=8):
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        data, sample_rate = sf.read(path, start=start, stop=end)
        assert sample_rate == self.sample_rate
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum('tc->ct', data) 
        return data[:num_channels, :]

    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        session, path, chunk_start, chunk_end = self.chunk_indices[idx]
        data = self.extract_wavforms(path, chunk_start, chunk_end)          # [start, end)
        
        # chunked annotations
        annotations_session = self.annotations_by_session.get(session, None)
        if annotations_session is None or len(annotations_session) == 0:
            chunked_annotations = np.zeros((0,), dtype=[("start","f"),("end","f"),("label_idx","i")])
            labels = []
            print(f'No annotations found for session: {session}')
        else:
            chunked_annotations = annotations_session[
                (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
            ]
            labels = list(np.unique(chunked_annotations['label_idx']))
        
        # discretize chunk annotations at model output resolution
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration
        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)
        
        # get list and number of labels for current scope
        num_labels = max(len(labels), 1)
        # determine dynamic frame length (pad-safe)
        max_end = int(np.max(end_idx)) if len(end_idx) > 0 else -1
        chunk_len = chunk_end - chunk_start
        est_frames = int(np.ceil(max(chunk_len / step, 1)))
        num_frames = max(est_frames, max_end + 1, 1)
        
        mask_label = np.zeros((num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        for start, end, label in zip(
            start_idx, end_idx, chunked_annotations['label_idx']
        ):
            mapped_label = mapping[label]
            end_clipped = min(end, num_frames - 1)
            mask_label[start : end_clipped + 1, mapped_label] = 1
        
        return data, mask_label, session