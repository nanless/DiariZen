# DiariZen EEND Module
This directory contains scripts for DiariZen EEND module training and global inference for speaker diarization. 


## Results (collar=0s)
| System     | Features       | AMI  | AISHELL-4 | AliMeeting |
|:------------|:----------------:|:------:|:------------:|:------------:|
| [Pyannote v3.1](https://github.com/pyannote/pyannote-audio)  | SincNet        | 22.4 | 12.2       | 24.4       |
| DiariZen   | Fbank          | 19.7 | 12.5       | 21.0       |
|            | WavLM-frozen   | 17.0 | 11.7       | 19.9       |
|            | WavLM-updated  | **15.4** | **11.7**       | **17.6**       |

## NEST XLarge Finetuning
`run_finetune_kaldi_merged_nest_xlarge_1219_all.sh` fine-tunes the DiariZen diarization head with NVIDIA NeMo NEST XLarge (`nvidia/ssl_en_nest_xlarge_v1.0`) instead of WavLM.

Prerequisites:
- NeMo repo is available, by default at `/root/code/github_repos/NeMo`.
- The script defaults to the `diarizen-nemo` conda environment, which should contain NeMo dependencies and PyTorch >= 2.6.
- `DATA_SRC` points to a Kaldi directory containing `wav.scp`, `rttm`, and `reco2dur`.

Typical flow:
```bash
cd recipes/diar_ssl
SKIP_TRAIN=1 bash run_finetune_kaldi_merged_nest_xlarge_1219_all.sh
bash run_finetune_kaldi_merged_nest_xlarge_1219_all.sh
tmux attach -t diarizen_nest_xlarge_ft
```

Useful overrides:
```bash
NEMO_ROOT=/root/code/github_repos/NeMo \
DATA_SRC=/path/to/kaldi_merged_1219_all \
NUM_GPUS=4 BATCH_SIZE=4 VAL_BATCH_SIZE=4 \
LR_NEST=3e-6 LR_HEAD=1e-4 \
bash run_finetune_kaldi_merged_nest_xlarge_1219_all.sh
```

One-step smoke test on GPU 1:
```bash
TMUX=1 CONDA_ENV=diarizen-nemo CUDA_VISIBLE_DEVICES=1 NUM_GPUS=1 \
BATCH_SIZE=1 VAL_BATCH_SIZE=1 MAX_STEPS=1 MAX_EPOCHS=1 \
MAX_TRAIN_CHUNKS=2 MAX_DEV_CHUNKS=2 \
bash run_finetune_kaldi_merged_nest_xlarge_1219_all.sh
```

Set `NEST_MODEL=/path/to/model.nemo` to use a local downloaded NEST checkpoint instead of fetching from HuggingFace. Set `RESUME=1` to continue from the latest checkpoint under the generated experiment directory.


## Citation
If you found this work helpful, please consider citing:
J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, and L. Burget, [Leveraging Self-Supervised Learning for Speaker Diarization](https://arxiv.org/pdf/2409.09408), in Proc. ICASSP, 2025.
```
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  booktitle={Proc. ICASSP},
  year={2025}
}

```
