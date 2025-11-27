import os
import sys
import glob
import json
from pyannote.audio import Pipeline


DEFAULT_DIR = \
    "/root/workspace/speech_enhancement/sc/merged_datasets_20251029/SC_CausalMelBandRNN_EDA_16k/inference/from_unlabeled_dir_0005_selected50"

DEFAULT_RTTM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "rttm_out_pyannote"))
DEFAULT_JSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "diar_summary_pyannote.json"))


def list_target_wavs(root_dir: str):
    pattern = os.path.join(root_dir, "**", "*speech_estimate.wav")
    return sorted(glob.glob(pattern, recursive=True))


def make_session_name(wav_path: str, root_dir: str) -> str:
    rel = os.path.relpath(wav_path, root_dir)
    base_no_ext = os.path.splitext(rel)[0]
    return base_no_ext.replace(os.sep, "__")


def count_unique_speakers(diarization_result) -> int:
    speakers = set()
    for _, _, speaker in diarization_result.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    rttm_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_RTTM_DIR
    json_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_JSON_PATH

    os.makedirs(rttm_dir, exist_ok=True)

    wav_paths = list_target_wavs(root_dir)
    if not wav_paths:
        print(f"未找到匹配文件: {os.path.join(root_dir, '**', '*speech_estimate.wav')}")
        return

    print(f"共找到 {len(wav_paths)} 个待处理音频文件（pyannote）。")

    # 若环境配置需要 token，可将其放在环境变量 HF_TOKEN 中
    hf_token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

    summary = []

    for wav_path in wav_paths:
        try:
            session_name = make_session_name(wav_path, root_dir)
            diar = pipeline(wav_path)
            diar.uri = session_name
            num_speakers = count_unique_speakers(diar)

            rttm_path = os.path.join(rttm_dir, f"{session_name}.rttm")
            with open(rttm_path, "w", encoding="utf-8") as f:
                # 大多数版本 Annotation 支持 to_rttm()
                f.write(diar.to_rttm())

            summary.append({
                "wav_path": os.path.abspath(wav_path),
                "session_name": session_name,
                "num_speakers": num_speakers,
                "rttm_path": rttm_path,
            })
            print(f"{wav_path}\t说话人数量={num_speakers}\tRTTM={rttm_path}")
        except Exception as e:
            print(f"处理失败: {wav_path}\t错误: {e}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "root_dir": os.path.abspath(root_dir),
            "rttm_out_dir": os.path.abspath(rttm_dir),
            "items": summary,
        }, f, ensure_ascii=False, indent=2)

    print(f"已写入JSON汇总: {json_path}")


if __name__ == "__main__":
    main()


