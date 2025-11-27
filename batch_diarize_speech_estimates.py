import os
import sys
import glob
import json
import time
import torch
import torchaudio
from diarizen.pipelines.inference import DiariZenPipeline
from pathlib import Path

# CPU 优化设置（全局标志，确保只设置一次）
_cpu_optimizations_setup = False

def setup_cpu_optimizations(force=False):
    """设置 CPU 优化参数（单线程模式）
    
    Args:
        force: 是否强制重新设置（不推荐，可能导致错误）
    """
    global _cpu_optimizations_setup
    
    # 如果已经设置过且不强制，则跳过
    if _cpu_optimizations_setup and not force:
        return
    
    # 使用单线程模式
    num_threads = 1
    
    # 只在第一次调用时设置 interop_threads（避免 "cannot set after parallel work started" 错误）
    if not _cpu_optimizations_setup:
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError as e:
            # 如果已经启动并行工作，忽略这个错误
            print(f"警告: 无法设置 interop_threads（可能已启动并行工作）: {e}")
    
    # 设置线程数（这个可以多次调用）
    torch.set_num_threads(num_threads)
    
    # 启用 MKL 优化
    if hasattr(torch.backends, 'mkl'):
        torch.backends.mkl.enabled = True
    
    # 设置环境变量优化（只在第一次设置）
    if not _cpu_optimizations_setup:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        os.environ['KMP_BLOCKTIME'] = '0'  # 减少线程阻塞时间
        os.environ['KMP_AFFINITY'] = 'granularity=fine,verbose,compact,1,0'  # 线程亲和性优化
    
    # 启用 CPU 的 JIT 优化
    torch.set_flush_denormal(True)
    
    _cpu_optimizations_setup = True
    
    if not force:
        print(f"CPU 优化设置（单线程模式）: 线程数={num_threads}, MKL={torch.backends.mkl.is_available() if hasattr(torch.backends, 'mkl') else 'N/A'}")


DEFAULT_DIR = \
    "/root/workspace/speech_enhancement/sc/merged_datasets_20251029/SC_CausalMelBandRNN_EDA_16k/inference/from_unlabeled_dir_0005_selected50"

DEFAULT_RTTM_DIR = None  # 若未指定，则使用 <root_dir>/DiariZen_rttm_out
DEFAULT_JSON_PATH = None  # 若未指定，则使用 <root_dir>/DiariZen_diar_summary.json

# 模型配置：本地路径或 Hugging Face repo_id
MODEL_CONFIGS = {
    "base": "models--BUT-FIT--diarizen-wavlm-base-s80-md",
    "large": "models--BUT-FIT--diarizen-wavlm-large-s80-md",
}


def list_target_wavs(root_dir: str):
    # patterns = ["*speech_estimate.wav", "*mixture.wav"]
    patterns = ["*mixture.wav"]
    # patterns = ["*speech_estimate.wav"]
    files = set()
    for pat in patterns:
        pattern = os.path.join(root_dir, "**", pat)
        files.update(glob.glob(pattern, recursive=True))
    return sorted(files)


def count_unique_speakers(diarization_result) -> int:
    speakers = set()
    for _, _, speaker in diarization_result.itertracks(yield_label=True):
        speakers.add(speaker)
    return len(speakers)


def make_session_name(wav_path: str, root_dir: str) -> str:
    rel = os.path.relpath(wav_path, root_dir)
    base_no_ext = os.path.splitext(rel)[0]
    # 将相对路径转换为合法且基本唯一的session名
    return base_no_ext.replace(os.sep, "__")


def find_model_path(model_name: str, cache_dir: str = "./cache") -> str:
    """查找模型路径：优先使用本地路径，否则使用 Hugging Face repo_id"""
    # 尝试作为本地路径查找（可能在 cache_dir 中）
    local_paths = [
        Path(model_name),
        Path(cache_dir) / model_name,
        Path.home() / ".cache" / "huggingface" / "hub" / model_name,
    ]
    
    for path in local_paths:
        if path.exists() and path.is_dir():
            config_file = path / "config.toml"
            if config_file.exists():
                print(f"找到本地模型: {path}")
                return str(path)
    
    # 如果本地找不到，尝试转换为 Hugging Face repo_id
    # models--BUT-FIT--diarizen-wavlm-base-s80-md -> BUT-FIT/diarizen-wavlm-base-s80-md
    if model_name.startswith("models--"):
        repo_id = model_name.replace("models--", "").replace("--", "/")
        print(f"使用 Hugging Face repo_id: {repo_id}")
        return repo_id
    
    return model_name


def run_inference_for_config(
    root_dir: str,
    model_name: str,
    device_name: str,
    cache_dir: str = "./cache",
    segmentation_only: bool = False,
):
    """为特定模型和设备配置运行推理"""
    # 保存原始 CUDA_VISIBLE_DEVICES
    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    # 确定设备
    if device_name.lower() == "cpu":
        device = torch.device("cpu")
        # 在 CPU 模式下，设置环境变量强制使用 CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # 应用 CPU 优化设置（只设置线程数，不设置 interop_threads）
        # 注意：setup_cpu_optimizations 会检查是否已设置，避免重复设置 interop_threads
        setup_cpu_optimizations()
    elif device_name.lower() == "gpu":
        if not torch.cuda.is_available():
            print(f"警告: GPU 不可用，跳过 {model_name} + GPU 配置")
            return None
        device = torch.device("cuda:0")
        # 恢复 CUDA 可见性
        if original_cuda_visible is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    else:
        raise ValueError(f"未知设备: {device_name}")
    
    # 查找模型路径
    model_path = find_model_path(model_name, cache_dir)
    
    # 创建输出目录
    model_short = model_name.split("--")[-1] if "--" in model_name else model_name.split("/")[-1]
    output_suffix = f"{model_short}_{device_name}"
    if segmentation_only:
        output_suffix += "_segonly"
    rttm_dir = os.path.join(root_dir, f"DiariZen_rttm_out_{output_suffix}")
    json_path = os.path.join(root_dir, f"DiariZen_diar_summary_{output_suffix}.json")
    
    os.makedirs(rttm_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"配置: 模型={model_name}, 设备={device_name}")
    print(f"RTTM输出目录: {rttm_dir}")
    print(f"JSON汇总文件: {json_path}")
    print(f"{'='*80}\n")
    
    # 加载 pipeline
    diar_pipeline = DiariZenPipeline.from_pretrained(
        model_path,
        rttm_out_dir=rttm_dir,
        cache_dir=cache_dir,
        device=device,
        segmentation_only=segmentation_only,
    )
    
    # 获取待处理文件列表
    wav_paths = list_target_wavs(root_dir)
    if not wav_paths:
        print(f"未找到匹配文件: {os.path.join(root_dir, '**', '*mixture.wav')}")
        return None
    
    print(f"共找到 {len(wav_paths)} 个待处理音频文件。")
    
    summary = []
    
    for wav_path in wav_paths:
        session_name = make_session_name(wav_path, root_dir)
        # 计算音频时长
        info = torchaudio.info(wav_path)
        duration = float(info.num_frames) / float(info.sample_rate) if info.sample_rate > 0 else 0.0
        
        # CPU 优化：在推理模式下运行，禁用梯度计算
        t0 = time.time()
        with torch.inference_mode():
            diar_results = diar_pipeline(wav_path, sess_name=session_name)
        t1 = time.time()
        elapsed = t1 - t0
        rtf = (elapsed / duration) if duration > 0 else None
        num_speakers = count_unique_speakers(diar_results)
        
        rttm_path = os.path.join(rttm_dir, f"DiariZen_{session_name}.rttm")
        summary.append({
            "wav_path": os.path.abspath(wav_path),
            "session_name": session_name,
            "num_speakers": num_speakers,
            "rttm_path": rttm_path,
            "duration_sec": duration,
            "elapsed_sec": elapsed,
            "rtf": rtf,
        })
        rtf_str = f"{rtf:.3f}" if rtf is not None else "NA"
        print(f"{wav_path}\t说话人数量={num_speakers}\tRTF={rtf_str}\tRTTM={rttm_path}")
    
    # RTF 统计
    rtf_values = [it["rtf"] for it in summary if isinstance(it.get("rtf"), (int, float))]
    rtf_stats = None
    if rtf_values:
        rtf_stats = {
            "count": len(rtf_values),
            "mean": sum(rtf_values) / len(rtf_values),
            "min": min(rtf_values),
            "max": max(rtf_values),
        }
    
    result = {
        "model_name": model_name,
        "device": device_name,
        "root_dir": os.path.abspath(root_dir),
        "rttm_out_dir": os.path.abspath(rttm_dir),
        "rtf_stats": rtf_stats,
        "items": summary,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n已写入JSON汇总: {json_path}")
    if rtf_stats:
        print(f"RTF统计: 平均={rtf_stats['mean']:.3f}, 最小={rtf_stats['min']:.3f}, 最大={rtf_stats['max']:.3f}")
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="批量测试 DiariZen 模型的 RTF 性能")
    parser.add_argument("root_dir", type=str, nargs="?", default=DEFAULT_DIR,
                        help="输入音频文件目录（默认: %(default)s）")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="模型缓存目录（默认: ./cache）")
    parser.add_argument("--segmentation-only", action="store_true",
                        help="只使用 segmentation 模型，跳过 embedding 和 clustering（加快速度但可能降低精度）")
    parser.add_argument("--devices", nargs="+", choices=["CPU", "GPU", "cpu", "gpu"], 
                        help="要测试的设备列表（默认: 根据 CUDA 可用性自动选择）")
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    cache_dir = args.cache_dir
    segmentation_only = args.segmentation_only
    
    # 设置 Hugging Face 镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    
    # 初始化 CPU 优化（如果后续使用 CPU 会再次调用，但提前设置也无妨）
    setup_cpu_optimizations()
    
    # 检查 GPU 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")
    else:
        print(f"CPU 核心数: {os.cpu_count()}")
        print(f"PyTorch 线程数: {torch.get_num_threads()}")
    
    # 定义所有测试配置
    if args.devices:
        devices = [d.upper() for d in args.devices]
    else:
        # 默认：根据 CUDA 可用性自动选择
        devices = (["GPU", "CPU"] if cuda_available else ["CPU"])
    
    all_results = {}
    
    # 打印配置信息
    print(f"\n{'='*80}")
    print("DiariZen RTF 测试配置")
    print(f"{'='*80}")
    print(f"输入目录: {root_dir}")
    print(f"输出目录: {os.path.join(root_dir, 'rtf_test_results')}")
    print(f"缓存目录: {cache_dir}")
    print(f"Segmentation-only 模式: {segmentation_only}")
    print(f"测试设备: {', '.join(devices)}")
    print(f"测试模型: {', '.join(MODEL_CONFIGS.keys())}")
    print(f"{'='*80}\n")
    
    # 遍历所有模型和设备组合
    for model_key, model_name in MODEL_CONFIGS.items():
        for device_name in devices:
            config_key = f"{model_key}_{device_name}"
            if segmentation_only:
                config_key += "_segonly"
            print(f"\n\n开始测试配置: {config_key}")
            result = run_inference_for_config(
                root_dir=root_dir,
                model_name=model_name,
                device_name=device_name,
                cache_dir=cache_dir,
                segmentation_only=segmentation_only,
            )
            if result:
                all_results[config_key] = result
    
    # 生成汇总报告
    summary_path = os.path.join(root_dir, "DiariZen_all_configs_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "root_dir": os.path.abspath(root_dir),
            "configs": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*80}")
    print("所有配置测试完成！")
    print(f"汇总报告: {summary_path}")
    print(f"{'='*80}\n")
    
    # 打印 RTF 对比表
    print("RTF 对比表:")
    print(f"{'配置':<30} {'平均RTF':<12} {'最小RTF':<12} {'最大RTF':<12} {'样本数':<10}")
    print("-" * 80)
    for config_key, result in all_results.items():
        if result.get("rtf_stats"):
            stats = result["rtf_stats"]
            print(f"{config_key:<30} {stats['mean']:<12.3f} {stats['min']:<12.3f} {stats['max']:<12.3f} {stats['count']:<10}")


if __name__ == "__main__":
    main()


