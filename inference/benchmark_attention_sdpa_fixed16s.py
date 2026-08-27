#!/usr/bin/env python3
"""Representative 16-second attention-core benchmark for DiariZen."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import time
import traceback
from pathlib import Path


def text(cmd):
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def idle_snapshot():
    gpu = text(["nvidia-smi", "--query-gpu=timestamp,index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate", "--format=csv,noheader,nounits"])
    procs = text(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    fields = [x.strip() for x in gpu.split(",")]
    snap = {"raw_gpu": gpu, "raw_compute_processes": procs, "memory_used_mib": int(fields[5]), "utilization_gpu_percent": int(fields[6])}
    if procs or snap["memory_used_mib"] > 256 or snap["utilization_gpu_percent"] > 10:
        raise RuntimeError(f"GPU not idle: {snap}")
    return snap


def pct(xs, q):
    ys = sorted(xs)
    return ys[int(round((len(ys)-1)*q))]


def stats(xs):
    return {"mean_ms": statistics.mean(xs)*1e3, "p50_ms": pct(xs,.5)*1e3, "p95_ms": pct(xs,.95)*1e3, "repeats": len(xs)}


def bench(torch, fn, warmup=20, repeats=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    xs=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); t=time.perf_counter(); fn(); torch.cuda.synchronize(); xs.append(time.perf_counter()-t)
    out=stats(xs); out["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/2**20
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 2:
        parser.error("--warmup must be >= 1 and --repeats must be >= 2")
    out={"status":"started", "started_at":dt.datetime.now(dt.timezone.utc).astimezone().isoformat(), "input_context":{"audio_seconds":16,"waveform_shape":[1,1,256000],"post_conv_sequence_length":799,"seed":args.seed}, "warmup":args.warmup,"repeats":args.repeats}
    try:
        out["gpu_preflight_after_lock"]=idle_snapshot()
        import torch
        import torch.nn.functional as F
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        out["environment"]={"torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),"capability":list(torch.cuda.get_device_capability(0))}
        cases={}
        for name,h,d in [("wavlm_large",16,64),("conformer",4,64)]:
            b,l=1,799
            q=torch.randn(b,h,l,d,device="cuda",dtype=torch.float16)
            k=torch.randn_like(q); v=torch.randn_like(q)
            # WavLM has a dense per-head gated relative-position additive bias. Conformer use_posi=false has no mask.
            mask=torch.randn(b,h,l,l,device="cuda",dtype=torch.float16)*0.05 if name=="wavlm_large" else None
            scale=d**-0.5
            def manual():
                scores=(q*scale)@k.transpose(-2,-1)
                if mask is not None: scores=scores+mask
                scores=scores-scores.max(dim=-1,keepdim=True).values
                return torch.softmax(scores,dim=-1)@v
            def sdpa_auto(): return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=False)
            ref=manual(); got=sdpa_auto(); torch.cuda.synchronize()
            diff=(ref.float()-got.float()).abs()
            case={"shape":{"q":[b,h,l,d],"dense_additive_mask":mask is not None},"manual":bench(torch,manual,args.warmup,args.repeats),"sdpa_auto":bench(torch,sdpa_auto,args.warmup,args.repeats),"sdpa_vs_manual":{"max_abs":float(diff.max()),"mean_abs":float(diff.mean())}}
            case["sdpa_auto"]["speedup_vs_manual"]=case["manual"]["mean_ms"]/case["sdpa_auto"]["mean_ms"]
            forced={}
            for backend,flags in {"flash_only":(True,False,False),"mem_efficient_only":(False,True,False),"math_only":(False,False,True)}.items():
                try:
                    def forced_fn(flags=flags):
                        with torch.backends.cuda.sdp_kernel(enable_flash=flags[0],enable_mem_efficient=flags[1],enable_math=flags[2]):
                            return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0,is_causal=False)
                    forced[backend]={"status":"ok","timing":bench(torch,forced_fn,args.warmup,args.repeats)}
                except Exception as e:
                    forced[backend]={"status":"failed","error_type":type(e).__name__,"error":str(e)}
            case["forced_backends"]=forced
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                    sdpa_auto(); torch.cuda.synchronize()
                case["sdpa_auto_profiler_attention_ops"]=[e.key for e in p.key_averages() if "attention" in e.key.lower() or "flash" in e.key.lower() or "efficient" in e.key.lower()]
            except Exception as e:
                case["profiler_error"]=str(e)
            cases[name]=case
        out["cases"]=cases
        out["status"]="ok"
    except Exception as e:
        out["status"]="failed"; out["error_type"]=type(e).__name__; out["error"]=str(e); out["traceback"]=traceback.format_exc()
    out["finished_at"]=dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(out,indent=2,ensure_ascii=False))
    return 0 if out["status"]=="ok" else 1


if __name__=="__main__": raise SystemExit(main())
