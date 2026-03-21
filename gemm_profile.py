import subprocess
import csv
import io

# 指标定义保持不变
METRIC_MAP = {
    "gpu__time_duration.sum": "Time (ns)",                       # 耗时
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM SOL (%)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM SOL (%)",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed": "FMA Pipeline (%)",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "Stall Shared (%)",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "Stall Global (%)",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "Bank Conflicts"
}
METRICS = list(METRIC_MAP.keys())
SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
VERSIONS = {0: "Naive", 1: "Tiled", 2: "Float4", 3: "PingPong"}

def get_cublas_tflops(m, n, k):
    """
    运行一段简单的独立脚本测试官方 cuBLAS 端到端的吞吐量作为对比 Baseline
    """
    script = f'''import torch
# 强制使用与自定义算子相同的精度 (FP32) 进行公平对比，关闭 TF32 以免虚高
torch.backends.cuda.matmul.allow_tf32 = False
A = torch.randn({m}, {k}, device="cuda", dtype=torch.float32)
B = torch.randn({k}, {n}, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(10): 
    torch.matmul(A, B)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(20): 
    torch.matmul(A, B)
end.record()
torch.cuda.synchronize()

time_ms = start.elapsed_time(end) / 20.0
time_ns = time_ms * 1e6

# FLOPs = 2 * M * N * K
tflops = (2.0 * {m} * {n} * {k}) / (time_ns * 1e3)
print(f"{{tflops:.2f}}")
'''
    res = subprocess.run(["python", "-c", script], capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except:
        return 0.0

def run_ncu_and_parse(m, n, k, v_id):
    print(f"Profiling {VERSIONS[v_id]} @ {m}x{n}x{k} (with 10 Warmups)...")
    
    cmd = [
        "ncu", "--csv", 
        "--launch-skip", "10", 
        "--launch-count", "1",
        "--metrics", ",".join(METRICS),
        "--kernel-name", "regex:gemm_.*_kernel", 
        "python", "task_launcher.py", str(m), str(n), str(k), str(v_id)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"NCU Failed with code: {result.returncode}")
        print(f"[STDOUT]: {result.stdout}")
        print(f"[STDERR]: {result.stderr}")
        return None

    f = io.StringIO(result.stdout)
    reader = csv.reader(f)
    stats = {}
    
    for row in reader:
        if len(row) < 2 or row[0] == "ID": continue
        try:
            m_name = next((m for m in METRICS if m in row), None)
            if m_name:
                val = row[-1].replace('%', '').replace(',', '')
                stats[METRIC_MAP[m_name]] = f"{float(val):.2f}"
        except:
            continue
            
    # 计算当前版本的 TFLOPS
    if "Time (ns)" in stats:
        time_ns = float(stats["Time (ns)"])
        tflops = (2.0 * m * n * k) / (time_ns * 1e3)
        stats["TFLOPS"] = f"{tflops:.2f}"
        
    return stats

def generate_markdown(results):
    # 添加新的 TFLOPS 与 cuBLAS 对比列
    headers = ["Shape", "Version", "TFLOPS", "cuBLAS TFLOPS", "Perf vs cuBLAS"] + list(METRIC_MAP.values())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in results:
        row = [
            r['shape'], 
            r['version'], 
            r.get('TFLOPS', 'N/A'),
            r.get('cuBLAS_TFLOPS', 'N/A'),
            r.get('Perf_Ratio', 'N/A')
        ] + [str(r.get(h, "N/A")) for h in METRIC_MAP.values()]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

if __name__ == "__main__":
    all_results = []
    for m, n, k in SHAPES:
        print(f"\n--- Benchmarking cuBLAS for {m}x{n}x{k} ---")
        cublas_tflops = get_cublas_tflops(m, n, k)
        print(f"cuBLAS Throughput: {cublas_tflops} TFLOPS\n")
        
        for v_id in VERSIONS.keys():
            data = run_ncu_and_parse(m, n, k, v_id)
            if data:
                data['version'] = VERSIONS[v_id]
                data['shape'] = f"{m}x{n}x{k}"
                data['cuBLAS_TFLOPS'] = f"{cublas_tflops:.2f}"
                
                # 计算与 cuBLAS 的比率
                if "TFLOPS" in data and cublas_tflops > 0:
                    ratio = (float(data["TFLOPS"]) / cublas_tflops) * 100
                    data["Perf_Ratio"] = f"{ratio:.2f}%"
                else:
                    data["Perf_Ratio"] = "N/A"
                    
                all_results.append(data)

    report = generate_markdown(all_results)
    print("\n### GEMM Final Profiling Report (Warmed Up)\n")
    print(report)
    with open("gemm_final_report.md", "w") as f:
        f.write("# GEMM Profiling Report\n\n" + report)