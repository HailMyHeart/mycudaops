import torch
import my_lib
import subprocess
import csv
import io
import sys

# ----------------- 理论分析参数 -----------------
# 我们测试三个 Shape: M(Batch*Seq), D(Feature_Dim)
SHAPES = [(1024, 1024), (4096, 4096), (8192, 4096)]
EPS = 1e-5

# 要抓取的 NCU metrics
METRICS_MAP = {
    "gpu__time_duration.sum": "Time (ns)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM SOL (%)",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM SOL (%)"
}

def create_task_launcher():
    # 创建一个轻量级的独立的 launcher 进行 profiling
    with open("ln_task_launcher.py", "w") as f:
        f.write("""import torch
import my_lib
import sys

M = int(sys.argv[1])
D = int(sys.argv[2])

input_tensor = torch.randn(M, D, device='cuda', dtype=torch.float32)
gamma = torch.randn(D, device='cuda', dtype=torch.float32)
beta = torch.randn(D, device='cuda', dtype=torch.float32)

for _ in range(5):
    my_lib.layernorm(input_tensor, gamma, beta, 1e-5)

torch.cuda.synchronize()
my_lib.layernorm(input_tensor, gamma, beta, 1e-5)
torch.cuda.synchronize()
""")

def run_ncu_and_parse(m, d):
    create_task_launcher()
    cmd = [
        "ncu", "--csv",
        "--launch-skip", "5",
        "--launch-count", "1",
        "--metrics", ",".join(METRICS_MAP.keys()),
        "--kernel-name", "layernorm_kernel",
        "python", "ln_task_launcher.py", str(m), str(d)
    ]
    
    print(f"Profiling LayerNorm @ M={m}, D={d} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Failed to profile M={m}, D={d}")
        return {}

    stats = {}
    reader = csv.reader(io.StringIO(res.stdout))
    for row in reader:
        if len(row) < 2 or row[0] == "ID":
            continue
        try:
            m_name = next((mk for mk in METRICS_MAP.keys() if mk in row), None)
            if m_name:
                val = row[-1].replace('%', '').replace(',', '')
                stats[METRICS_MAP[m_name]] = float(val)
        except Exception as e:
            continue
    return stats

def main():
    results = []
    
    for M, D in SHAPES:
        # 手动计算理论值
        # 访存量：读入两次(求方差均值，求结果各自一次)，写出一次。忽略相对很少的 gamma/beta
        bytes_accessed = 3.0 * M * D * 4.0 
        
        # 计算量：均值方差3 Ops，计算归一化 4 Ops
        flops = 7.0 * M * D

        stats = run_ncu_and_parse(M, D)
        if not stats: continue
            
        time_ns = stats.get("Time (ns)", 0)
        
        # 计算吞吐量
        # GB/s = Bytes / ns
        bw_gbps = (bytes_accessed / time_ns) if time_ns > 0 else 0
        
        # GFLOPS = FLOPs / ns
        tflops = (flops / time_ns) / 1000.0 if time_ns > 0 else 0
        
        row_data = {
            "Shape": f"{M}x{D}",
            "Theo BW (GB/s)": f"{bw_gbps:.2f}",
            "Theo TFLOPS": f"{tflops:.4f}",
            "Time (ns)": f"{time_ns:.0f}",
            "DRAM SOL (%)": f"{stats.get('DRAM SOL (%)', 0):.2f}",
            "SM SOL (%)": f"{stats.get('SM SOL (%)', 0):.2f}"
        }
        results.append(row_data)
        
    # 生成 Markdown
    headers = list(results[0].keys())
    lines = []
    lines.append("# LayerNorm Profiling Report\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for r in results:
        lines.append("| " + " | ".join([r[h] for h in headers]) + " |")
        
    report = "\n".join(lines)
    print("\n" + report)
    
    with open("layernorm_report.md", "w") as f:
        f.write(report + "\n")

if __name__ == "__main__":
    main()