import subprocess
import csv
import io

# 指标定义保持不变
METRIC_MAP = {
        "gpu__time_duration.sum": "Time (ns)",                       # 耗时

    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM SOL (%)",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM SOL (%)",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active": "ALU Pipeline (%)",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "FMA Pipeline (%)",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "Stall Shared (%)",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "Stall Global (%)",
    "l1tex__data_bank_conflicts_pipe_lsu.sum": "Bank Conflicts"
}
METRICS = list(METRIC_MAP.keys())
SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
VERSIONS = {0: "Naive", 1: "Tiled", 2: "Float4", 3: "PingPong"}

def run_ncu_and_parse(m, n, k, v_id):
    print(f"Profiling {VERSIONS[v_id]} @ {m}x{n}x{k} (with 10 Warmups)...")
    
    # --- 核心 NCU 参数解释 ---
    # --launch-skip 10: 跳过前 10 次匹配的 Kernel 启动（即跳过预热）
    # --launch-count 1: 只采集之后的第 1 次启动
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

    # 解析逻辑 (适配 NCU 可能输出的多行 CSV)
    f = io.StringIO(result.stdout)
    reader = csv.reader(f)
    stats = {}
    
    for row in reader:
        if len(row) < 2 or row[0] == "ID": continue
        try:
            # 匹配 Metric Name 并提取 Value
            m_name = next((m for m in METRICS if m in row), None)
            if m_name:
                val = row[-1].replace('%', '').replace(',', '')
                stats[METRIC_MAP[m_name]] = f"{float(val):.2f}"
        except:
            continue
    return stats

def generate_markdown(results):
    headers = ["Shape", "Version"] + list(METRIC_MAP.values())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in results:
        row = [r['shape'], r['version']] + [r.get(h, "N/A") for h in METRIC_MAP.values()]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

if __name__ == "__main__":
    all_results = []
    for m, n, k in SHAPES:
        for v_id in VERSIONS.keys():
            data = run_ncu_and_parse(m, n, k, v_id)
            if data:
                data.update({'version': VERSIONS[v_id], 'shape': f"{m}x{n}x{k}"})
                all_results.append(data)

    report = generate_markdown(all_results)
    print("\n### GEMM Final Profiling Report (Warmed Up)\n")
    print(report)
    with open("gemm_final_report.md", "w") as f:
        f.write("# GEMM Profiling Report\n\n" + report)