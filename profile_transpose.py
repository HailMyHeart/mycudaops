import subprocess
import csv
import io

def run_profile(mode, kernel_prefix, m=8192, n=8192):
    # 终极指标清单
    metrics = [
        "gpu__time_duration.sum",                                      # 耗时
        "dram__bytes.sum.per_second",                                 # 物理带宽 GB/s
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",        # DRAM SOL
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",       # L1/TEX SOL
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",   # 读冲突
        "smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum" # 写冲突
    ]
    
    cmd = [
         "ncu", "--csv", 
        "--metrics", ",".join(metrics),
        "--kernel-name", kernel_prefix,
        "--launch-skip", "10", 
        "--launch-count", "1",
        "python", "test_transpose.py", "--mode", mode, "--m", str(m), "--n", str(n)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    f = io.StringIO(result.stdout)
    lines = f.readlines()
    csv_body = ""
    start_capture = False
    for line in lines:
        if '"ID"' in line or "ID," in line:
            start_capture = True
        if start_capture:
            csv_body += line

    if not csv_body:
        return None

    reader = csv.DictReader(io.StringIO(csv_body))
    stats = {}
    for row in reader:
        m_name = (row.get("Metric Name") or row.get('"Metric Name"') or "").strip('"')
        m_val = (row.get("Metric Value") or row.get('"Metric Value"') or "0").strip('"').replace(',', '')
        m_unit = (row.get("Metric Unit") or row.get('"Metric Unit"') or "").strip('"')
        
        if "bytes.sum.per_second" in m_name:
            val = float(m_val)
            # 统一换算为 GB/s
            stats["BW_GBs"] = val / 1e9 if "Gbyte" not in m_unit else val
        else:
            stats[m_name] = m_val
                
    return stats

def main():
    tasks = [
        ("naive", "transpose_naive_kernel"),
        ("shared_p0", "transpose_shared_kernel"),
        ("shared_p1", "transpose_padding_kernel"),
        ("swizzle", "transpose_swizzle_kernel")
    ]
    
    results = []
    for mode, k_prefix in tasks:
        print(f"📊 正在全量采集 [{mode}] 的硬件性能数据...")
        data = run_profile(mode, k_prefix)
        if data:
            results.append({"Mode": mode, **data})

    # 打印最终的全能版报告
    print("\n### 4090 Transpose 终极性能报告 (8192x8192)\n")
    header = "| 模式 | 耗时 (ns) | 带宽 (GB/s) | DRAM SOL | L1 SOL | LDS 冲突 | STS 冲突 |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    
    for r in results:
        dur = r.get("gpu__time_duration.sum", "N/A")
        bw = f"{r.get('BW_GBs', 0):.2f}"
        dram_sol = f"{r.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', '0')}%"
        l1_sol = f"{r.get('l1tex__throughput.avg.pct_of_peak_sustained_elapsed', '0')}%"
        ld_conflict = r.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "0")
        st_conflict = r.get("smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "0")
        
        print(f"| {r['Mode']} | {dur} | **{bw}** | {dram_sol} | {l1_sol} | {ld_conflict} | {st_conflict} |")

if __name__ == "__main__":
    main()