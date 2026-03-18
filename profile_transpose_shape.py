import subprocess
import csv
import io

def run_shape_profile(m, n):
    kernel_name = "transpose_padding_kernel" # 固定测试 Padding 版本
    
    # 针对 4090 的核心指标清单
    metrics = [
        "gpu__time_duration.sum",                                      # 耗时
        "dram__bytes.sum.per_second",                                 # 物理带宽
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",        # DRAM SOL
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",       # L1/TEX SOL
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"    # L2 命中率 (重要!)
    ]
    
    cmd = [
         "ncu", "--csv", 
        "--metrics", ",".join(metrics),
        "--kernel-name", kernel_name,
        "--launch-skip", "5", 
        "--launch-count", "1",
        "python", "test_transpose.py", "--mode", "shared_p1", "--m", str(m), "--n", str(n)
    ]

    print(f"📐 正在测试 Shape: {m} x {n} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        if "out of memory" in result.stderr.lower():
            return "OOM"
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
            stats["BW_GBs"] = val / 1e9 if "Gbyte" not in m_unit else val
        else:
            stats[m_name] = m_val
                
    return stats

def main():
    # 针对 4090 显存(24GB)设计的实验矩阵
    # 确保总占用 (M*N*4*2) 不超过 20GB
    test_shapes = [
        (8192, 8192),      # 基准正方形 (512MB)
        (8191, 8191),      # 非对齐正方形
        (16384, 16384),    # 大正方形 (2GB)
        (32, 524288),      # 极瘦长 (Skinny - 128MB)
        (524288, 32),      # 极宽平 (Wide - 128MB)
        (32768, 32768),    # 极限大正方形 (8GB)
    ]
    
    results = []
    for m, n in test_shapes:
        data = run_shape_profile(m, n)
        if data == "OOM":
            results.append({"Shape": f"{m}x{n}", "Note": "CUDA Out of Memory"})
        elif data:
            results.append({"Shape": f"{m}x{n}", **data})

    print("\n### 4090 Padding Kernel Shape 敏感度报告\n")
    header = "| Shape (MxN) | 耗时 (ns) | 带宽 (GB/s) | DRAM SOL | L1 SOL | L2 Hit % |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header)
    print(sep)
    
    for r in results:
        shape = r["Shape"]
        if "Note" in r:
            print(f"| {shape} | - | - | - | - | **{r['Note']}** |")
            continue
            
        dur = r.get("gpu__time_duration.sum", "N/A")
        bw = f"{r.get('BW_GBs', 0):.2f}"
        dram = f"{r.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', '0')}%"
        l1 = f"{r.get('l1tex__throughput.avg.pct_of_peak_sustained_elapsed', '0')}%"
        l2 = f"{r.get('l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate.pct', '0')}%"
        
        print(f"| {shape} | {dur} | **{bw}** | {dram} | {l1} | {l2} |")

if __name__ == "__main__":
    main()