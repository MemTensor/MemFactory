"""
验证脚本：从 eval_results/ 直接读取 JSON 文件，打印与报告一一对应的准确率表格。
用途：与 tmp/eval_results_report.md 中的数据交叉核对。

使用方式：
    python evaluation/verify_results.py
"""

import os
import json
import glob

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")

AGENT_TYPES = ["memagent", "no_memory"]
MODELS_MAP = {
    "memagent": [
        "models_Qwen3-1.7B",
        "models_Qwen3-4B-Instruct",
        "MemAgent-RL-Qwen3-1.7B_checkpoint_250",
        "MemAgent-RL-Qwen3-4B-Instruct_checkpoint_250",
    ],
    "no_memory": [
        "models_Qwen3-1.7B",
        "models_Qwen3-4B-Instruct",
        "NoMemory-GRPO-Qwen3-1.7B_checkpoint_250",
        "NoMemory-GRPO-Qwen3-4B-Instruct_checkpoint_250",
    ],
}
DATASETS = ["eval_50", "eval_100", "eval_200", "eval_400", "eval_fwe_16384", "eval_fwe_32768"]
DS_LABELS = ["eval_50", "eval_100", "eval_200", "eval_400", "fwe_16k", "fwe_32k"]

COL_W = 9   # column width


def load_results():
    results = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        fname = os.path.basename(f).replace(".json", "")
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            s = data.get("summary", {})
            results[fname] = {
                "acc":       s.get("current_accuracy"),
                "processed": s.get("processed", 0),
                "total":     s.get("total", 0),
            }
        except Exception as e:
            results[fname] = {"acc": None, "processed": 0, "total": 0, "error": str(e)}
    return results


def cell(results, key):
    r = results.get(key)
    if r is None:
        return "MISSING"
    if r.get("acc") is None:
        return "ERR"
    if r["processed"] != r["total"] or r["total"] == 0:
        pct = r["processed"] / r["total"] * 100 if r["total"] > 0 else 0
        return f"run{pct:.0f}%"
    return f"{r['acc']:.4f}"


def main():
    results = load_results()

    header = f"{'模型':<45}" + "".join(f"{h:>{COL_W}}" for h in DS_LABELS)
    sep = "-" * len(header)

    n_ok, n_missing, n_incomplete = 0, 0, 0

    for at in AGENT_TYPES:
        print(f"\n{'='*6} {at} {'='*6}")
        print(header)
        print(sep)
        for model in MODELS_MAP[at]:
            row_cells = []
            for ds in DATASETS:
                key = f"{at}_{model}_{ds}"
                c = cell(results, key)
                row_cells.append(c)
                if c == "MISSING":
                    n_missing += 1
                elif c.startswith("run"):
                    n_incomplete += 1
                else:
                    n_ok += 1
            short = model.replace("NoMemory-GRPO-", "GRPO-").replace("MemAgent-RL-", "RL-") \
                        .replace("_checkpoint_250", " ckpt250").replace("models_", "")
            print(f"{short:<45}" + "".join(f"{c:>{COL_W}}" for c in row_cells))

    total = n_ok + n_missing + n_incomplete
    print(f"\n{'='*50}")
    print(f"完成: {n_ok}/{total}  |  进行中: {n_incomplete}  |  缺失: {n_missing}")
    print(f"数据来源: summary.current_accuracy（已验证 processed == total）")


if __name__ == "__main__":
    main()
