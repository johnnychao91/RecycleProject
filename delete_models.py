import os
from pathlib import Path

# 找出 PyTorch 預設快取路徑
cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints'

if cache_dir.exists():
    checkpoint_files = list(cache_dir.glob('*.pth'))
    if checkpoint_files:
        print(f"🧠 已下載的模型快取（共 {len(checkpoint_files)} 個）:")
        for i, f in enumerate(checkpoint_files, 1):
            print(f"{i}. {f.name} - {round(f.stat().st_size / 1024 / 1024, 2)} MB")
    else:
        print("📭 沒有模型快取檔案。")
else:
    print("⚠️ 尚未建立快取資料夾，尚未下載任何模型。")

# 問你是否要刪除
if checkpoint_files:
    delete = input("\n是否要刪除這些快取檔案？(y/N): ").strip().lower()
    if delete == 'y':
        for f in checkpoint_files:
            f.unlink()
        print("✅ 所有模型快取已刪除。")
    else:
        print("🚫 未刪除任何檔案。")
