"""
清理项目缓存文件的脚本

使用方法:
    python cleanup.py
"""

import os
import sys
import shutil
from pathlib import Path


def find_and_remove_cache():
    """查找并删除缓存文件"""
    script_dir = Path(__file__).parent
    removed_items = []
    total_size = 0
    
    # 查找 __pycache__ 目录
    for pycache_dir in script_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            size = sum(f.stat().st_size for f in pycache_dir.rglob("*") if f.is_file())
            try:
                shutil.rmtree(pycache_dir)
                removed_items.append(f"📁 {pycache_dir.relative_to(script_dir)}")
                total_size += size
            except Exception as e:
                print(f"❌ 无法删除 {pycache_dir}: {e}")
    
    # 查找 .pyc 文件
    for pyc_file in script_dir.rglob("*.pyc"):
        if pyc_file.is_file():
            size = pyc_file.stat().st_size
            try:
                pyc_file.unlink()
                removed_items.append(f"📄 {pyc_file.relative_to(script_dir)}")
                total_size += size
            except Exception as e:
                print(f"❌ 无法删除 {pyc_file}: {e}")
    
    # 查找 .pyo 文件
    for pyo_file in script_dir.rglob("*.pyo"):
        if pyo_file.is_file():
            size = pyo_file.stat().st_size
            try:
                pyo_file.unlink()
                removed_items.append(f"📄 {pyo_file.relative_to(script_dir)}")
                total_size += size
            except Exception as e:
                print(f"❌ 无法删除 {pyo_file}: {e}")
    
    return removed_items, total_size


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    """主函数"""
    print("=" * 60)
    print("🧹 清理项目缓存文件")
    print("=" * 60)
    print()
    
    print("🔍 正在查找缓存文件...")
    removed_items, total_size = find_and_remove_cache()
    
    if not removed_items:
        print("✅ 没有找到需要清理的缓存文件")
        print("   项目已经很干净了！")
    else:
        print(f"\n✅ 已清理 {len(removed_items)} 个缓存项")
        print(f"📦 释放空间: {format_size(total_size)}")
        print()
        print("已删除的文件/目录:")
        for item in removed_items[:10]:  # 只显示前10个
            print(f"  {item}")
        if len(removed_items) > 10:
            print(f"  ... 还有 {len(removed_items) - 10} 个文件")
    
    print()
    print("=" * 60)
    print("💡 提示: 这些缓存文件会在下次运行时自动重新生成")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 已取消清理")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

