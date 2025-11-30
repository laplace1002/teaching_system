"""
启动 Streamlit UI 的便捷脚本

使用方法:
    python run_ui.py          # ✅ 推荐方式
    或
    streamlit run ui.py       # ✅ 直接运行 UI
"""

import os
import sys
import subprocess

def check_streamlit_installed():
    """检查 streamlit 是否已安装"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def main():
    """主函数"""
    # 检查是否在 streamlit 环境中运行（避免双重启动）
    if "streamlit" in sys.modules:
        print("⚠️  检测到已在 streamlit 环境中运行")
        print("💡 请直接运行: streamlit run ui.py")
        return
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_file = os.path.join(script_dir, "ui.py")
    
    # 检查 streamlit 是否安装
    if not check_streamlit_installed():
        print("=" * 60)
        print("❌ Streamlit 未安装")
        print("=" * 60)
        print("请运行以下命令安装:")
        print("    pip install streamlit")
        print("=" * 60)
        sys.exit(1)
    
    # 检查 ui.py 是否存在
    if not os.path.exists(ui_file):
        print(f"❌ 错误: 找不到 {ui_file}")
        sys.exit(1)
    
    # 打印启动信息
    print("=" * 60)
    print("🚀 启动多智能体教学系统 Web UI (v2.0 性能优化版)")
    print("=" * 60)
    print("📝 浏览器会自动打开 http://localhost:8501")
    print("⚠️  请确保已配置 .env 文件中的 DEEPSEEK_API_KEY")
    print("💡 提示: 如果没有 API key，系统会使用后备模式")
    print("=" * 60)
    print()
    
    # 切换到脚本目录
    os.chdir(script_dir)
    
    # 启动 streamlit
    try:
        # 直接调用 streamlit，不使用 subprocess 的等待
        subprocess.call(["streamlit", "run", ui_file])
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("👋 已关闭 UI")
        print("=" * 60)
    except FileNotFoundError:
        print("\n" + "=" * 60)
        print("❌ 错误: 找不到 streamlit 命令")
        print("=" * 60)
        print("💡 请确保 streamlit 已正确安装:")
        print("   pip install streamlit")
        print("=" * 60)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 提示: 请检查 streamlit 是否正确安装")
        sys.exit(1)

if __name__ == "__main__":
    main()

