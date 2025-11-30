"""
Multi-Agent Teaching System - Streamlit UI
为教学系统提供交互式 Web 界面
"""

import streamlit as st
import json
import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from system import TeachingSystem, DeepSeekConfig
    from agents import DifficultyLevel
except ImportError:
    # Try relative import
    from .system import TeachingSystem, DeepSeekConfig
    from .agents import DifficultyLevel


def init_session_state():
    """初始化 session state"""
    if "teaching_system" not in st.session_state:
        st.session_state.teaching_system = None
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = False


def configure_api_sidebar():
    """侧边栏：API 配置"""
    with st.sidebar:
        st.header("⚙️ API 配置")
        
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="从环境变量读取或在此输入"
        )
        
        api_base = st.text_input(
            "API Base URL",
            value="https://api.deepseek.com",
            help="DeepSeek API 基础 URL"
        )
        
        model = st.selectbox(
            "模型",
            ["deepseek-chat", "deepseek-reasoner"],
            help="选择使用的模型"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="控制输出的随机性"
        )
        
        max_iterations = st.number_input(
            "最大修订次数",
            min_value=1,
            max_value=5,
            value=2,
            help="每个节点的最大修订迭代次数"
        )
        
        if st.button("🔧 初始化系统"):
            config = DeepSeekConfig(
                api_key=api_key if api_key else None,
                api_base=api_base,
                chat_model=model,
                temperature=temperature
            )
            
            with st.spinner("初始化教学系统..."):
                st.session_state.teaching_system = TeachingSystem(
                    config=config,
                    max_revision_iterations=max_iterations
                )
                st.session_state.api_configured = True
            
            st.success("✅ 系统初始化成功！")
        
        st.divider()
        
        st.header("📊 系统状态")
        if st.session_state.api_configured:
            st.success("🟢 系统已配置")
            if st.session_state.teaching_system.llm:
                st.info("🤖 LLM: 已连接")
            else:
                st.warning("⚠️ LLM: 后备模式")
        else:
            st.error("🔴 系统未配置")


def render_learning_path_form():
    """渲染学习路径创建表单"""
    st.header("🎓 个性化学习路径规划委员会")
    st.markdown("""
    这是一个多维度的元认知（Meta-cognitive）教学系统。
    
    **四位专家 Agent 将为你工作：**
    - 🏗️ **架构师 (Curriculum Designer)**: 设计知识图谱和学习大纲
    - 🧠 **心理学家 (Pedagogue)**: 确保难度合适，避免受挫
    - 📚 **图书管理员 (Librarian)**: 查找高质量学习资源
    - 🎓 **模拟学生 (Simulated Student)**: 从学习者角度测试内容
    
    **关键特性：** 反馈闭环 + 压力测试 = 经过验证的学习路径
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input(
            "🎯 学习主题",
            placeholder="例如：量子物理、机器学习、西班牙语...",
            help="输入你想学习的任何主题"
        )
    
    with col2:
        user_level = st.selectbox(
            "📊 当前水平",
            ["beginner", "intermediate", "advanced", "expert"],
            help="你目前对该主题的了解程度"
        )
    
    target_level = st.selectbox(
        "🎯 目标水平",
        ["beginner", "intermediate", "advanced", "expert"],
        index=1,
        help="你希望达到的水平"
    )
    
    if not st.session_state.api_configured:
        st.warning("⚠️ 请先在侧边栏配置 API")
        return
    
    if st.button("🚀 创建学习路径", type="primary", use_container_width=True):
        if not topic:
            st.error("请输入学习主题")
            return
        
        with st.spinner("🔄 多位专家正在为你工作..."):
            try:
                system = st.session_state.teaching_system
                result = system.create_learning_path(
                    topic=topic,
                    user_level=user_level,
                    target_level=target_level
                )
                
                st.session_state.current_result = result
                st.success("✅ 学习路径创建成功！")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")
                import traceback
                with st.expander("查看详细错误"):
                    st.code(traceback.format_exc())


def render_process_log(result: Dict[str, Any]):
    """渲染处理日志"""
    with st.expander("📜 处理日志（查看 Agent 工作流程）", expanded=False):
        messages = result.get("messages", [])
        for msg in messages:
            st.text(msg)


def render_learning_path(result: Dict[str, Any]):
    """渲染学习路径"""
    if not result or not result.get("completed"):
        return
    
    st.header("📚 你的个性化学习路径")
    
    learning_path = result.get("learning_path", {})
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📖 主题", learning_path.get("topic", "N/A"))
    with col2:
        st.metric("⏱️ 总学时", f"{learning_path.get('total_hours', 0):.1f} 小时")
    with col3:
        st.metric("📊 节点数", len(learning_path.get("nodes", [])))
    
    st.divider()
    
    # Nodes
    nodes = learning_path.get("nodes", [])
    resources = result.get("resources", {})
    
    for i, node in enumerate(nodes):
        render_node_card(node, resources, result, i + 1)


def render_node_card(
    node: Dict[str, Any],
    resources: Dict[str, Any],
    result: Dict[str, Any],
    index: int
):
    """渲染单个学习节点卡片"""
    node_id = node.get("id")
    
    # Difficulty color
    difficulty = node.get("difficulty", "beginner")
    difficulty_colors = {
        "beginner": "🟢",
        "intermediate": "🟡",
        "advanced": "🟠",
        "expert": "🔴"
    }
    difficulty_icon = difficulty_colors.get(difficulty, "⚪")
    
    with st.container():
        st.subheader(f"{index}. {node.get('title', 'Untitled')} {difficulty_icon}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**描述：** {node.get('description', 'No description')}")
            
            key_concepts = node.get("key_concepts", [])
            if key_concepts:
                st.markdown(f"**关键概念：** {', '.join(key_concepts)}")
            
            prerequisites = node.get("prerequisites", [])
            if prerequisites:
                st.markdown(f"**前置要求：** {', '.join(prerequisites)}")
        
        with col2:
            st.metric("⏱️ 预计学时", f"{node.get('estimated_hours', 1):.1f}h")
            st.metric("📈 难度", difficulty)
        
        # Resources
        node_resources = resources.get(node_id, [])
        if node_resources:
            with st.expander(f"📚 学习资源 ({len(node_resources)})"):
                for res in node_resources:
                    render_resource(res)
        
        # Feedback
        render_node_feedback(node_id, result)
        
        st.divider()


def render_resource(resource: Dict[str, Any]):
    """渲染单个资源"""
    res_type = resource.get("type", "unknown")
    
    type_icons = {
        "video": "🎥",
        "article": "📄",
        "book": "📖",
        "course": "🎓",
        "exercise": "✍️"
    }
    
    icon = type_icons.get(res_type, "📌")
    title = resource.get("title", "Untitled")
    description = resource.get("description", "")
    url = resource.get("url")
    
    if url:
        st.markdown(f"{icon} **[{title}]({url})**")
    else:
        st.markdown(f"{icon} **{title}**")
    
    if description:
        st.caption(description)
    
    if resource.get("estimated_time"):
        st.caption(f"⏱️ 约 {resource.get('estimated_time'):.1f} 小时")


def render_node_feedback(node_id: str, result: Dict[str, Any]):
    """渲染节点反馈（心理学家 + 模拟学生）"""
    pedagogue_feedback = [
        f for f in result.get("pedagogue_feedback", [])
        if f.get("node_id") == node_id
    ]
    
    student_feedback = [
        f for f in result.get("student_feedback", [])
        if f.get("node_id") == node_id
    ]
    
    if not pedagogue_feedback and not student_feedback:
        return
    
    with st.expander("🔍 专家反馈（Pedagogue + Student）"):
        if pedagogue_feedback:
            st.markdown("### 🧠 心理学家评估")
            for feedback in pedagogue_feedback:
                approved = feedback.get("approved", False)
                if approved:
                    st.success("✅ 已批准")
                else:
                    st.warning("⚠️ 需要改进")
                
                reasoning = feedback.get("reasoning", "")
                if reasoning:
                    st.info(reasoning)
                
                issues = feedback.get("issues", [])
                if issues:
                    st.markdown("**问题：**")
                    for issue in issues:
                        st.markdown(f"- {issue}")
                
                suggestions = feedback.get("suggestions", [])
                if suggestions:
                    st.markdown("**建议：**")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
        
        if student_feedback:
            st.markdown("### 🎓 模拟学生测试")
            for feedback in student_feedback:
                understood = feedback.get("understood", False)
                score = feedback.get("comprehension_score", 0.0)
                
                if understood and score >= 0.8:
                    st.success(f"✅ 理解良好 (得分: {score:.2f})")
                elif score >= 0.6:
                    st.info(f"ℹ️ 基本理解 (得分: {score:.2f})")
                else:
                    st.warning(f"⚠️ 理解困难 (得分: {score:.2f})")
                
                reasoning = feedback.get("reasoning", "")
                if reasoning:
                    st.markdown(f"**反馈：** {reasoning}")
                
                confusion = feedback.get("confusion_points", [])
                if confusion:
                    st.markdown("**困惑点：**")
                    for point in confusion:
                        st.markdown(f"- {point}")
                
                missing = feedback.get("missing_prerequisites", [])
                if missing:
                    st.markdown("**缺失的前置知识：**")
                    for prereq in missing:
                        st.markdown(f"- {prereq}")


def render_download_section(result: Dict[str, Any]):
    """渲染下载部分"""
    if not result:
        return
    
    st.header("💾 导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 下载 JSON",
            data=json_str,
            file_name="learning_path.json",
            mime="application/json"
        )
    
    with col2:
        # Export as Markdown
        md_content = generate_markdown_report(result)
        st.download_button(
            label="📥 下载 Markdown",
            data=md_content,
            file_name="learning_path.md",
            mime="text/markdown"
        )


def generate_markdown_report(result: Dict[str, Any]) -> str:
    """生成 Markdown 格式的学习报告"""
    learning_path = result.get("learning_path", {})
    nodes = learning_path.get("nodes", [])
    resources = result.get("resources", {})
    
    md = f"""# 个性化学习路径：{learning_path.get("topic", "Unknown")}

**目标水平：** {learning_path.get("target_level", "N/A")}
**总学时：** {learning_path.get("total_hours", 0):.1f} 小时
**节点数：** {len(nodes)}

---

## 学习路径

"""
    
    for i, node in enumerate(nodes):
        md += f"""
### {i + 1}. {node.get("title", "Untitled")}

**难度：** {node.get("difficulty", "N/A")}
**预计学时：** {node.get("estimated_hours", 1):.1f} 小时

{node.get("description", "")}

**关键概念：** {", ".join(node.get("key_concepts", []))}

**前置要求：** {", ".join(node.get("prerequisites", [])) or "无"}

#### 学习资源

"""
        node_resources = resources.get(node.get("id"), [])
        for res in node_resources:
            url = res.get("url")
            title = res.get("title", "Untitled")
            if url:
                md += f"- [{title}]({url}) ({res.get('type', 'resource')})\n"
            else:
                md += f"- {title} ({res.get('type', 'resource')})\n"
            
            if res.get("description"):
                md += f"  - {res.get('description')}\n"
        
        md += "\n---\n"
    
    return md


def main():
    """主函数"""
    st.set_page_config(
        page_title="多智能体教学系统",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎓 多智能体个性化教学系统")
    st.markdown("**个性化学习路径规划委员会** - 为你量身定制的学习路径")
    
    init_session_state()
    configure_api_sidebar()
    
    # Main content
    render_learning_path_form()
    
    # Show results if available
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        render_process_log(result)
        render_learning_path(result)
        render_download_section(result)


if __name__ == "__main__":
    main()

