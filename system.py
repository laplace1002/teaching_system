"""
Multi-Agent Teaching System - Main Orchestrator
主系统入口，提供简单的 API
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from .workflow import TeachingSystemWorkflow
    from .agents import DifficultyLevel
except ImportError:
    # Direct execution fallback
    from workflow import TeachingSystemWorkflow
    from agents import DifficultyLevel


@dataclass
class DeepSeekConfig:
    """DeepSeek API 配置"""
    api_key: Optional[str] = None
    api_base: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"
    reasoner_model: str = "deepseek-reasoner"
    temperature: float = 0.7


class TeachingSystem:
    """
    多智能体教学系统主类
    
    用法示例:
    ```python
    from teaching_sys import TeachingSystem
    
    system = TeachingSystem()
    result = system.create_learning_path(
        topic="量子物理",
        user_level="beginner",
        target_level="intermediate"
    )
    
    print(result["learning_path"])
    print(result["messages"])
    ```
    """
    
    def __init__(
        self,
        config: Optional[DeepSeekConfig] = None,
        llm: Optional[object] = None,
        max_revision_iterations: int = 1,  # OPTIMIZED: Reduced from 3 to 1
        enable_caching: bool = True  # NEW: Enable response caching
    ):
        """
        初始化教学系统
        
        Args:
            config: DeepSeek API 配置（如果为 None，从环境变量读取）
            llm: 自定义 LLM 实例（高级用法）
            max_revision_iterations: 每个节点的最大修订次数 (默认1以提高速度)
            enable_caching: 是否启用缓存以减少API调用
        """
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        self.enable_caching = enable_caching
        
        # Initialize LLM
        if llm is None and ChatOpenAI is not None:
            if config.api_key:
                # Set environment variables for LangChain
                os.environ["OPENAI_API_KEY"] = config.api_key
                os.environ["OPENAI_API_BASE"] = config.api_base
                
                llm = ChatOpenAI(
                    model=config.chat_model,
                    temperature=config.temperature,
                    request_timeout=45,  # ⚡ 强制 45秒超时，防止挂死
                    max_tokens=1000,      # 限制 Token，防止生成论文
                    max_retries=1         # LangChain 内部重试次数
                )
            else:
                print("Warning: No API key provided. Running in fallback mode.")
        
        self.llm = llm
        
        # Initialize workflow
        self.workflow = TeachingSystemWorkflow(
            llm=llm,
            max_revision_iterations=max_revision_iterations,
            enable_caching=enable_caching,
            max_workers=3  # ⚡ 并行处理线程数
        )
    
    def _load_config_from_env(self) -> DeepSeekConfig:
        """从环境变量加载配置"""
        # Try to load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        return DeepSeekConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            reasoner_model=os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7"))
        )
    
    def create_learning_path(
        self,
        topic: str,
        user_level: str = "beginner",
        target_level: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        创建个性化学习路径
        
        这个学习路径经过了多个 Agent 的审核和"压力测试"：
        1. 架构师设计知识图谱
        2. 心理学家审核难度
        3. 资源搜集者查找材料
        4. 模拟学生测试理解度
        
        Args:
            topic: 学习主题（例如："量子物理", "机器学习", "西班牙语"）
            user_level: 用户当前水平 ("beginner", "intermediate", "advanced", "expert")
            target_level: 目标水平 ("beginner", "intermediate", "advanced", "expert")
            
        Returns:
            包含以下内容的字典：
            - learning_path: 完整的学习路径
            - resources: 每个节点的学习资源
            - pedagogue_feedback: 心理学家的反馈
            - student_feedback: 模拟学生的反馈
            - messages: 处理过程的消息日志
            - completed: 是否成功完成
        """
        user_level_enum = DifficultyLevel(user_level)
        target_level_enum = DifficultyLevel(target_level)
        
        result = self.workflow.create_learning_path(
            topic=topic,
            user_level=user_level_enum,
            target_level=target_level_enum
        )
        
        return result
    
    def get_node_details(self, result: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """
        获取特定学习节点的详细信息
        
        Args:
            result: create_learning_path 的返回结果
            node_id: 节点 ID
            
        Returns:
            节点详情，包括资源、反馈等
        """
        learning_path = result.get("learning_path", {})
        nodes = learning_path.get("nodes", [])
        
        # Find the node
        node = None
        for n in nodes:
            if n["id"] == node_id:
                node = n
                break
        
        if node is None:
            return {"error": f"Node {node_id} not found"}
        
        # Get resources
        resources = result.get("resources", {}).get(node_id, [])
        
        # Get feedback
        pedagogue_feedback = [
            f for f in result.get("pedagogue_feedback", [])
            if f.get("node_id") == node_id
        ]
        
        student_feedback = [
            f for f in result.get("student_feedback", [])
            if f.get("node_id") == node_id
        ]
        
        return {
            "node": node,
            "resources": resources,
            "pedagogue_feedback": pedagogue_feedback,
            "student_feedback": student_feedback
        }


def create_teaching_system(
    api_key: Optional[str] = None,
    api_base: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    **kwargs
) -> TeachingSystem:
    """
    便捷函数：创建教学系统实例
    
    Args:
        api_key: DeepSeek API key（如果为 None，从环境变量读取）
        api_base: API 基础 URL
        model: 使用的模型名称
        **kwargs: 其他参数传递给 TeachingSystem
        
    Returns:
        TeachingSystem 实例
    """
    config = DeepSeekConfig(
        api_key=api_key,
        api_base=api_base,
        chat_model=model
    )
    
    return TeachingSystem(config=config, **kwargs)


__all__ = ["TeachingSystem", "DeepSeekConfig", "create_teaching_system"]

