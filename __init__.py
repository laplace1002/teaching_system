"""
Multi-Agent Teaching System (多智能体教学系统)
个性化学习路径规划委员会 (The "Board of Education" for One)

A meta-cognitive teaching system using multiple AI agents to create,
review, and pressure-test personalized learning paths.

Agents:
- Curriculum Designer (架构师): Designs knowledge graphs and learning outlines
- Pedagogue (心理学家): Reviews difficulty and learning psychology
- Librarian (资源搜集者): Finds and curates learning resources
- Simulated Student (用户模拟器): Tests comprehension from learner's perspective

Powered by LangChain/LangGraph and DeepSeek API.
"""

from .agents import (
    DifficultyLevel,
    AgentRole,
    KnowledgeNode,
    LearningPath,
    Resource,
    PedagogicalFeedback,
    StudentFeedback,
    CurriculumDesigner,
    Pedagogue,
    Librarian,
    SimulatedStudent,
)

from .workflow import TeachingSystemWorkflow, WorkflowState

from .system import TeachingSystem, create_teaching_system

__version__ = "0.1.0"

__all__ = [
    # Enums
    "DifficultyLevel",
    "AgentRole",
    
    # Data structures
    "KnowledgeNode",
    "LearningPath",
    "Resource",
    "PedagogicalFeedback",
    "StudentFeedback",
    "WorkflowState",
    
    # Agents
    "CurriculumDesigner",
    "Pedagogue",
    "Librarian",
    "SimulatedStudent",
    
    # Workflow
    "TeachingSystemWorkflow",
    
    # Main system
    "TeachingSystem",
    "create_teaching_system",
]

