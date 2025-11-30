"""
Multi-Agent Teaching System - Robust Agents

⚡ 改进：增强 JSON 解析鲁棒性，增加 Safe Invoke

"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None
    AIMessage = None


class DifficultyLevel(str, Enum):
    """学习难度等级"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AgentRole(str, Enum):
    """Agent 角色类型"""
    ARCHITECT = "curriculum_designer"
    PEDAGOGUE = "pedagogue"
    LIBRARIAN = "librarian"
    SIMULATED_STUDENT = "simulated_student"


@dataclass
class KnowledgeNode:
    """知识图谱节点"""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    estimated_hours: float = 1.0
    key_concepts: List[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """学习路径"""
    topic: str
    target_level: DifficultyLevel
    nodes: List[KnowledgeNode] = field(default_factory=list)
    total_hours: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """学习资源"""
    title: str
    type: str  # video, article, book, exercise, etc.
    url: Optional[str] = None
    description: str = ""
    difficulty: Optional[DifficultyLevel] = None
    estimated_time: Optional[float] = None


@dataclass
class PedagogicalFeedback:
    """心理学家的反馈"""
    approved: bool
    node_id: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    adjusted_difficulty: Optional[DifficultyLevel] = None
    reasoning: str = ""


@dataclass
class StudentFeedback:
    """模拟学生的反馈"""
    understood: bool
    node_id: str
    comprehension_score: float  # 0.0 to 1.0
    confusion_points: List[str] = field(default_factory=list)
    missing_prerequisites: List[str] = field(default_factory=list)
    reasoning: str = ""


def _clean_json_response(text: str) -> str:
    """Ultra-robust JSON cleaner"""
    if not text: return "{}"
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 尝试找到第一个 { 和最后一个 }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    return text.strip()


def safe_invoke_llm(llm, messages, default_return={}, retries=1):
    """
    ⚡ 安全调用 LLM 的包装器
    包含：超时处理、错误捕获、自动重试、JSON解析保护
    """
    if not llm: return default_return
    
    content = ""
    for attempt in range(retries + 1):
        try:
            # 调用 LLM
            response = llm.invoke(messages)
            content = getattr(response, "content", str(response))
            
            # 清理和解析 JSON
            cleaned_json = _clean_json_response(content)
            return json.loads(cleaned_json)
            
        except json.JSONDecodeError:
            if attempt == retries:
                print(f"⚠️ JSON Parse Failed after {retries} retries. Content: {content[:50]}...")
                return default_return
        except Exception as e:
            if attempt == retries:
                print(f"⚠️ LLM Invoke Failed: {str(e)}")
                return default_return
            time.sleep(1) # Backoff
            
    return default_return


def detect_context_drift(
    content: str,
    allowed_keywords: List[str],
    forbidden_keywords: List[str]
) -> Dict[str, Any]:
    """
    关键词守护机制 - 检测上下文漂移
    
    Args:
        content: 要检查的内容（节点描述、资源标题等）
        allowed_keywords: 允许的关键词列表
        forbidden_keywords: 禁止的关键词列表
        
    Returns:
        包含以下内容的字典：
        - has_drift: 是否检测到漂移
        - forbidden_found: 发现的禁止关键词列表
        - confidence: 漂移置信度 (0.0-1.0)
    """
    content_lower = content.lower()
    
    forbidden_found = []
    for keyword in forbidden_keywords:
        if keyword.lower() in content_lower:
            forbidden_found.append(keyword)
    
    allowed_count = sum(1 for kw in allowed_keywords if kw.lower() in content_lower)
    
    has_drift = len(forbidden_found) > 0
    
    # Calculate confidence: ratio of forbidden to total relevant keywords
    total_relevant = allowed_count + len(forbidden_found)
    confidence = len(forbidden_found) / max(total_relevant, 1)
    
    return {
        "has_drift": has_drift,
        "forbidden_found": forbidden_found,
        "confidence": confidence,
        "allowed_count": allowed_count
    }


class CurriculumDesigner:
    """
    架构师 Agent - 负责设计大纲和知识图谱
    
    职责：
    - 将学习目标拆解为知识图谱
    - 设计学习路径的结构
    - 定义知识节点和依赖关系
    """
    
    SYSTEM_PROMPT = """Curriculum Designer: Create learning path. Output ONLY JSON. No markdown.
Format: {"nodes": [{"id": "str", "title": "str", "description": "str", "difficulty": "level", "prerequisites": [], "estimated_hours": float, "key_concepts": []}]}"""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm
        self.role = AgentRole.ARCHITECT
    
    def calibrate_curriculum(
        self,
        topic: str,
        user_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """
        课程校准 - 确认用户的真实水平和需要的基础知识
        
        Returns:
            包含以下内容的字典：
            - needs_fundamentals: 是否需要基础课程
            - missing_skills: 缺失的基础技能列表
            - recommended_prerequisites: 推荐的前置课程
            - confirmed_level: 确认的用户水平
        """
        if not self.llm:
            return self._default_calibration(user_level)
        
        prompt = f"""Calibrate: {topic} for {user_level.value}
What foundation needed? Prerequisites? Avoid what?

JSON:
{{
  "needs_fundamentals": true/false,
  "missing_skills": ["skill1"],
  "recommended_prerequisites": ["prereq1"],
  "confirmed_level": "{user_level.value}",
  "reasoning": "1 sentence",
  "global_constraints": {{
    "primary_domain": "field",
    "forbidden_topics": ["unrelated1"]
  }}
}}"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        # 默认值
        default = {
            "needs_fundamentals": False, 
            "primary_domain": "General",
            "missing_skills": [],
            "recommended_prerequisites": [],
            "confirmed_level": user_level.value,
            "global_constraints": {
                "primary_domain": "General",
                "forbidden_topics": []
            }
        }
        return safe_invoke_llm(self.llm, messages, default)
    
    def _default_calibration(self, user_level: DifficultyLevel) -> Dict[str, Any]:
        """默认校准"""
        if user_level == DifficultyLevel.BEGINNER:
            return {
                "needs_fundamentals": True,
                "missing_skills": ["basic programming", "fundamental concepts"],
                "recommended_prerequisites": ["Introduction course"],
                "confirmed_level": user_level.value,
                "reasoning": "Beginners typically need foundational concepts",
                "global_constraints": {
                    "primary_domain": "General",
                    "forbidden_topics": []
                }
            }
        return {
            "needs_fundamentals": False,
            "missing_skills": [],
            "recommended_prerequisites": [],
            "confirmed_level": user_level.value,
            "reasoning": "Intermediate/advanced learners have foundations",
            "global_constraints": {
                "primary_domain": "General",
                "forbidden_topics": []
            }
        }
        
    def design_curriculum(
        self,
        topic: str,
        user_level: DifficultyLevel,
        target_level: DifficultyLevel,
        constraints: Optional[Dict[str, Any]] = None,
        calibration: Optional[Dict[str, Any]] = None
    ) -> LearningPath:
        """设计课程大纲"""
        
        if not self.llm:
            return self._fallback_curriculum(topic, user_level, target_level)
        
        # Use calibration if provided
        constraints_str = ""
        if constraints:
            constraints_str = f"\nConstraints: {json.dumps(constraints, ensure_ascii=False)}"
        
        calibration_context = ""
        if calibration:
            calibration_context = f"""
Calibration: Needs fundamentals={calibration.get('needs_fundamentals', False)}
Missing: {', '.join(calibration.get('missing_skills', [])[:2])}
Domain: {calibration.get('global_constraints', {}).get('primary_domain', 'N/A')}
AVOID: {', '.join(calibration.get('global_constraints', {}).get('forbidden_topics', [])[:2])}
"""
        
        prompt = f"""Design path: {topic}
{user_level.value} → {target_level.value}
{calibration_context}
Create 4-6 nodes.

JSON:
{{
  "nodes": [
    {{
      "id": "node_1",
      "title": "str",
      "description": "what learner understands",
      "difficulty": "beginner|intermediate|advanced|expert",
      "prerequisites": ["node_0"],
      "estimated_hours": 2.5,
      "key_concepts": ["c1", "c2"]
    }}
  ]
}}"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        data = safe_invoke_llm(self.llm, messages, {"nodes": []})
        
        nodes = []
        raw_nodes = data.get("nodes", [])
        
        # 兜底：如果 LLM 返回空，生成一个默认节点
        if not raw_nodes:
            raw_nodes = [{"id": "1", "title": f"Intro to {topic}", "description": "Basics", "difficulty": "beginner"}]
            
        for n in raw_nodes:
            # 容错处理：确保所有字段都有默认值
            nodes.append(KnowledgeNode(
                id=str(n.get("id", len(nodes)+1)),
                title=str(n.get("title", "Untitled")),
                description=str(n.get("description", "")),
                difficulty=DifficultyLevel(n.get("difficulty", "beginner")) if n.get("difficulty") in ["beginner", "intermediate", "advanced", "expert"] else DifficultyLevel.BEGINNER,
                estimated_hours=float(n.get("estimated_hours", 1.0)),
                prerequisites=n.get("prerequisites", []),
                key_concepts=n.get("key_concepts", [])
            ))
            
        total_hours = sum(n.estimated_hours for n in nodes)
        return LearningPath(
            topic=topic,
            target_level=target_level,
            nodes=nodes,
            total_hours=total_hours,
            metadata={"user_level": user_level.value}
        )
    
    def _fallback_curriculum(
        self,
        topic: str,
        user_level: DifficultyLevel,
        target_level: DifficultyLevel
    ) -> LearningPath:
        """后备方案：简单的课程结构"""
        nodes = [
            KnowledgeNode(
                id="intro",
                title=f"Introduction to {topic}",
                description=f"Basic concepts and overview of {topic}",
                difficulty=DifficultyLevel.BEGINNER,
                estimated_hours=2.0
            ),
            KnowledgeNode(
                id="fundamentals",
                title=f"Fundamentals of {topic}",
                description=f"Core principles and theories",
                difficulty=DifficultyLevel.INTERMEDIATE,
                prerequisites=["intro"],
                estimated_hours=4.0
            ),
        ]
        
        return LearningPath(
            topic=topic,
            target_level=target_level,
            nodes=nodes,
            total_hours=6.0
        )


class Pedagogue:
    """
    心理学家 Agent - 负责调整难度和确保学习心理学原则
    
    职责：
    - 评估课程设计的难度是否合适
    - 检查是否会导致学习者受挫
    - 建议增加或调整前置知识
    - 确保认知负荷适当
    """
    
    SYSTEM_PROMPT = """Educational Psychologist: Evaluate difficulty. Output ONLY JSON. No markdown. No reasoning outside JSON.
Format: {"approved": bool, "issues": [], "suggestions": [], "adjusted_difficulty": "level|null", "reasoning": "brief"}"""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm
        self.role = AgentRole.PEDAGOGUE
        
    def evaluate_node(
        self,
        node: KnowledgeNode,
        user_level: DifficultyLevel,
        previous_nodes: List[KnowledgeNode]
    ) -> PedagogicalFeedback:
        """评估单个知识节点"""
        
        if not self.llm:
            return self._simple_evaluation(node, user_level)
        
        prev_summary = "\n".join([
            f"- {n.title} (Difficulty: {n.difficulty.value})"
            for n in previous_nodes
        ])
        
        prompt = f"""Evaluate for {user_level.value}:
Node: {node.title} ({node.difficulty.value})
Concepts: {', '.join(node.key_concepts[:3])}
Previous: {prev_summary if prev_summary else "None"}

Is difficulty appropriate? Will learner understand?

JSON:
{{
  "approved": true/false,
  "issues": ["issue1"],
  "suggestions": ["sug1"],
  "adjusted_difficulty": "level|null",
  "reasoning": "1 sentence"
}}"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        default_feedback = {
            "approved": True,
            "issues": [],
            "suggestions": [],
            "adjusted_difficulty": None,
            "reasoning": "Fallback evaluation"
        }
        data = safe_invoke_llm(self.llm, messages, default_feedback)
        
        adjusted_diff = None
        if data.get("adjusted_difficulty") and data["adjusted_difficulty"] != "null":
            try:
                adjusted_diff = DifficultyLevel(data["adjusted_difficulty"])
            except:
                pass
        
        return PedagogicalFeedback(
            approved=data.get("approved", True),
            node_id=node.id,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            adjusted_difficulty=adjusted_diff,
            reasoning=data.get("reasoning", "")
        )
    
    def _simple_evaluation(
        self,
        node: KnowledgeNode,
        user_level: DifficultyLevel
    ) -> PedagogicalFeedback:
        """简单的启发式评估"""
        difficulty_order = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]
        
        user_idx = difficulty_order.index(user_level)
        node_idx = difficulty_order.index(node.difficulty)
        
        # If node is more than 1 level above user, reject
        if node_idx - user_idx > 1:
            return PedagogicalFeedback(
                approved=False,
                node_id=node.id,
                issues=["Difficulty too high for current level"],
                suggestions=["Add intermediate steps", "Provide more prerequisites"],
                reasoning="Gap between user level and content is too large"
            )
        
        return PedagogicalFeedback(
            approved=True,
            node_id=node.id,
            reasoning="Difficulty appears appropriate"
        )


class Librarian:
    """
    资源搜集 Agent - 负责查找和推荐学习资源
    
    职责：
    - 根据知识节点查找相关资源
    - 推荐视频、文章、书籍等
    - 评估资源质量和适配度
    """
    
    SYSTEM_PROMPT = """Librarian: Find best resources. Output ONLY JSON. No markdown.
Format: {"resources": [{"title": "str", "type": "video|article|book", "url": "str", "description": "brief", "difficulty": "level", "estimated_time": float}]}"""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm
        self.role = AgentRole.LIBRARIAN
        
    def find_resources(
        self,
        node: KnowledgeNode,
        num_resources: int = 3,
        global_context: Optional[Dict[str, Any]] = None
    ) -> List[Resource]:
        """为知识节点查找学习资源"""
        
        if not self.llm:
            return self._fallback_resources(node)
        
        # Build global constraints
        context_constraints = ""
        if global_context:
            primary_domain = global_context.get("primary_domain", "")
            forbidden_topics = global_context.get("forbidden_topics", [])
            topic = global_context.get("topic", "")
            
            if primary_domain or forbidden_topics or topic:
                context_constraints = f"""
⚠️ GLOBAL CONTEXT CONSTRAINTS - CRITICAL:
- Main Topic: {topic}
- Primary Domain: {primary_domain}
- FORBIDDEN TOPICS (DO NOT recommend resources about): {', '.join(forbidden_topics)}
- All resources MUST be directly related to {topic} in the context of {primary_domain}

YOU MUST REJECT any resource that strays into forbidden topics!
"""
        
        prompt = f"""Find {num_resources} resources:
{node.title} ({node.difficulty.value})
Concepts: {', '.join(node.key_concepts[:3])}
{context_constraints}

JSON:
{{
  "resources": [
    {{
      "title": "str",
      "type": "video|article|book",
      "url": "https://...",
      "description": "brief",
      "difficulty": "{node.difficulty.value}",
      "estimated_time": 2.0
    }}
  ]
}}"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        data = safe_invoke_llm(self.llm, messages, {"resources": []})
        
        resources = []
        for r in data.get("resources", []):
            try:
                difficulty = None
                if r.get("difficulty"):
                    difficulty = DifficultyLevel(r["difficulty"])
                resources.append(Resource(
                    title=r.get("title", "Resource"),
                    type=r.get("type", "article"),
                    url=r.get("url"),
                    description=r.get("description", ""),
                    difficulty=difficulty,
                    estimated_time=r.get("estimated_time")
                ))
            except Exception as e:
                # Skip invalid resources
                print(f"⚠️ Skipping invalid resource: {e}")
                continue
        
        # 如果没找到资源，使用兜底资源
        if not resources:
            return self._fallback_resources(node)
        
        return resources
    
    def _fallback_resources(self, node: KnowledgeNode) -> List[Resource]:
        """后备资源推荐"""
        return [
            Resource(
                title=f"Introduction to {node.title}",
                type="article",
                description=f"Learn about {node.title}",
                difficulty=node.difficulty
            )
        ]


class RepairAgent:
    """
    修复者 Agent - 负责根据反馈修复和优化课程节点
    
    职责：
    - 接收 Pedagogue 和 Student 的反馈
    - 修改节点描述、难度或拆分节点
    - 自动插入缺失的前置节点（Scaffolding）
    - 确保修复后的节点符合要求
    """
    
    SYSTEM_PROMPT = """Repair Agent: Fix node issues. Output ONLY JSON. No markdown.
Actions: "revise" (simplify), "split" (break into 2), "insert_prerequisite" (add foundation).
Format: {"action": "str", "reasoning": "brief", "revised_node": {...} OR "new_nodes": [...]}"""

    def __init__(self, llm: Optional[object] = None):
        self.llm = llm
        self.role = "repair_agent"
    
    def repair_node(
        self,
        node: KnowledgeNode,
        pedagogue_feedback: PedagogicalFeedback,
        student_feedback: Optional[StudentFeedback],
        topic_context: str,
        user_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """
        修复存在问题的节点
        
        Returns:
            包含以下内容的字典：
            - action: "revise" (修改节点) | "split" (拆分节点) | "insert_prerequisite" (插入前置节点)
            - revised_node: 修改后的节点（如果action是revise）
            - new_nodes: 新节点列表（如果action是split或insert_prerequisite）
            - reasoning: 修复原因说明
        """
        if not self.llm:
            return self._simple_repair(node, pedagogue_feedback, student_feedback)
        
        issues_str = "\n".join([f"- {issue}" for issue in pedagogue_feedback.issues])
        suggestions_str = "\n".join([f"- {sug}" for sug in pedagogue_feedback.suggestions])
        
        student_issues_str = ""
        if student_feedback:
            student_issues_str = f"""
Student Feedback:
- Understood: {student_feedback.understood}
- Comprehension Score: {student_feedback.comprehension_score}
- Confusion Points: {', '.join(student_feedback.confusion_points)}
- Missing Prerequisites: {', '.join(student_feedback.missing_prerequisites)}
"""
        
        prompt = f"""Repair: {node.title} ({node.difficulty.value})
User: {user_level.value} | Topic: {topic_context}
Issues: {'; '.join(pedagogue_feedback.issues[:1])}
{student_issues_str}

Action? revise|split|insert_prerequisite

JSON:
{{
  "action": "revise|split|insert_prerequisite",
  "reasoning": "1 sentence",
  "revised_node": {{  // if revise
    "id": "{node.id}",
    "title": "str",
    "description": "str",
    "difficulty": "{user_level.value}",
    "prerequisites": [],
    "estimated_hours": 2.0,
    "key_concepts": ["c1"]
  }},
  "new_nodes": [...]  // if split/insert
}}"""
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        default_result = {
            "action": "revise",
            "reasoning": "Fallback repair",
            "revised_node": None
        }
        data = safe_invoke_llm(self.llm, messages, default_result)
        
        result = {
            "action": data.get("action", "revise"),
            "reasoning": data.get("reasoning", "")
        }
        
        if data.get("action") == "revise" and "revised_node" in data and data["revised_node"]:
            try:
                result["revised_node"] = KnowledgeNode(
                    id=str(data["revised_node"].get("id", node.id)),
                    title=str(data["revised_node"].get("title", node.title)),
                    description=str(data["revised_node"].get("description", node.description)),
                    difficulty=DifficultyLevel(data["revised_node"].get("difficulty", node.difficulty.value)),
                    prerequisites=data["revised_node"].get("prerequisites", []),
                    estimated_hours=float(data["revised_node"].get("estimated_hours", node.estimated_hours)),
                    key_concepts=data["revised_node"].get("key_concepts", [])
                )
            except Exception as e:
                print(f"⚠️ Error parsing revised_node: {e}")
                return self._simple_repair(node, pedagogue_feedback, student_feedback)
        
        if ("split" in data.get("action", "") or "insert" in data.get("action", "")) and "new_nodes" in data:
            result["new_nodes"] = []
            for n in data["new_nodes"]:
                try:
                    result["new_nodes"].append(KnowledgeNode(
                        id=str(n.get("id", "node")),
                        title=str(n.get("title", "Node")),
                        description=str(n.get("description", "")),
                        difficulty=DifficultyLevel(n.get("difficulty", "beginner")),
                        prerequisites=n.get("prerequisites", []),
                        estimated_hours=float(n.get("estimated_hours", 1.0)),
                        key_concepts=n.get("key_concepts", [])
                    ))
                except Exception as e:
                    print(f"⚠️ Error parsing new_node: {e}")
                    continue
        
        return result
    
    def _simple_repair(
        self,
        node: KnowledgeNode,
        pedagogue_feedback: PedagogicalFeedback,
        student_feedback: Optional[StudentFeedback]
    ) -> Dict[str, Any]:
        """简单的启发式修复"""
        # 如果建议了调整难度，就降低一级
        if pedagogue_feedback.adjusted_difficulty:
            revised = KnowledgeNode(
                id=node.id,
                title=node.title,
                description=f"[Simplified] {node.description}",
                difficulty=pedagogue_feedback.adjusted_difficulty,
                prerequisites=node.prerequisites,
                estimated_hours=node.estimated_hours,
                key_concepts=node.key_concepts
            )
            return {
                "action": "revise",
                "revised_node": revised,
                "reasoning": "Adjusted difficulty based on feedback"
            }
        
        return {
            "action": "revise",
            "revised_node": node,
            "reasoning": "No changes needed"
        }


class SimulatedStudent:
    """
    用户模拟 Agent - 模拟真实学生测试内容
    
    职责：
    - 从用户当前水平出发，尝试理解学习内容
    - 识别理解困难点
    - 检测缺失的前置知识
    - 提供学习者视角的反馈
    
    这是关键的反馈闭环机制！
    """
    
    SYSTEM_PROMPT_TEMPLATE = """Simulated {level} student: Can you understand this? Output ONLY JSON. No markdown.
Format: {{"understood": bool, "comprehension_score": 0.0-1.0, "confusion_points": [], "missing_prerequisites": [], "reasoning": "brief"}}"""

    def __init__(self, user_level: DifficultyLevel, llm: Optional[object] = None):
        self.llm = llm
        self.user_level = user_level
        self.role = AgentRole.SIMULATED_STUDENT
        self.knowledge_profile = []  # Track what student has learned
        
    def test_comprehension(
        self,
        node: KnowledgeNode,
        resources: List[Resource],
        previous_nodes: List[KnowledgeNode]
    ) -> StudentFeedback:
        """测试对节点内容的理解程度"""
        
        if not self.llm:
            return self._simple_comprehension_test(node)
        
        # Build dynamic student profile based on learned nodes
        learned_skills = []
        for pn in previous_nodes:
            if pn.id in self.knowledge_profile:
                learned_skills.append(f"✓ {pn.title} (mastered)")
            else:
                learned_skills.append(f"- {pn.title}")
        
        profile_update = ""
        if self.knowledge_profile:
            profile_update = f"""
YOUR UPDATED KNOWLEDGE PROFILE:
You have successfully mastered {len(self.knowledge_profile)} previous node(s).
Your knowledge now includes: {', '.join([n.title for n in previous_nodes if n.id in self.knowledge_profile])}
You should leverage this knowledge when evaluating the current node.
"""
        
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(level=self.user_level.value) + profile_update
        
        prev_knowledge = "\n".join(learned_skills)
        
        resources_str = "\n".join([
            f"- {r.title} ({r.type}): {r.description}"
            for r in resources[:3]  # Limit to avoid token bloat
        ])
        
        prompt = f"""As {self.user_level.value} student: {node.title}
Difficulty: {node.difficulty.value} | Concepts: {', '.join(node.key_concepts[:2])}
Previous: {prev_knowledge if prev_knowledge else "None"}

Can you understand? What's confusing? Rate 0.0-1.0.

JSON:
{{
  "understood": true/false,
  "comprehension_score": 0.0-1.0,
  "confusion_points": ["point1"],
  "missing_prerequisites": ["prereq1"],
  "reasoning": "1 sentence"
}}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        default_feedback = {
            "understood": True,
            "comprehension_score": 0.7,
            "confusion_points": [],
            "missing_prerequisites": [],
            "reasoning": "Fallback feedback"
        }
        data = safe_invoke_llm(self.llm, messages, default_feedback)
        
        feedback = StudentFeedback(
            understood=data.get("understood", True),
            node_id=node.id,
            comprehension_score=float(data.get("comprehension_score", 0.7)),
            confusion_points=data.get("confusion_points", []),
            missing_prerequisites=data.get("missing_prerequisites", []),
            reasoning=data.get("reasoning", "")
        )
        
        # Update student profile if understood well
        if feedback.understood and feedback.comprehension_score >= 0.7:
            self.knowledge_profile.append(node.id)
        
        return feedback
    
    def _simple_comprehension_test(self, node: KnowledgeNode) -> StudentFeedback:
        """简单的理解度测试"""
        difficulty_order = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]
        
        user_idx = difficulty_order.index(self.user_level)
        node_idx = difficulty_order.index(node.difficulty)
        
        gap = node_idx - user_idx
        
        if gap > 1:
            return StudentFeedback(
                understood=False,
                node_id=node.id,
                comprehension_score=0.3,
                confusion_points=["Content too advanced"],
                missing_prerequisites=["Intermediate concepts needed"],
                reasoning="Difficulty gap is too large"
            )
        elif gap == 1:
            return StudentFeedback(
                understood=True,
                node_id=node.id,
                comprehension_score=0.7,
                confusion_points=[],
                reasoning="Challenging but manageable"
            )
        else:
            return StudentFeedback(
                understood=True,
                node_id=node.id,
                comprehension_score=0.9,
                reasoning="Content matches my level well"
            )


__all__ = [
    "DifficultyLevel",
    "AgentRole",
    "KnowledgeNode",
    "LearningPath",
    "Resource",
    "PedagogicalFeedback",
    "StudentFeedback",
    "CurriculumDesigner",
    "Pedagogue",
    "Librarian",
    "SimulatedStudent",
    "RepairAgent",
]

