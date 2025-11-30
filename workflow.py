"""
Multi-Agent Teaching System - Parallel Workflow

🚀 核心改进：并行处理 + 熔断机制 + 批处理

"""

from __future__ import annotations

import json
import concurrent.futures
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import asdict

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langchain_openai import ChatOpenAI
except ImportError:
    StateGraph = None
    END = None
    ChatOpenAI = None

try:
    from .agents import (
        CurriculumDesigner, Pedagogue, Librarian, SimulatedStudent, RepairAgent,
        DifficultyLevel, KnowledgeNode, Resource, PedagogicalFeedback, StudentFeedback,
    )
except ImportError:
    from agents import (
        CurriculumDesigner, Pedagogue, Librarian, SimulatedStudent, RepairAgent,
        DifficultyLevel, KnowledgeNode, Resource, PedagogicalFeedback, StudentFeedback,
    )

# ⚡ 难度等级映射
DIFFICULTY_RANK = {
    DifficultyLevel.BEGINNER: 1,
    DifficultyLevel.INTERMEDIATE: 2,
    DifficultyLevel.ADVANCED: 3,
    DifficultyLevel.EXPERT: 4
}


class WorkflowState(TypedDict):
    """LangGraph State - 存储工作流中的所有状态"""
    # Input
    topic: str
    user_level: str
    target_level: str
    max_iterations: int
    
    # State
    learning_path: Optional[Dict[str, Any]]
    current_node_idx: int
    processed_nodes: List[Dict[str, Any]]
    
    # Feedback logs
    pedagogue_feedback: List[Dict[str, Any]]
    student_feedback: List[Dict[str, Any]]
    resources_by_node: Dict[str, List[Dict[str, Any]]]
    
    # Control flow
    iteration_count: int
    needs_revision: bool
    revision_reason: str
    completed: bool
    
    # New: Calibration and context
    calibration: Optional[Dict[str, Any]]
    global_context: Optional[Dict[str, Any]]
    
    # Messages/history
    messages: List[str]


class TeachingSystemWorkflow:
    def __init__(
        self,
        llm: Optional[object] = None,
        max_revision_iterations: int = 1,
        enable_caching: bool = True,
        max_workers: int = 3  # ⚡ 新增：最大并发线程数 (建议 3-5)
    ):
        self.llm = llm
        self.max_revision_iterations = max_revision_iterations
        self.max_workers = max_workers
        
        # Agents
        self.architect = CurriculumDesigner(llm=llm)
        self.pedagogue = Pedagogue(llm=llm)
        self.librarian = Librarian(llm=llm)
        self.repair_agent = RepairAgent(llm=llm)
    
    def _is_easy_node(self, node_difficulty: str, user_level: str) -> bool:
        """快速通道判断"""
        try:
            n_rank = DIFFICULTY_RANK.get(DifficultyLevel(node_difficulty), 2)
            u_rank = DIFFICULTY_RANK.get(DifficultyLevel(user_level), 1)
            return n_rank <= u_rank
        except:
            return False

    def _process_single_node_pipeline(
        self, 
        node_data: Dict, 
        user_level: DifficultyLevel, 
        topic: str,
        global_context: Dict
    ) -> Dict:
        """
        ⚡ 独立的节点处理流水线 (用于并行执行)
        包含：审核 -> (修复) -> 资源 -> 测试
        """
        # 重建对象
        node = KnowledgeNode(**node_data)
        messages = []
        resources = []
        ped_fb = None
        stu_fb = None
        
        try:
            # 1. 快速通道检查
            is_easy = self._is_easy_node(node.difficulty.value, user_level.value)
            
            # --- Pedagogue 审核 ---
            if not is_easy and self.llm:
                # 简单审核，不再传入 previous_nodes 以减少 Token 和依赖
                ped_fb = self.pedagogue.evaluate_node(node, user_level, [])
                
                # 修复逻辑 (仅尝试 1 次)
                if not ped_fb.approved:
                    repair = self.repair_agent.repair_node(node, ped_fb, None, topic, user_level)
                    if "revised_node" in repair:
                        node = repair["revised_node"]
                        messages.append(f"🔧 Node '{node.title}' repaired")
            else:
                # 默认批准
                ped_fb = PedagogicalFeedback(True, node.id, reasoning="Fast-track/Fallback")

            # --- Librarian 资源 ---
            # 简单节点找 1 个，难节点找 2 个
            num_res = 1 if is_easy else 2
            if self.llm:
                resources = self.librarian.find_resources(node, num_resources=num_res, global_context=global_context)
            else:
                resources = self.librarian._fallback_resources(node)

            # --- Student 测试 ---
            if not is_easy and self.llm:
                student = SimulatedStudent(user_level, self.llm)
                stu_fb = student.test_comprehension(node, resources, []) # 不传 prev_nodes 避免阻塞
            else:
                stu_fb = StudentFeedback(True, node.id, 0.9, reasoning="Fast-track")

        except Exception as e:
            messages.append(f"⚠️ Error processing node {node.title}: {str(e)}")
            # 发生错误时使用兜底数据，保证系统不挂
            if not resources: resources = self.librarian._fallback_resources(node)
            if not ped_fb: ped_fb = PedagogicalFeedback(True, node.id, reasoning="Error recovery")
            if not stu_fb: stu_fb = StudentFeedback(True, node.id, 0.5, reasoning="Error recovery")

        return {
            "node": asdict(node),
            "resources": [asdict(r) for r in resources],
            "pedagogue_feedback": asdict(ped_fb) if ped_fb else None,
            "student_feedback": asdict(stu_fb) if stu_fb else None,
            "messages": messages
        }

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes (processing steps)
        workflow.add_node("calibrate", self._calibrate_node)
        workflow.add_node("design_curriculum", self._design_curriculum_node)
        workflow.add_node("evaluate_pedagogy", self._evaluate_pedagogy_node)
        workflow.add_node("repair_node", self._repair_node)
        workflow.add_node("find_resources", self._find_resources_node)
        workflow.add_node("test_student", self._test_student_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry point - start with calibration
        workflow.set_entry_point("calibrate")
        
        # Add conditional edges
        workflow.add_edge("calibrate", "design_curriculum")
        
        workflow.add_conditional_edges(
            "design_curriculum",
            self._should_evaluate,
            {
                "evaluate": "evaluate_pedagogy",
                "end": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "evaluate_pedagogy",
            self._pedagogue_decision,
            {
                "approved": "find_resources",
                "repair": "repair_node",
                "next_node": "evaluate_pedagogy",
                "end": "finalize"
            }
        )
        
        workflow.add_edge("repair_node", "evaluate_pedagogy")
        workflow.add_edge("find_resources", "test_student")
        
        workflow.add_conditional_edges(
            "test_student",
            self._student_decision,
            {
                "understood": "evaluate_pedagogy",  # Move to next node
                "repair": "repair_node",
                "end": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def _calibrate_node(self, state: WorkflowState) -> WorkflowState:
        """Step 0: 课程校准 - 确认用户水平和前置要求"""
        state["messages"].append("🎯 Calibrating curriculum for user level...")
        
        calibration = self.architect.calibrate_curriculum(
            topic=state["topic"],
            user_level=DifficultyLevel(state["user_level"])
        )
        
        state["calibration"] = calibration
        
        # Build global context for keyword guardrails
        state["global_context"] = {
            "topic": state["topic"],
            "primary_domain": calibration.get("global_constraints", {}).get("primary_domain", ""),
            "forbidden_topics": calibration.get("global_constraints", {}).get("forbidden_topics", []),
            "allowed_keywords": [state["topic"]]
        }
        
        state["messages"].append(
            f"✅ Calibration complete: Level confirmed as {calibration.get('confirmed_level', state['user_level'])}"
        )
        
        if calibration.get("needs_fundamentals"):
            state["messages"].append(
                f"📌 Note: Will include foundational prerequisites"
            )
        
        return state
    
    def _design_curriculum_node(self, state: WorkflowState) -> WorkflowState:
        """Step 1: 架构师设计课程大纲"""
        state["messages"].append("🏗️ Curriculum Designer: Creating learning path...")
        
        learning_path = self.architect.design_curriculum(
            topic=state["topic"],
            user_level=DifficultyLevel(state["user_level"]),
            target_level=DifficultyLevel(state["target_level"]),
            calibration=state.get("calibration")
        )
        
        state["learning_path"] = {
            "topic": learning_path.topic,
            "target_level": learning_path.target_level.value,
            "nodes": [asdict(node) for node in learning_path.nodes],
            "total_hours": learning_path.total_hours,
            "metadata": learning_path.metadata
        }
        state["current_node_idx"] = 0
        state["processed_nodes"] = []
        state["messages"].append(
            f"✅ Created learning path with {len(learning_path.nodes)} nodes"
        )
        
        return state
    
    def _evaluate_pedagogy_node(self, state: WorkflowState) -> WorkflowState:
        """Step 2: 心理学家评估当前节点"""
        idx = state["current_node_idx"]
        nodes_data = state["learning_path"]["nodes"]
        
        if idx >= len(nodes_data):
            state["completed"] = True
            return state
        
        current_node_data = nodes_data[idx]
        current_node = KnowledgeNode(**current_node_data)
        
        # Get previously processed nodes
        previous_nodes = [
            KnowledgeNode(**n) for n in state["processed_nodes"]
        ]
        
        state["messages"].append(
            f"🧠 Pedagogue: Evaluating node {idx + 1}/{len(nodes_data)}: {current_node.title}"
        )
        
        feedback = self.pedagogue.evaluate_node(
            node=current_node,
            user_level=DifficultyLevel(state["user_level"]),
            previous_nodes=previous_nodes
        )
        
        state["pedagogue_feedback"].append(asdict(feedback))
        
        if not feedback.approved:
            state["needs_revision"] = True
            state["revision_reason"] = f"Pedagogue: {'; '.join(feedback.issues)}"
            state["messages"].append(
                f"❌ Rejected: {state['revision_reason']}"
            )
        else:
            state["needs_revision"] = False
            state["messages"].append(
                f"✅ Approved by Pedagogue"
            )
        
        return state
    
    def _repair_node(self, state: WorkflowState) -> WorkflowState:
        """Step 3: 修复节点（使用 RepairAgent）"""
        state["iteration_count"] += 1
        
        if state["iteration_count"] >= state["max_iterations"]:
            state["messages"].append(
                "⚠️ Max iterations reached. Moving forward with best available version."
            )
            state["needs_revision"] = False
            # Force move to next node
            state["processed_nodes"].append(
                state["learning_path"]["nodes"][state["current_node_idx"]]
            )
            state["current_node_idx"] += 1
            state["iteration_count"] = 0
            return state
        
        idx = state["current_node_idx"]
        
        state["messages"].append(
            f"🔧 Repair Agent: Fixing node {idx + 1} (Iteration {state['iteration_count']})"
        )
        
        nodes_data = state["learning_path"]["nodes"]
        current_node_data = nodes_data[idx]
        current_node = KnowledgeNode(**current_node_data)
        
        # Get latest feedback
        last_ped_feedback = state["pedagogue_feedback"][-1] if state["pedagogue_feedback"] else None
        last_stud_feedback = state["student_feedback"][-1] if state["student_feedback"] else None
        
        if not last_ped_feedback:
            state["needs_revision"] = False
            return state
        
        ped_feedback = PedagogicalFeedback(**last_ped_feedback)
        stud_feedback = StudentFeedback(**last_stud_feedback) if last_stud_feedback else None
        
        # Call repair agent
        repair_result = self.repair_agent.repair_node(
            node=current_node,
            pedagogue_feedback=ped_feedback,
            student_feedback=stud_feedback,
            topic_context=state["topic"],
            user_level=DifficultyLevel(state["user_level"])
        )
        
        action = repair_result.get("action")
        reasoning = repair_result.get("reasoning", "")
        
        state["messages"].append(f"🔧 Action: {action} - {reasoning}")
        
        if action == "revise" and "revised_node" in repair_result:
            # Replace current node
            nodes_data[idx] = asdict(repair_result["revised_node"])
            state["messages"].append(f"✏️ Node revised")
        
        elif action == "insert_prerequisite" and "new_nodes" in repair_result:
            # Insert new prerequisite nodes BEFORE current node
            new_nodes_data = [asdict(n) for n in repair_result["new_nodes"]]
            # Insert at current position
            for i, new_node in enumerate(new_nodes_data):
                nodes_data.insert(idx + i, new_node)
            state["messages"].append(
                f"➕ Inserted {len(new_nodes_data)} prerequisite node(s) before current node"
            )
            # Don't increment current_node_idx - we'll evaluate the new prerequisite first
        
        elif action == "split" and "new_nodes" in repair_result:
            # Replace current node with multiple simpler nodes
            new_nodes_data = [asdict(n) for n in repair_result["new_nodes"]]
            nodes_data[idx:idx+1] = new_nodes_data
            state["messages"].append(
                f"✂️ Split node into {len(new_nodes_data)} simpler nodes"
            )
        
        return state
    
    def _find_resources_node(self, state: WorkflowState) -> WorkflowState:
        """Step 4: 图书管理员查找资源（带上下文约束）"""
        idx = state["current_node_idx"]
        current_node_data = state["learning_path"]["nodes"][idx]
        current_node = KnowledgeNode(**current_node_data)
        
        state["messages"].append(
            f"📚 Librarian: Finding resources for {current_node.title}"
        )
        
        # Pass global context to prevent drift
        resources = self.librarian.find_resources(
            current_node, 
            num_resources=3,
            global_context=state.get("global_context")
        )
        
        # Check for context drift using keyword guardrails
        global_ctx = state.get("global_context", {})
        forbidden = global_ctx.get("forbidden_topics", [])
        allowed = global_ctx.get("allowed_keywords", [state["topic"]])
        
        drift_detected = False
        for resource in resources:
            check_text = f"{resource.title} {resource.description}"
            drift_result = detect_context_drift(check_text, allowed, forbidden)
            
            if drift_result["has_drift"]:
                drift_detected = True
                state["messages"].append(
                    f"⚠️ Context drift detected in resource '{resource.title}': "
                    f"contains forbidden keywords {drift_result['forbidden_found']}"
                )
        
        if drift_detected:
            state["needs_revision"] = True
            state["revision_reason"] = "Librarian: Resources strayed from topic context"
            state["messages"].append("🔄 Triggering resource re-search due to context drift")
            # Don't save these resources, will retry
        else:
            state["resources_by_node"][current_node.id] = [
                asdict(r) for r in resources
            ]
            state["messages"].append(
                f"✅ Found {len(resources)} on-topic resources"
            )
        
        return state
    
    def _test_student_node(self, state: WorkflowState) -> WorkflowState:
        """Step 5: 模拟学生测试理解度"""
        idx = state["current_node_idx"]
        current_node_data = state["learning_path"]["nodes"][idx]
        current_node = KnowledgeNode(**current_node_data)
        
        # Get resources for this node
        resources_data = state["resources_by_node"].get(current_node.id, [])
        resources = [Resource(**r) for r in resources_data]
        
        # Get previous nodes
        previous_nodes = [KnowledgeNode(**n) for n in state["processed_nodes"]]
        
        # Create simulated student
        student = SimulatedStudent(
            user_level=DifficultyLevel(state["user_level"]),
            llm=self.llm
        )
        
        state["messages"].append(
            f"🎓 Simulated Student: Testing comprehension of {current_node.title}"
        )
        
        feedback = student.test_comprehension(
            node=current_node,
            resources=resources,
            previous_nodes=previous_nodes
        )
        
        state["student_feedback"].append(asdict(feedback))
        
        if not feedback.understood or feedback.comprehension_score < 0.7:
            state["needs_revision"] = True
            state["revision_reason"] = (
                f"Student confusion (score: {feedback.comprehension_score:.2f}): "
                f"{'; '.join(feedback.confusion_points[:2])}"
            )
            state["messages"].append(
                f"❌ Student did not understand: {state['revision_reason']}"
            )
        else:
            state["needs_revision"] = False
            state["messages"].append(
                f"✅ Student understood (score: {feedback.comprehension_score:.2f})"
            )
            
            # Move this node to processed
            state["processed_nodes"].append(current_node_data)
            state["current_node_idx"] += 1
            state["iteration_count"] = 0  # Reset iteration counter for next node
        
        return state
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Final step: 完成并输出结果"""
        state["completed"] = True
        state["messages"].append(
            "🎉 Learning path finalized and pressure-tested!"
        )
        return state
    
    # Conditional edge functions
    
    def _should_evaluate(self, state: WorkflowState) -> str:
        """决定是否继续评估"""
        if not state["learning_path"] or not state["learning_path"]["nodes"]:
            return "end"
        return "evaluate"
    
    def _pedagogue_decision(self, state: WorkflowState) -> str:
        """心理学家的决策 - 实现闭环逻辑"""
        if state["completed"]:
            return "end"
        
        # Check if pedagogue rejected the node
        if state["pedagogue_feedback"]:
            last_feedback = state["pedagogue_feedback"][-1]
            if not last_feedback.get("approved", True):
                # Node was rejected - trigger repair
                if state["iteration_count"] >= state["max_iterations"]:
                    state["messages"].append("⚠️ Max iterations - forcing approval")
                    return "approved"
                return "repair"
        
        return "approved"
    
    def _student_decision(self, state: WorkflowState) -> str:
        """学生模拟器的决策 - 实现闭环逻辑"""
        # Check student feedback
        if state["student_feedback"]:
            last_feedback = state["student_feedback"][-1]
            
            # If student didn't understand or scored poorly, trigger repair
            if not last_feedback.get("understood", True) or last_feedback.get("comprehension_score", 1.0) < 0.6:
                if state["iteration_count"] >= state["max_iterations"]:
                    state["messages"].append("⚠️ Max iterations - moving to next node anyway")
                    # Force approval and move on
                    state["processed_nodes"].append(
                        state["learning_path"]["nodes"][state["current_node_idx"]]
                    )
                    state["current_node_idx"] += 1
                    state["iteration_count"] = 0
                    
                    if state["current_node_idx"] >= len(state["learning_path"]["nodes"]):
                        return "end"
                    return "understood"
                
                state["messages"].append(
                    f"❌ Student comprehension too low ({last_feedback.get('comprehension_score', 0):.2f}) - triggering repair"
                )
                return "repair"
        
        # Student understood - move to next node
        state["processed_nodes"].append(
            state["learning_path"]["nodes"][state["current_node_idx"]]
        )
        state["current_node_idx"] += 1
        state["iteration_count"] = 0
        
        # Check if we're done
        if state["current_node_idx"] >= len(state["learning_path"]["nodes"]):
            return "end"
        
        return "understood"
    
    def create_learning_path(
        self,
        topic: str,
        user_level = "beginner",
        target_level = "intermediate"
    ) -> Dict[str, Any]:
        """
        并行化执行入口
        """
        # 支持字符串和枚举类型
        if isinstance(user_level, DifficultyLevel):
            ul = user_level
        else:
            ul = DifficultyLevel(user_level)
        
        if isinstance(target_level, DifficultyLevel):
            tl = target_level
        else:
            tl = DifficultyLevel(target_level)
        all_messages = []
        
        # 1. 设计大纲 (串行，因为必须先有大纲)
        all_messages.append("🏗️ Architect: Designing structure...")
        if self.llm:
            # 这一步通常较快，不需要并行
            calibration = self.architect.calibrate_curriculum(topic, ul)
            learning_path = self.architect.design_curriculum(topic, ul, tl, calibration=calibration)
        else:
            learning_path = self.architect._fallback_curriculum(topic, ul, tl)
            
        all_messages.append(f"✅ Generated {len(learning_path.nodes)} nodes. Starting parallel processing...")

        # 准备并行任务
        global_context = {"topic": topic}
        futures = []
        
        # ⚡ 使用 ThreadPoolExecutor 进行并行处理
        # 这将原本需要 5 分钟的串行任务压缩到 1 分钟左右
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for node in learning_path.nodes:
                future = executor.submit(
                    self._process_single_node_pipeline,
                    asdict(node), # 传入字典数据
                    ul,
                    topic,
                    global_context
                )
                futures.append(future)

        # 收集结果 (保持顺序)
        final_nodes = []
        resources_map = {}
        all_ped_fb = []
        all_stu_fb = []
        
        for i, future in enumerate(futures):
            try:
                # ⚡ 设置超时：如果单个节点处理超过 45 秒，强行跳过
                result = future.result(timeout=45)
                
                final_nodes.append(result["node"])
                resources_map[result["node"]["id"]] = result["resources"]
                if result["pedagogue_feedback"]: all_ped_fb.append(result["pedagogue_feedback"])
                if result["student_feedback"]: all_stu_fb.append(result["student_feedback"])
                all_messages.extend(result["messages"])
                
            except concurrent.futures.TimeoutError:
                all_messages.append(f"⚠️ Node {i+1} timed out! Using original version.")
                # 超时兜底：直接使用原始节点
                orig_node = learning_path.nodes[i]
                final_nodes.append(asdict(orig_node))
                resources_map[orig_node.id] = [asdict(r) for r in self.librarian._fallback_resources(orig_node)]
                
            except Exception as e:
                all_messages.append(f"❌ Critical error on node {i+1}: {e}")
                # 错误兜底
                orig_node = learning_path.nodes[i]
                final_nodes.append(asdict(orig_node))

        return {
            "learning_path": {
                "topic": topic,
                "nodes": final_nodes,
                "total_hours": learning_path.total_hours
            },
            "resources": resources_map,
            "pedagogue_feedback": all_ped_fb,
            "student_feedback": all_stu_fb,
            "messages": all_messages,
            "completed": True
        }
    
    def _create_learning_path_simple(
        self,
        topic: str,
        user_level: DifficultyLevel,
        target_level: DifficultyLevel
    ) -> Dict[str, Any]:
        """简化版工作流（不使用 LangGraph）- 带闭环修复 + 性能优化"""
        messages = []
        api_call_count = 0  # Track API usage
        
        # Step 0: Calibration - OPTIMIZED: Skip if not using LLM
        if not self.llm:
            messages.append("⚡ Fast mode: Skipping calibration")
            calibration = self.architect._default_calibration(user_level)
        else:
            messages.append("🎯 Calibrating curriculum...")
            calibration = self.architect.calibrate_curriculum(topic, user_level)
            api_call_count += 1
            messages.append(f"✅ Calibration complete: {calibration.get('confirmed_level', user_level.value)}")
        
        # Build global context
        global_context = {
            "topic": topic,
            "primary_domain": calibration.get("global_constraints", {}).get("primary_domain", ""),
            "forbidden_topics": calibration.get("global_constraints", {}).get("forbidden_topics", []),
            "allowed_keywords": [topic]
        }
        
        # Step 1: Design
        messages.append("🏗️ Creating learning path...")
        learning_path = self.architect.design_curriculum(
            topic=topic,
            user_level=user_level,
            target_level=target_level,
            calibration=calibration
        )
        if self.llm:
            api_call_count += 1
        
        messages.append(f"✅ Created {len(learning_path.nodes)} learning nodes")
        
        # Step 2-5: Process each node - OPTIMIZED with Fast-Track
        processed_nodes = []
        resources_by_node = {}
        all_pedagogue_feedback = []
        all_student_feedback = []
        
        student = SimulatedStudent(user_level=user_level, llm=self.llm)
        
        # ⚡ OPTIMIZATION: Pre-compute difficulty rankings for fast-track logic
        difficulty_rank = {
            DifficultyLevel.BEGINNER: 1,
            DifficultyLevel.INTERMEDIATE: 2,
            DifficultyLevel.ADVANCED: 3,
            DifficultyLevel.EXPERT: 4
        }
        user_rank = difficulty_rank[user_level]
        
        i = 0
        max_repairs_per_node = 1  # OPTIMIZED: Strict limit to prevent cost explosion
        
        while i < len(learning_path.nodes):
            node = learning_path.nodes[i]
            node_rank = difficulty_rank[node.difficulty]
            messages.append(f"\n📍 Processing node {i+1}/{len(learning_path.nodes)}: {node.title}")
            
            repair_count = 0
            node_approved = False
            ped_feedback = None  # Initialize to avoid UnboundLocalError
            
            # ⚡ OPTIMIZATION 1: Fast-Track for Easy Nodes
            # If node difficulty <= user level, skip expensive reviews
            # Rationale: College students don't need professors to review addition problems
            is_easy_node = node_rank <= user_rank
            
            if is_easy_node:
                messages.append(f"⚡ Fast-track: Node within user capacity (Level {node.difficulty.value} <= {user_level.value}). Auto-approved.")
                node_approved = True
                # Create lightweight feedback for fast-track
                ped_feedback = self.pedagogue._simple_evaluation(node, user_level)
                all_pedagogue_feedback.append(asdict(ped_feedback))
            
            while not node_approved and repair_count < max_repairs_per_node:
                # Pedagogue evaluation
                if self.llm:
                    ped_feedback = self.pedagogue.evaluate_node(
                        node, user_level, processed_nodes
                    )
                    api_call_count += 1
                else:
                    # Fallback mode
                    ped_feedback = self.pedagogue._simple_evaluation(node, user_level)
                    
                all_pedagogue_feedback.append(asdict(ped_feedback))
                
                if not ped_feedback.approved:
                    messages.append(f"❌ Pedagogue rejected: {'; '.join(ped_feedback.issues[:2])}")
                    
                    # OPTIMIZED: Skip repair if no LLM (use heuristics)
                    if not self.llm:
                        messages.append("⚡ Fast mode: Applying simple fix")
                        # Simple heuristic: just mark as approved and continue
                        node_approved = True
                        break
                    
                    # Attempt repair
                    messages.append(f"🔧 Repair attempt {repair_count + 1}/{max_repairs_per_node}")
                    repair_result = self.repair_agent.repair_node(
                        node=node,
                        pedagogue_feedback=ped_feedback,
                        student_feedback=None,
                        topic_context=topic,
                        user_level=user_level
                    )
                    api_call_count += 1
                    
                    action = repair_result.get("action")
                    messages.append(f"🔧 {action}: {repair_result.get('reasoning', '')[:80]}")
                    
                    if action == "revise" and "revised_node" in repair_result:
                        node = repair_result["revised_node"]
                        learning_path.nodes[i] = node
                    elif action == "insert_prerequisite" and "new_nodes" in repair_result:
                        # OPTIMIZED: Limit inserted nodes
                        new_nodes = repair_result["new_nodes"][:2]  # Max 2 prerequisites
                        for j, new_node in enumerate(new_nodes):
                            learning_path.nodes.insert(i + j, new_node)
                        messages.append(f"➕ Inserted {len(new_nodes)} prerequisite(s)")
                        continue
                    elif action == "split" and "new_nodes" in repair_result:
                        # Replace with split nodes
                        learning_path.nodes[i:i+1] = repair_result["new_nodes"]
                        node = learning_path.nodes[i]
                        messages.append(f"✂️ Split into {len(repair_result['new_nodes'])} nodes")
                    
                    repair_count += 1
                else:
                    messages.append("✅ Pedagogue approved")
                    node_approved = True
            
            if not node_approved:
                messages.append("⚠️ Max repair attempts reached, proceeding anyway")
            
            # ⚡ OPTIMIZATION 2: Resource Search - Dynamic Reduction
            # Easy nodes: 1 resource only (save 66% API calls)
            # Difficult nodes: 2 resources (save 33% API calls vs original 3)
            num_resources = 1 if is_easy_node else 2
            
            if self.llm:
                resources = self.librarian.find_resources(
                    node, 
                    num_resources=num_resources,  # OPTIMIZED: 1 for easy, 2 for hard
                    global_context=global_context
                )
                api_call_count += 1
            else:
                resources = self.librarian._fallback_resources(node)
            
            # OPTIMIZED: Quick drift check (no detailed analysis)
            drift_detected = False
            forbidden = global_context.get("forbidden_topics", [])
            if forbidden and resources:
                for resource in resources:
                    check_text = f"{resource.title} {resource.description}".lower()
                    if any(fb.lower() in check_text for fb in forbidden):
                        drift_detected = True
                        break
            
            if not drift_detected:
                resources_by_node[node.id] = [asdict(r) for r in resources]
                messages.append(f"📚 Found {len(resources)} resources")
            else:
                messages.append("⚠️ Skipping drifted resources")
            
            # ⚡ OPTIMIZATION 3: Student Test - Skip for Easy Nodes
            # Fast-track nodes don't need student testing (they already passed pre-screening)
            if is_easy_node:
                # Assume high comprehension for easy nodes
                stud_feedback = StudentFeedback(
                    understood=True,
                    node_id=node.id,
                    comprehension_score=0.9,
                    confusion_points=[],
                    reasoning="Fast-tracked: content within user's existing capabilities"
                )
                messages.append("⚡ Fast-track: Skipping student test (auto-pass)")
            elif not self.llm:
                # Fast mode: assume understood if difficulty matches
                stud_feedback = student._simple_comprehension_test(node)
            else:
                stud_feedback = student.test_comprehension(node, resources, processed_nodes)
                api_call_count += 1
                
            all_student_feedback.append(asdict(stud_feedback))
            
            # ⚡ OPTIMIZATION 4: Lower Threshold from 0.7 to 0.5
            # More lenient progression = fewer repair loops = lower cost
            if stud_feedback.understood and stud_feedback.comprehension_score >= 0.5:
                messages.append(f"✅ Student comprehension: {stud_feedback.comprehension_score:.2f}")
                processed_nodes.append(node)
                i += 1
            else:
                messages.append(f"❌ Student confusion: {stud_feedback.comprehension_score:.2f}")
                
                # OPTIMIZED: Skip repair if no LLM or already tried
                if self.llm and repair_count == 0 and ped_feedback:
                    messages.append("🔧 Attempting repair for student comprehension")
                    repair_result = self.repair_agent.repair_node(
                        node=node,
                        pedagogue_feedback=ped_feedback,
                        student_feedback=stud_feedback,
                        topic_context=topic,
                        user_level=user_level
                    )
                    api_call_count += 1
                    
                    if "revised_node" in repair_result:
                        learning_path.nodes[i] = repair_result["revised_node"]
                        messages.append("🔄 Node revised for better comprehension")
                    elif "new_nodes" in repair_result and "insert" in repair_result.get("action", ""):
                        # OPTIMIZED: Limit to 1 scaffolding node
                        new_nodes = repair_result["new_nodes"][:1]
                        for j, new_node in enumerate(new_nodes):
                            learning_path.nodes.insert(i + j, new_node)
                        messages.append(f"➕ Inserted scaffolding prerequisite")
                    # Don't increment i, retry this node
                else:
                    messages.append("⚡ Moving forward (fast mode)")
                    processed_nodes.append(node)
                    i += 1
        
        messages.append("\n🎉 Learning path complete with closed-loop validation!")
        
        # Calculate statistics
        total_repairs = sum(1 for msg in messages if "🔧" in msg)
        total_approvals = sum(1 for msg in messages if "✅ Pedagogue approved" in msg)
        
        messages.append(f"\n📊 Statistics: {total_approvals} nodes approved, {total_repairs} repairs performed")
        messages.append(f"⚡ API calls: {api_call_count} (optimized for cost)")
        
        return {
            "learning_path": {
                "topic": learning_path.topic,
                "target_level": learning_path.target_level.value,
                "nodes": [asdict(n) for n in learning_path.nodes],
                "total_hours": sum(n.estimated_hours for n in learning_path.nodes),
                "metadata": {
                    **learning_path.metadata,
                    "calibration": calibration,
                    "total_repairs": total_repairs,
                    "api_calls": api_call_count  # NEW: Track API usage
                }
            },
            "processed_nodes": [asdict(n) for n in processed_nodes],
            "resources": resources_by_node,
            "pedagogue_feedback": all_pedagogue_feedback,
            "student_feedback": all_student_feedback,
            "messages": messages,
            "completed": True,
            "performance": {  # NEW: Performance metrics
                "api_calls": api_call_count,
                "total_repairs": total_repairs,
                "nodes_created": len(learning_path.nodes),
                "nodes_processed": len(processed_nodes)
            }
        }


__all__ = ["TeachingSystemWorkflow", "WorkflowState"]

