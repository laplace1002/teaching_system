"""
多智能体教学系统 - 演示程序
Multi-Agent Teaching System - Demo

本演示展示系统的核心功能：
1. 🔧 RepairAgent - 自动修复被拒绝的节点
2. 🎯 Curriculum Calibration - 校准用户水平
3. 🛡️ Global Context Constraints - 防止上下文漂移
4. 📊 Student Profile Update - 动态跟踪学习进度
5. 🔍 Keyword Guardrails - 检测资源是否跑偏
6. ➕ Scaffolding - 自动插入前置节点
7. 🔄 Closed-Loop Logic - 循环修复直到通过
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from system import TeachingSystem, DeepSeekConfig


def print_section(title: str):
    """打印分隔符"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_basic_system():
    """基础示例：展示自动修复机制"""
    print_section("Demo 1: Automatic Repair Mechanism")
    
    # Initialize system
    system = TeachingSystem(max_revision_iterations=3)
    
    # Create learning path - intentionally challenging for beginners
    print("\n📚 Creating learning path for: 'High Performance Computing' (beginner level)")
    print("   (This is intentionally challenging to trigger the repair mechanism)\n")
    
    result = system.create_learning_path(
        topic="High Performance Computing",
        user_level="beginner",
        target_level="intermediate"
    )
    
    # Print execution flow
    print_section("Execution Flow (Messages)")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    # Print calibration results
    if "calibration" in result["learning_path"]["metadata"]:
        print_section("Curriculum Calibration Results")
        calibration = result["learning_path"]["metadata"]["calibration"]
        print(f"  Confirmed Level: {calibration.get('confirmed_level')}")
        print(f"  Needs Fundamentals: {calibration.get('needs_fundamentals')}")
        print(f"  Missing Skills: {', '.join(calibration.get('missing_skills', []))}")
        print(f"  Primary Domain: {calibration.get('global_constraints', {}).get('primary_domain', 'N/A')}")
        print(f"  Forbidden Topics: {', '.join(calibration.get('global_constraints', {}).get('forbidden_topics', []))}")
    
    # Print repair statistics
    total_repairs = result["learning_path"]["metadata"].get("total_repairs", 0)
    print_section("Repair Statistics")
    print(f"  Total Repairs Performed: {total_repairs}")
    print(f"  Total Nodes: {len(result['learning_path']['nodes'])}")
    print(f"  Nodes Processed: {len(result['processed_nodes'])}")
    
    # Show nodes that were rejected and repaired
    print_section("Pedagogue Feedback Summary")
    for i, feedback in enumerate(result["pedagogue_feedback"]):
        node_id = feedback["node_id"]
        approved = feedback["approved"]
        status = "✅ APPROVED" if approved else "❌ REJECTED"
        
        print(f"\n  Node {i+1} ({node_id}): {status}")
        if not approved:
            print(f"    Issues: {'; '.join(feedback['issues'][:3])}")
            if feedback.get("suggestions"):
                print(f"    Suggestions: {'; '.join(feedback['suggestions'][:2])}")
    
    # Show student comprehension scores
    print_section("Student Comprehension Summary")
    for i, feedback in enumerate(result["student_feedback"]):
        node_id = feedback["node_id"]
        score = feedback["comprehension_score"]
        understood = feedback["understood"]
        
        status = "✅" if understood and score >= 0.6 else "❌"
        print(f"  {status} Node {i+1} ({node_id}): Score {score:.2f}")
        
        if not understood or score < 0.6:
            print(f"      Confusion: {'; '.join(feedback['confusion_points'][:2])}")
            if feedback.get("missing_prerequisites"):
                print(f"      Missing: {'; '.join(feedback['missing_prerequisites'][:2])}")
    
    return result


def demo_context_drift_detection():
    """展示上下文漂移检测"""
    print_section("Demo 2: Context Drift Detection")
    
    from agents import detect_context_drift
    
    # Simulate HPC-related content (good)
    good_content = "Understanding MPI message passing in distributed computing with C++ and parallel algorithms"
    
    # Simulate drifted content (bad - mentions Node.js)
    bad_content = "Performance optimization with Node.js event loop and JavaScript async programming"
    
    allowed_keywords = ["HPC", "High Performance Computing", "MPI", "C++", "supercomputer", "parallel"]
    forbidden_keywords = ["Node.js", "JavaScript", "web development", "browser", "HTML"]
    
    print("\n  Testing GOOD content:")
    print(f"    '{good_content}'")
    result_good = detect_context_drift(good_content, allowed_keywords, forbidden_keywords)
    print(f"    ✅ Has Drift: {result_good['has_drift']}")
    print(f"    ✅ Confidence: {result_good['confidence']:.2f}")
    
    print("\n  Testing BAD content (with drift):")
    print(f"    '{bad_content}'")
    result_bad = detect_context_drift(bad_content, allowed_keywords, forbidden_keywords)
    print(f"    ⚠️ Has Drift: {result_bad['has_drift']}")
    print(f"    ⚠️ Forbidden Found: {result_bad['forbidden_found']}")
    print(f"    ⚠️ Confidence: {result_bad['confidence']:.2f}")


def demo_student_profile_update():
    """展示学生知识档案动态更新"""
    print_section("Demo 3: Student Profile Update")
    
    from agents import SimulatedStudent, KnowledgeNode, DifficultyLevel, Resource
    
    student = SimulatedStudent(user_level=DifficultyLevel.BEGINNER, llm=None)
    
    # Create mock nodes
    node1 = KnowledgeNode(
        id="node_1",
        title="Introduction to Programming",
        description="Basic programming concepts",
        difficulty=DifficultyLevel.BEGINNER,
        estimated_hours=2.0,
        key_concepts=["variables", "loops"]
    )
    
    node2 = KnowledgeNode(
        id="node_2",
        title="Functions and Modules",
        description="Advanced programming concepts",
        difficulty=DifficultyLevel.INTERMEDIATE,
        estimated_hours=3.0,
        key_concepts=["functions", "modules"]
    )
    
    print(f"\n  Initial student knowledge profile: {student.knowledge_profile}")
    
    # Test node 1
    feedback1 = student.test_comprehension(node1, [], [])
    print(f"\n  After testing Node 1:")
    print(f"    - Understood: {feedback1.understood}")
    print(f"    - Score: {feedback1.comprehension_score:.2f}")
    print(f"    - Updated profile: {student.knowledge_profile}")
    
    # Test node 2 (with node 1 as previous knowledge)
    feedback2 = student.test_comprehension(node2, [], [node1])
    print(f"\n  After testing Node 2 (with Node 1 knowledge):")
    print(f"    - Understood: {feedback2.understood}")
    print(f"    - Score: {feedback2.comprehension_score:.2f}")
    print(f"    - Updated profile: {student.knowledge_profile}")
    
    print("\n  📝 Note: In real scenarios with LLM, the student would leverage")
    print("      its accumulated knowledge when evaluating new nodes.")


def demo_system_features():
    """展示系统的核心特性"""
    print_section("Demo 4: System Core Features")
    
    print("\n  🎯 系统工作流程:")
    print("     1. 📊 Calibration: 确认用户水平和前置要求")
    print("     2. 🏗️ Designer: 基于校准结果创建课程")
    print("     3. 🧠 Pedagogue: 评估难度")
    print("        → 如果拒绝 → 🔧 RepairAgent 修复 → 重新评估")
    print("     4. 📚 Librarian: 查找资源（带全局约束）")
    print("        → 🔍 关键词守护检测漂移 → 必要时重试")
    print("     5. 🎓 Student: 测试理解度")
    print("        → 如果困惑 (< 0.6) → 🔧 修复/插入前置 → 重新测试")
    print("        → 📊 更新学习档案")
    print("     6. 🔄 循环直到通过或达到最大迭代次数")
    
    print("\n  ✨ 核心特性:")
    features = [
        ("🔧 RepairAgent", "主动修复被拒绝的节点（修改/拆分/插入前置）"),
        ("🎯 Calibration", "校准用户真实水平，明确需要的基础"),
        ("🛡️ Context Constraints", "防止资源跑偏到无关主题"),
        ("🔍 Keyword Guardrails", "检测并阻止上下文漂移"),
        ("📊 Student Profile", "动态跟踪已掌握的知识"),
        ("➕ Scaffolding", "自动插入缺失的前置节点"),
        ("🔄 Closed-Loop", "循环验证直到质量达标")
    ]
    
    for feature, description in features:
        print(f"     {feature}: {description}")
    
    print("\n  💡 结果: 真正的质量保证，而不只是内容推荐")


def main():
    """运行所有演示"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█    🎓 多智能体教学系统 - 演示程序 🎓    ".center(68) + "█")
    print("█    Multi-Agent Teaching System - Demo    ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        # Demo 1: 基础系统演示
        result = demo_basic_system()
        
        # Demo 2: 上下文漂移检测
        demo_context_drift_detection()
        
        # Demo 3: 学生档案更新
        demo_student_profile_update()
        
        # Demo 4: 系统特性展示
        demo_system_features()
        
        print_section("✅ 所有演示完成!")
        print("\n  系统核心能力:")
        print("  ✅ 自动修复被拒绝的节点")
        print("  ✅ 检测并防止上下文漂移")
        print("  ✅ 根据学生理解度触发修正动作")
        print("  ✅ 自动插入缺失的前置知识")
        print("  ✅ 动态跟踪学习进度")
        print("\n  智能体不再只是抱怨问题，而是协作解决问题！🚀\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

