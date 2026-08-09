"""LangGraph 多智能体工作流编排：意图识别 → 条件路由 → 业务 Agent。"""

from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

from agents.content_agent import content_generator
from agents.graph_agent import graph_query_handler
from agents.intent_agent import identify_intent
from agents.publish_agent import publish_handler
from agents.qa_agent import qa_handler


class AgentState(MessagesState):
    """多 Agent 共享状态：消息流 + 意图 + 各 Agent 产出。"""

    intent: str
    answer: str
    sources: list
    cypher: str
    graph_data: list
    generated_content: str
    topic: str
    publish_result: dict


def _route_by_intent(state: AgentState) -> str:
    """条件路由：依据意图分发到对应业务 Agent。"""
    intent = state.get("intent", "qa")
    return {
        "qa": "qa_agent",
        "graph_query": "graph_agent",
        "content_generation": "content_agent",
        "publish": "publish_agent",
    }.get(intent, "qa_agent")


def build_agent_graph():
    """构建并编译多智能体协同图。"""
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_agent", identify_intent)
    workflow.add_node("qa_agent", qa_handler)
    workflow.add_node("graph_agent", graph_query_handler)
    workflow.add_node("content_agent", content_generator)
    workflow.add_node("publish_agent", publish_handler)

    workflow.set_entry_point("intent_agent")
    workflow.add_conditional_edges("intent_agent", _route_by_intent, {
        "qa_agent": "qa_agent",
        "graph_agent": "graph_agent",
        "content_agent": "content_agent",
        "publish_agent": "publish_agent",
    })
    for node in ("qa_agent", "graph_agent", "content_agent", "publish_agent"):
        workflow.add_edge(node, END)

    return workflow.compile()


_app = None


def run_agent(question: str) -> dict:
    """运行一次完整 Agent 流程，返回最终状态。"""
    global _app
    if _app is None:
        _app = build_agent_graph()
    return _app.invoke({"messages": [HumanMessage(content=question)]})
