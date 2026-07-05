from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from simulator.agents_graphs.langgraph_tool import AgentTools, ToolNode, should_continue


def test_should_continue_routes_to_tools_when_tool_calls_exist():
    assert should_continue(
        {
            "messages": [
                AIMessage(content="use tool", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
            ]
        }
    ) == "tools"


def test_should_continue_routes_to_end_when_no_tool_calls():
    assert should_continue({"messages": [AIMessage(content="final answer")]}) == END


def test_tool_node_merges_state_args_and_returns_tool_message():
    def lookup(query, session):
        return f"{query}:{session}"

    node = ToolNode([SimpleNamespace(name="lookup", func=lookup)])
    state = {
        "messages": [
            HumanMessage(content="start"),
            AIMessage(
                content="call tool",
                tool_calls=[{"name": "lookup", "args": {"query": "abc"}, "id": "1"}],
            ),
        ],
        "args": {"session": "s1"},
    }

    result = node._func(state)

    assert result["args"] == {"session": "s1"}
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result["messages"][0].content == "abc:s1"
    assert result["messages"][0].tool_call_id == "1"


def test_agent_tools_get_call_model_wraps_llm_response():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="hello there")
    agent = AgentTools(llm=llm, tools=[])

    result = agent.get_call_model()({"messages": [HumanMessage(content="hi")], "args": {}})

    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "hello there"
    assert result["messages"][0].type == "ai"
    llm.invoke.assert_called_once()
    assert len(llm.invoke.call_args.args) == 1
    assert llm.invoke.call_args.args[0][0].content == "hi"
    assert llm.invoke.call_args.args[0][0].type == "human"


def test_agent_tools_init_with_no_tools_builds_graph():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="final")
    agent = AgentTools(llm=llm, tools=[])

    assert agent.tools == []
    assert agent.graph is not None
