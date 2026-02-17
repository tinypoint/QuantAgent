"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT.
    The agent uses LLM tool-calling to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]

        # --- System prompt for LLM ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名高频交易(HFT)技术指标分析助手，工作在时间敏感场景。"
                    "你需要通过技术指标分析支持快速交易执行。\n\n"
                    "你可调用以下工具: compute_rsi、compute_macd、compute_roc、compute_stoch、compute_willr。"
                    "调用时请传入合适参数，例如 `kline_data` 与对应周期参数。\n\n"
                    f"⚠️ 当前OHLC数据周期为 {time_frame}，代表近期市场行为。"
                    "请快速且准确地解释指标信号。\n\n"
                    "以下是OHLC数据:\n{kline_data}。\n\n"
                    "请按需调用工具并完成分析。",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(state["kline_data"], ensure_ascii=False, indent=2))

        chain = prompt | llm.bind_tools(tools)
        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="开始技术指标分析。")]

        # --- Step 1: Ask for tool calls ---
        ai_response = chain.invoke(messages)
        messages.append(ai_response)

        # --- Step 2: Collect tool results ---
        if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
            for call in ai_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result, ensure_ascii=False)
                    )
                )

        # --- Step 3: Re-run with tool results until text answer ---
        max_iterations = 5
        iteration = 0
        final_response = None
        while iteration < max_iterations:
            iteration += 1
            final_response = chain.invoke(messages)
            messages.append(final_response)

            if not hasattr(final_response, "tool_calls") or not final_response.tool_calls:
                break

            for call in final_response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]
                tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                tool_fn = next(t for t in tools if t.name == tool_name)
                tool_result = tool_fn.invoke(tool_args)
                messages.append(
                    ToolMessage(
                        tool_call_id=call["id"], content=json.dumps(tool_result, ensure_ascii=False)
                    )
                )

        # Extract content with fallback
        if final_response:
            report_content = final_response.content
            if not report_content or (
                isinstance(report_content, str) and not report_content.strip()
            ):
                for msg in reversed(messages):
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and isinstance(msg.content, str)
                        and msg.content.strip()
                        and not hasattr(msg, "tool_calls")
                    ):
                        report_content = msg.content
                        break
        else:
            report_content = "技术指标分析完成，但未生成详细报告。"

        return {
            "messages": messages,
            "indicator_report": report_content if report_content else "技术指标分析完成。",
        }

    return indicator_agent_node
