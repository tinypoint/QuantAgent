"""
Agent for trend analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to generate and interpret trendline charts for short-term prediction.
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError


# --- Retry wrapper for LLM invocation ---
def invoke_with_retry(call_fn, *args, retries=3, wait_sec=4):
    """
    Retry a function call with exponential backoff for rate limits or errors.
    """
    for attempt in range(retries):
        try:
            result = call_fn(*args)
            return result
        except RateLimitError:
            print(
                f"触发限流，{wait_sec}秒后重试 (attempt {attempt + 1}/{retries})..."
            )
        except Exception as e:
            print(
                f"调用异常: {e}，{wait_sec}秒后重试 (attempt {attempt + 1}/{retries})..."
            )
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("超过最大重试次数")


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node for HFT. The agent uses precomputed images from state or falls back to tool generation.
    """

    def trend_agent_node(state):
        tools = [toolkit.generate_trend_image]
        time_frame = state["time_frame"]

        trend_image_b64 = state.get("trend_image")

        messages = []

        if not trend_image_b64:
            print("state中没有预生成趋势图，开始使用工具生成...")

            system_prompt = (
                "你是高频交易场景下的K线趋势识别助手。"
                "你必须先调用 `generate_trend_image` 工具，并传入 `kline_data`。"
                "图像生成后，再分析支撑/阻力趋势线以及可识别K线结构。"
                "最后给出短期趋势判断: 上涨、下跌或震荡。"
                "在图像生成并分析完成前，不得提前给出预测。"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"以下是近期K线数据:\n{json.dumps(state['kline_data'], indent=2)}"
                ),
            ]

            chain = tool_llm.bind_tools(tools)

            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    import copy

                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = tool_fn.invoke(tool_args)
                    trend_image_b64 = tool_result.get("trend_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("使用state中预生成的趋势图")

        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"这是一张 {time_frame} 周期K线图，图中已自动叠加趋势线: 蓝线为支撑线，红线为阻力线，均来自近期收盘价拟合。\n\n"
                        "请分析价格与趋势线的互动关系: 是否反弹、跌破/突破，或在区间内持续压缩。\n\n"
                        "结合趋势线斜率、线间距和近期K线行为，判断短期趋势更可能是: 上涨、下跌或震荡。"
                        "请给出结论并说明依据(关键信号与推理过程)。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)

            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")

            messages = [
                SystemMessage(
                    content="你是高频交易场景下的K线趋势识别助手，任务是分析含支撑/阻力线的K线图并判断短期走势。"
                ),
                human_msg,
            ]

            try:
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                if "at least one message" in error_str.lower():
                    print("Anthropic消息转换异常，改为仅传HumanMessage重试...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
                else:
                    raise
        else:
            final_response = invoke_with_retry(chain.invoke, messages)

        return {
            "messages": messages + [final_response],
            "trend_report": final_response.content,
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "带支撑/阻力趋势线的K线图"
                if trend_image_b64
                else None
            ),
        }

    return trend_agent_node
