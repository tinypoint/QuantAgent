import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import RateLimitError


def invoke_tool_with_retry(tool_fn, tool_args, retries=3, wait_sec=4):
    """
    Invoke a tool function with retries if the result is missing an image.
    """
    for attempt in range(retries):
        result = tool_fn.invoke(tool_args)
        img_b64 = result.get("pattern_image")
        if img_b64:
            return result
        print(
            f"工具未返回图像，{wait_sec}秒后重试 (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_sec)
    raise RuntimeError("多次重试后仍未生成图像")


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses precomputed images from state or falls back to tool generation.
    """

    def pattern_agent_node(state):
        tools = [toolkit.generate_kline_image]
        time_frame = state["time_frame"]
        pattern_text = """
        请参考以下经典K线形态:

        1. 倒头肩底: 三个低点, 中间最低且结构相对对称, 常预示上行。
        2. 双底: 两个相近低点, 中间反弹, 形成“W”形。
        3. 圆弧底: 价格缓慢下行后再缓慢回升, 呈“U”形。
        4. 隐藏底部平台: 横盘整理后向上突破。
        5. 下降楔形: 价格向下收敛, 常向上突破。
        6. 上升楔形: 价格缓慢上行并收敛, 常向下破位。
        7. 上升三角形: 下方支撑抬高、上方阻力水平, 常向上突破。
        8. 下降三角形: 上方阻力下移、下方支撑水平, 常向下突破。
        9. 看涨旗形: 急涨后短暂下倾整理, 随后继续上行。
        10. 看跌旗形: 急跌后短暂上倾整理, 随后继续下行。
        11. 矩形整理: 价格在水平支撑与阻力间震荡。
        12. 岛形反转: 两个反向缺口形成“孤岛”区域。
        13. V形反转: 急跌急拉或急拉急跌。
        14. 圆弧顶/圆弧底: 缓慢见顶或见底形成弧线。
        15. 扩散三角形: 高低点振幅逐步扩大, 波动增强。
        16. 对称三角形: 高低点逐步收敛至顶点, 常伴随突破。
        """

        pattern_image_b64 = state.get("pattern_image")

        def invoke_with_retry(call_fn, *args, retries=3, wait_sec=8):
            for attempt in range(retries):
                try:
                    return call_fn(*args)
                except RateLimitError:
                    print(
                        f"触发限流，{wait_sec}秒后重试 (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
                except Exception as e:
                    print(
                        f"调用异常: {e}，{wait_sec}秒后重试 (attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
            raise RuntimeError("超过最大重试次数")

        messages = state.get("messages", [])

        if not pattern_image_b64:
            print("state中没有预生成形态图，开始使用工具生成...")

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是交易形态识别助手，负责识别经典高频交易形态。"
                        "你可使用工具 generate_kline_image，并应通过 `kline_data` 生成图像。\n\n"
                        "生成图像后，请对照经典形态定义，判断是否存在可识别形态。",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

            chain = prompt | tool_llm.bind_tools(tools)

            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = invoke_tool_with_retry(tool_fn, tool_args)
                    pattern_image_b64 = tool_result.get("pattern_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("使用state中预生成的形态图")

        if pattern_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"这是一张基于近期OHLC市场数据生成的 {time_frame} 周期K线图。\n\n"
                        f"{pattern_text}\n\n"
                        "请判断图中是否匹配上述任一形态。"
                        "如匹配，请明确给出形态名称，并基于结构、趋势与对称性说明理由。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)

            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")

            messages = [
                SystemMessage(
                    content="你是交易形态识别助手，负责基于K线图进行形态分析。"
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
            "pattern_report": final_response.content,
        }

    return pattern_agent_node
