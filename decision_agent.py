"""
Agent for making final trade decisions in high-frequency trading (HFT) context.
Combines indicator, pattern, and trend reports to issue a LONG or SHORT order.
"""


def create_final_trade_decider(llm):
    """
    Create a trade decision agent node. The agent uses LLM to synthesize indicator, pattern, and trend reports
    and outputs a final trade decision (LONG or SHORT) with justification and risk-reward ratio.
    """

    def trade_decision_node(state) -> dict:
        indicator_report = state["indicator_report"]
        pattern_report = state["pattern_report"]
        trend_report = state["trend_report"]
        time_frame = state["time_frame"]
        stock_name = state["stock_name"]

        # --- System prompt for LLM ---
        prompt = f"""你是一名高频量化交易(HFT)分析师，当前正在分析 {stock_name} 的 {time_frame} 周期K线。你的任务是给出**立即执行**的交易指令: **LONG** 或 **SHORT**。HFT场景下禁止输出 HOLD。

你的判断需要预测未来 **N 根K线** 的方向，其中:
- 例如: TIME_FRAME=15min, N=1，表示预测未来15分钟。
- TIME_FRAME=4hour, N=1，表示预测未来4小时。

请综合以下三份报告的强度、一致性与时效性后再下结论:

---

### 1. 技术指标报告
- 评估动量类指标(如 MACD、ROC)与震荡类指标(如 RSI、Stochastic、Williams %R)。
- 对方向性强信号给予更高权重，例如: MACD金叉/死叉、RSI背离、超买超卖极值。
- 中性或相互矛盾信号应降权，除非多个指标同向共振。

---

### 2. 形态报告
- 只有在以下条件成立时，才可据此执行多空:
- 形态清晰可辨且接近完成；
- 已出现突破/破位，或根据价格与动量显示极高概率即将突破/破位(如长影线、放量、吞没)。
- 对早期、猜测性形态不要直接交易。若无其他报告确认，不应将纯整理结构视作可执行信号。

---

### 3. 趋势报告
- 分析价格与支撑/阻力线的关系:
- 上升支撑线通常代表买盘承接。
- 下降阻力线通常代表卖压主导。
- 若价格在趋势线之间压缩:
- 仅当存在强K线或指标共振时，才预测突破方向。
- 不要仅凭几何形态主观猜测突破方向。

---

### 决策策略

1. 仅基于**已确认**信号决策，避免早期、投机或冲突信号。
2. 优先选择三份报告(指标/形态/趋势)同向一致的机会。
3. 更重视以下证据:
- 最近动量显著增强(如 MACD交叉、RSI突破)
- 明确价格行为(如突破实体K线、拒绝影线、支撑反弹)
4. 若报告不一致:
- 选择**确认更强、时间更近**的一侧
- 优先动量确认，弱震荡提示降权
5. 若市场明显盘整或信号混杂:
- 默认遵循主导趋势线斜率方向(如下降通道优先 SHORT)
- 不要拍方向，选择证据更充分的一侧
6. 结合当前波动与趋势强度，给出 **1.2~1.8** 区间内的合理风险收益比。

---
### 输出格式(必须是可解析JSON):

```
{{
"forecast_horizon": "Predicting next 3 candlestick (15 minutes, 1 hour, etc.)",
"decision": "<LONG or SHORT>",
"justification": "<Concise, confirmed reasoning based on reports>",
"risk_reward_ratio": "<float between 1.2 and 1.8>"
}}
```

### 语言要求(必须遵守):
- 最终输出 JSON 的所有 value 必须使用英文(English)。
- `decision` 仅允许 `LONG` 或 `SHORT`。
- 不要输出中文，不要输出 JSON 以外的额外文本。

--------
**Technical Indicator Report**  
{indicator_report}

**Pattern Report**  
{pattern_report}

**Trend Report**  
{trend_report}
"""

        # --- LLM call for decision ---
        response = llm.invoke(prompt)

        return {
            "final_trade_decision": response.content,
            "messages": [response],
            "decision_prompt": prompt,
        }

    return trade_decision_node
