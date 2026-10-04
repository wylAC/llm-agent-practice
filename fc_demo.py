# -*- coding: utf-8 -*-
"""
Function Calling 最小 demo（Day4 动手任务）
========================================
演示四步流程：
  ① 程序把「工具清单」随 prompt 发给模型
  ② 模型输出结构化 JSON，声明「调哪个函数、参数是什么」
  ③ 程序（不是模型！）解析 JSON 并执行真实函数
  ④ 结果以 role="tool" 回填 → 再请求模型 → 得到最终答案

用法
----
真实模式（需要 DeepSeek 或任一 OpenAI 兼容平台的 API key）：
  $env:DEEPSEEK_API_KEY="sk-你的key"
  python fc_demo.py                      （默认走 DeepSeek，零配置）
  python fc_demo.py --base-url https://ark.cn-beijing.volces.com/api/v3 --model 你的模型ID --api-key 你的key
  （火山方舟/豆包、OpenAI 等兼容端点均可）

Mock 模式（无 key，本地模拟模型输出，跑通全流程逻辑）：
  python fc_demo.py --mock

════════════════════════════════════════════
阅读前的 Python 语法小抄（不熟的语法都在这）
════════════════════════════════════════════
- 字典 dict：{"key": value}，用 key 取值，如 d["city"]
- 列表 list：[a, b, c]，可追加元素 list.append(新元素)
- json.dumps(字典)：把 Python 字典变成 JSON 字符串（模型只能读字符串）
- json.loads(字符串)：把 JSON 字符串变回 Python 字典（程序要操作数据）
- f"...{变量}..."：f-string，字符串里嵌入变量值
- def 函数名(参数: 类型) -> 返回类型：定义函数（类型标注只是提示，不强制）
- **字典：把字典展开成 关键字参数，如 f(**{"city":"北京"}) 等价于 f(city="北京")
- for x in 列表：遍历；range(1,6) 生成 1,2,3,4,5
- 三元表达式：A if 条件 else B（条件为真取 A，否则取 B）
- if __name__ == "__main__"：只有直接运行本文件时才执行（被 import 时不执行）
"""
import argparse  # 标准库：解析命令行参数（--mock、--question 那些）
import json      # 标准库：处理 JSON 字符串 ↔ Python 字典的转换
import os        # 标准库：读环境变量（os.getenv）

# ----------------------------------------------------------------------
# ① 工具清单（Tools 组件）
# ----------------------------------------------------------------------
# 这是一个「列表」TOOLS，里面装着两个「字典」，每个字典描述一个工具。
# 结构：TOOLS = [ 工具1的说明书, 工具2的说明书, ... ]
# 每个说明书有 3 个关键字段（模型只能看到这三样，看不到你的函数代码）：
#   name        → 函数名，模型声明调用时用它
#   description → 功能描述，模型靠这段文字判断「该不该调这个工具」
#   parameters  → JSON Schema：告诉模型参数叫什么、什么类型、哪个必填
#
# 面试必懂：模型不会读你的 Python 代码，它只读这份「说明书」。
# description 写得好不好，直接决定模型选工具准不准。
TOOLS = [
    # ---------- 工具 1：查天气 ----------
    {
        "type": "function",                      # 固定写法：声明这是个函数型工具
        "function": {
            "name": "get_weather",               # 函数名（必须和真实函数同名！）
            "description": "查询指定城市的当前天气，返回天气状况与温度",
            "parameters": {
                "type": "object",                # 参数整体是个「对象」（键值对）
                "properties": {                  # 对象里有哪些参数：
                    "city": {                    #   参数 1：城市（字符串，必填）
                        "type": "string",
                        "description": "城市名，例如：北京",
                    },
                    "unit": {                    #   参数 2：温度单位（可选，二选一）
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],   # enum = 只允许这两个值
                        "description": "温度单位，默认 celsius",
                    },
                },
                "required": ["city"],            # 必填参数列表：只要求 city
            },
        },
    },
    # ---------- 工具 2：查时间 ----------
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询指定城市的当前时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如：东京"},
                },
                "required": ["city"],
            },
        },
    },
]


# ----------------------------------------------------------------------
# ③ 真实工具函数：由程序执行（模型只声明，不执行！）
# ----------------------------------------------------------------------
def get_weather(city: str, unit: str = "celsius") -> str:
    # : str 是类型提示（参数是字符串）；= "celsius" 是默认值（不传 unit 就用它）
    # json.dumps(字典) 把字典变成字符串返回——因为最后要作为「文本」回填给模型
    # ensure_ascii=False：让中文正常显示（否则会变成 \uXXXX 转义码）
    return json.dumps(
        {"city": city, "weather": "晴", "temperature": 25, "unit": unit},
        ensure_ascii=False,
    )


def get_time(city: str) -> str:
    # 注意：这里的时间是写死的假数据，真实项目应接真实时间 API
    return json.dumps({"city": city, "time": "2026-10-02 15:30:00"}, ensure_ascii=False)


# 工具名 → 真实函数的对照字典。模型说出函数名，程序靠它找到真正的函数。
# 注意：值是函数本身（没有括号 = 不调用，只是把函数对象放进去）
TOOL_EXECUTORS = {"get_weather": get_weather, "get_time": get_time}


# ----------------------------------------------------------------------
# ③ 执行工具 + ④ 结果回填
# ----------------------------------------------------------------------
def execute_tool_calls(messages, tool_calls):
    """
    输入：
      messages   = 对话记录列表（每项是一个字典：role 角色 + content 内容）
      tool_calls = 模型声明要调用的列表，每项是 (调用ID, 函数名, 参数JSON字符串)
    作用：逐个解析参数 → 执行真实函数 → 把结果以 role="tool" 拼回对话记录
    """
    for call_id, name, args_json in tool_calls:
        # 遍历每一个待执行的调用；一次拆出三个变量：
        #   call_id  → 模型给这次调用编的 ID（回填结果时要原样带回去）
        #   name     → 函数名字符串，如 "get_weather"
        #   args_json→ 参数 JSON 字符串，如 '{"city": "北京"}'（注意：是字符串！）

        args = json.loads(args_json or "{}")
        # json.loads：把 JSON 字符串解析成 Python 字典
        # args_json or "{}"：如果参数为空就解析成空字典，防止解析报错
        # 例：'{"city": "北京"}' → {"city": "北京"}

        print(f"[程序执行工具] {name}({args})")

        result = TOOL_EXECUTORS[name](**args)
        # TOOL_EXECUTORS[name]：按函数名找到真实函数
        # (**args)：把字典展开成关键字参数 → 等价于 get_weather(city="北京")
        # result 是函数的返回值（一个 JSON 字符串）

        print(f"[工具返回]       {result}")

        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
        # ④ 关键一步：把结果拼回对话记录，并告诉模型「这是回答你哪次调用的」：
        #   role="tool"       → 标记这是一条工具返回的消息
        #   tool_call_id      → 必须和模型发出调用时的 ID 一致（模型靠它认领）
        #   content=result    → 工具的真实返回内容
        # append 之后，messages 多一条；下一轮循环时整个 messages 都会发给模型


# ----------------------------------------------------------------------
# Mock 模型：本地假模型（替身演员），让没 key 也能跑通全流程
# ----------------------------------------------------------------------
def mock_model_reply(messages):
    """
    扮演「模型」的角色，接收对话记录，返回模型该说的话。
    真实模式下，这一步由 DeepSeek 服务器完成（client.chat.completions.create）。
    返回两个东西：
      assistant_entry = 模型这轮的消息（字典，可能带 tool_calls 声明）
      tool_calls      = 待执行的调用列表（没有调用就是空列表 []）
    """
    question = messages[0]["content"]
    # messages[0] 是第一条消息（用户提问）；["content"] 取出文字内容

    # ---- 第 1 轮：模仿模型「声明调用工具」 ----
    if len(messages) == 1:
        # 对话里只有用户一句话 → 该声明调工具了。
        # 用最简单的规则选工具（真实模型靠 description 理解语义）：
        if "时间" in question:
            name, args = "get_time", {"city": "东京"}
        elif "天气" in question:
            name, args = "get_weather", {"city": "北京"}
        else:  # 都没有就默认查天气
            name, args = "get_weather", {"city": "北京"}

        # 打印「模型声明调用」这行，方便你看清楚 mock 在干什么
        print("[模型声明调用] ", json.dumps({"name": name, "arguments": args}, ensure_ascii=False))

        # 拼出「助手消息」字典：content 为空，但带 tool_calls 声明
        assistant_entry = {
            "role": "assistant",
            "content": None,   # 这轮不回答用户，只声明调用
            "tool_calls": [    # 声明列表（可以多个，这里 1 个）
                {
                    "id": "call_mock_1",          # mock 版固定 ID
                    "type": "function",
                    "function": {
                        "name": name,             # 要调的函数名
                        "arguments": json.dumps(args, ensure_ascii=False),  # 参数转 JSON 字符串
                    },
                }
            ],
        }
        # 同时准备「待执行列表」：(调用ID, 函数名, 参数JSON字符串)
        return assistant_entry, [("call_mock_1", name, json.dumps(args, ensure_ascii=False))]

    # ---- 第 2 轮：工具结果已在对话里 → 模仿模型给最终回答 ----
    tool_result = messages[-1]["content"]
    # messages[-1] 是最后一条消息（程序刚回填的工具结果）
    data = json.loads(tool_result)   # 把结果字符串解析回字典（顺便演示 json.loads）
    assistant_entry = {
        "role": "assistant",
        "content": f"根据工具返回的真实数据：{data}，我来回答你的问题。",
        # 这是死模板，所以你能一眼认出 mock 输出；真实模型的回答是自由发挥的
    }
    return assistant_entry, []   # 第二个元素 [] = 不再声明调用，循环结束


# ----------------------------------------------------------------------
# Agent 主循环（程序是导演：调模型 → 看输出 → 分流处理 → 再调模型）
# ----------------------------------------------------------------------
def run_agent(client, model, question, mock=False):
    # client：OpenAI 客户端对象（真实模式才有；mock 模式传 None 用不上）
    # question：用户问题；mock：True 走假模型，False 走真实 API

    messages = [{"role": "user", "content": question}]
    # messages = 对话记录。初始只有一条用户消息。
    # 之后每轮往里 append，发给模型时整体发送——这就是「上下文/记忆」在代码里的样子

    print(f"[用户] {question}\n")

    for step in range(1, 6):
        # range(1,6) → 1,2,3,4,5：最多循环 5 轮，防止模型一直调工具导致死循环
        print(f"----- 第 {step} 轮 -----")

        # ═══ ② 请求模型：把对话记录 + 工具清单一起发过去 ═══
        if mock:
            # mock 分支：本地规则扮演模型（不发网络请求）
            assistant_entry, tool_calls = mock_model_reply(messages)
        else:
            # 真实分支：这一行才调用 API（key 随请求发出，花你的余额）
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS
            )
            # resp 是 API 的返回对象；choices[0] 是第一个候选回答，.message 是消息
            msg = resp.choices[0].message

            # 模型可能声明调用工具，也可能没声明：
            # 三元表达式：有 tool_calls 就逐项拆出 (ID, 函数名, 参数)，否则给空列表
            tool_calls = (
                [(tc.id, tc.function.name, tc.function.arguments) for tc in msg.tool_calls]
                if msg.tool_calls
                else []
            )

            # 把 API 返回的消息转成字典存进对话记录（否则下一轮没法发给 API）
            assistant_entry = {"role": "assistant", "content": msg.content}
            if tool_calls:   # 如果有调用声明，也要一并存进去
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_json},
                    }
                    for tc_id, name, args_json in tool_calls   # 列表推导式，逐项转字典
                ]

        messages.append(assistant_entry)   # 模型的发言进入对话记录

        # ═══ 分流判断：模型这轮有没有声明调工具？ ═══
        if not tool_calls:
            # 空列表 = 模型直接回答了 → 打印最终答案，结束整个循环
            print(f"[最终回答] {assistant_entry['content']}\n")
            return   # return 直接退出函数（循环自然结束）

        # 有声明 → 走 ③ 执行 + ④ 回填，然后回到循环顶部再问模型
        execute_tool_calls(messages, tool_calls)

    # 循环 5 次都用完还没拿到最终答案（模型一直在调工具）——兜底提示
    print("[结束] 达到最大轮数，未得到最终答案（演示用上限）")


def main():
    # ── 用 argparse 解析命令行参数 ──
    # 例：python fc_demo.py --mock --question "东京现在几点？"
    # 解析后 args.mock=True、args.question="东京现在几点？"
    parser = argparse.ArgumentParser(description="Function Calling 最小 demo")
    parser.add_argument("--mock", action="store_true", help="无 API key 的本地模拟模式")
    # action="store_true"：只要命令行出现了 --mock 就是 True，没出现就是 False
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容端点，默认 DeepSeek：https://api.deepseek.com")
    parser.add_argument("--model", default=None, help="模型 ID，默认 deepseek-chat")
    parser.add_argument("--api-key", default=None, help="API key（也可用环境变量 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ARK_API_KEY / OPENAI_API_KEY）")
    parser.add_argument("--question", default="北京今天天气怎么样？", help="要问的问题，如「东京现在几点？」")
    args = parser.parse_args()   # 真正读取命令行，结果存进 args

    # ── 分支 1：mock 模式（不需要 openai 库、不需要 key） ──
    if args.mock:
        run_agent(None, None, args.question, mock=True)
        # client 和 model 传 None（用不上）；mock=True 走假模型
        print("（mock 模式结束：全流程四步已跑通，换成真实 key 即可看到真模型行为）")
        return   # 直接结束，后面的真实模式代码不会执行

    # ── 分支 2：真实模式 ──
    from openai import OpenAI
    # 延迟导入：只在真实模式才 import（mock 模式没装 openai 也能跑）

    # 依次找 key：命令行参数 → 环境变量 DEEPSEEK → DASHSCOPE → ARK → OPENAI
    # A or B or C：从左到右，第一个「非空」的值生效
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    # 同理：base_url 和 model 都有默认值（DeepSeek），不给参数也能跑
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
    model = args.model or os.getenv("OPENAI_MODEL") or "deepseek-chat"

    if not api_key:
        # not api_key = api_key 为空：提示用户后直接退出程序
        raise SystemExit("缺少 API key。示例：$env:DEEPSEEK_API_KEY='sk-xxx' 后再运行本脚本")

    client = OpenAI(api_key=api_key, base_url=base_url)
    # 创建客户端对象：把 key 和服务器地址存进去（此刻还没联网，只是登记）
    # 真正用 key 发请求，是在 run_agent 里的 client.chat.completions.create(...)

    run_agent(client, model, args.question)   # 启动主循环


if __name__ == "__main__":
    # 直接运行本文件时（python fc_demo.py），__name__ 就是 "__main__"，执行 main()
    # 若被其他文件 import，则不会自动执行——这是 Python 脚本的标准写法
    main()
