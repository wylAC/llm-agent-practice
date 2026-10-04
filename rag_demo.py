# -*- coding: utf-8 -*-
"""
RAG 最小 demo（Day5 动手任务）
==============================
RAG = Retrieval-Augmented Generation（检索增强生成）全流程：

【离线阶段：建知识库】
  corpus/ 里的 txt → 按段落切成 chunk → 每个 chunk 算成「向量」→ 存起来

【在线阶段：问答】
  问题 → 算成「查询向量」→ 和所有 chunk 向量算相似度 → 取 top-k 最像的
      → 把检索到的原文 + 问题拼进 prompt → DeepSeek 生成回答

两种向量化方式（本脚本自动选择）：
  TF-IDF（默认）：手写实现，字面匹配，零模型零下载 —— 今天跑这个
  bge 语义向量：需要装 sentence-transformers + 下载模型 —— 明天可升级

用法：
  $env:DEEPSEEK_API_KEY="sk-你的key"
  python rag_demo.py                          # 问一个默认问题（完整 RAG）
  python rag_demo.py --question "你的问题"     # 问自定义问题
  python rag_demo.py --top-k 5                # 调整检索数量
  python rag_demo.py --no-llm                 # 调试模式：只看检索结果，不需要 key
"""
import argparse
import glob
import math
import os
import re
from collections import Counter

# 脚本所在目录（用 __file__ 拿到本文件路径，取目录部分）。
# 这样无论从哪个目录运行本脚本，都能找到语料——不依赖「当前工作目录」。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════
# 【离线阶段·第一步】加载语料 + 切分 chunk
# ══════════════════════════════════════════════════════════════════
def load_chunks(folder=None):
    """
    把语料目录里每个 txt 按「段落」切成一个个 chunk。
    folder 不给时，默认用「脚本所在目录下的 corpus/」。
    返回列表，每个 chunk 是一个字典：
      {"source": 文件名, "para": 第几段, "text": 段落文字}
    """
    if folder is None:
        folder = os.path.join(SCRIPT_DIR, "corpus")
    chunks = []
    # glob.glob：找出所有 .txt 文件路径（* 是通配符）
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:   # with open：安全读写文件
            text = f.read()
        # split("\n") 按换行切成段落；strip() 去掉首尾空白；if p.strip() 跳过空段
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        for i, para in enumerate(paras):
            chunks.append({"source": os.path.basename(path), "para": i, "text": para})
    return chunks


# ══════════════════════════════════════════════════════════════════
# 【离线阶段·第二步】向量化 —— 手写 TF-IDF（不需要任何模型）
# ══════════════════════════════════════════════════════════════════
def tokenize(text):
    """
    把一段中文文字切成「词」。
    中文没有空格分词，教学版用最简单的办法：相邻两个字滑窗（bigram）。
    例："北京天气" → ["北京", "京天", "天气"]
    英文单词则整词切出来。
    """
    # 只保留中文、英文、数字，其余符号删掉，并统一小写
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text.lower())
    # 中文字符两字滑窗：text[i:i+2] 是「切片」语法，取从 i 开始长度为 2 的子串
    grams = [text[i : i + 2] for i in range(len(text) - 1)]
    # 英文数字词整词提取（re.findall 找出所有匹配的片段）
    grams += re.findall(r"[a-z0-9]+", text)
    return grams


def build_tfidf(chunks):
    """
    给每个 chunk 算一个 TF-IDF 向量。
    TF-IDF 思想（两个乘法）：
      TF  = 词在这个 chunk 里出现次数        → 这段在聊什么
      IDF = log(总chunk数 / 含该词的chunk数) → 这个词有多「稀有」（常见词权重低）
    一句话：一个词越在这段里频繁出现、越在全库稀有，它的分量越大。
    返回 (词表 vocab, 所有 chunk 的向量列表 vectors, 每个词的 idf)
    """
    # 第一步：统计每个 chunk 的词频，以及每个词出现在多少个 chunk（df）
    df = Counter()          # 词 → 出现在几个 chunk 里
    tokenized = []          # 每个 chunk 的词频表
    for c in chunks:
        toks = Counter(tokenize(c["text"]))
        tokenized.append(toks)
        for term in toks:
            df[term] += 1

    n_docs = len(chunks)          # 总共几个 chunk
    vocab = sorted(df.keys())     # 词表（排好序，所有向量同一套下标）

    # 第二步：按词表下标，给每个 chunk 填出 TF-IDF 向量
    vectors = []
    for toks in tokenized:
        vec = [0.0] * len(vocab)          # 先全 0，再填
        for j, term in enumerate(vocab):
            if term in toks:
                idf = math.log(n_docs / df[term])   # 稀有词 idf 大
                vec[j] = toks[term] * idf            # tf × idf
        vectors.append(vec)
    return vocab, vectors


def vectorize_query(question, vocab):
    """
    把查询问题也变成同维度的 TF-IDF 向量（才能和 chunk 向量比较）。
    """
    toks = Counter(tokenize(question))
    vec = [0.0] * len(vocab)
    for j, term in enumerate(vocab):
        if term in toks:
            # 查询短，只算 tf 即可（简化版，教学用）
            vec[j] = toks[term]
    return vec


# ══════════════════════════════════════════════════════════════════
# 【可选·语义向量化】bge 模型：文字 → 语义向量
# 和 TF-IDF 的本质区别：TF-IDF 看「字面有没有相同的词」，
# bge 看「意思近不近」——「拿过什么奖」和「获奖经历」在 bge 里很近，
# 在 TF-IDF 里可能完全不沾边。需要 torch + sentence-transformers。
# ══════════════════════════════════════════════════════════════════
_embed_model = None   # 全局变量：缓存加载好的模型，避免每次重复加载


def get_embed_model():
    """第一次调用才加载 bge 模型（首次会下载 ~100MB，之后用本地缓存）"""
    global _embed_model
    if _embed_model is None:
        # 国内镜像加速模型下载；模型已缓存时这行不影响速度。
        # setdefault：只在用户没设置过 HF_ENDPOINT 时才写默认值
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer
        # bge-small-zh-v1.5：中文小模型，效果/速度平衡；面试说「bge 系列」即可
        _embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _embed_model


def embed_texts(texts):
    """把多段文字批量变成语义向量；normalize 归一化后，点积就是余弦相似度"""
    model = get_embed_model()
    return model.encode(texts, normalize_embeddings=True)


# ══════════════════════════════════════════════════════════════════
# 【在线阶段·第一步】余弦相似度检索 top-k
# ══════════════════════════════════════════════════════════════════
def cosine(a, b):
    """余弦相似度 = 点积 / (长度A × 长度B)，值越接近 1 越像"""
    dot = sum(x * y for x, y in zip(a, b))        # zip 把两个列表配对
    na = math.sqrt(sum(x * x for x in a))          # 向量 a 的长度
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(query_vec, chunks, vectors, top_k=3):
    """
    把查询向量和所有 chunk 向量逐个算相似度，取 top-k。
    真实生产环境里这步用 FAISS 加速（几百万向量也毫秒级），
    这里手写循环是为了让你看清「相似度到底在算什么」。
    """
    scores = []
    for i, vec in enumerate(vectors):
        scores.append((cosine(query_vec, vec), i))    # (分数, chunk 下标)
    scores.sort(reverse=True, key=lambda x: x[0])     # 按分数从高到低排
    return [(i, s) for s, i in scores[:top_k]]        # 取前 top_k 个


# ══════════════════════════════════════════════════════════════════
# 【在线阶段·第二步】拼 prompt + LLM 生成
# ══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。请只根据下面提供的【参考资料】回答问题。\n"
    "规则：如果参考资料中没有答案，直接回答「不知道」，不要编造；"
    "如果资料与问题无关，也回答「不知道」。"
)


def ask_llm(client, model, question, retrieved_chunks):
    """把检索到的原文拼进 prompt，调 DeepSeek 生成"""
    # 参考资料按编号列出（[1][2]...），让模型可以引用
    refs = "\n\n".join(
        f"[{i + 1}] {text}" for i, (_, text) in enumerate(retrieved_chunks)
    )
    user_prompt = f"【参考资料】\n{refs}\n\n【问题】{question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},   # 角色设定（约束行为）
            {"role": "user", "content": user_prompt},       # 资料 + 问题
        ],
    )
    return resp.choices[0].message.content


# ══════════════════════════════════════════════════════════════════
# 主流程：把两阶段串起来（支持 TF-IDF / 语义双模式 + 对比）
# ══════════════════════════════════════════════════════════════════
def _print_top(title, chunks, query_vec, vectors, top_k):
    """把一次检索的 top-k 结果打印出来（对比模式复用）"""
    print(f"[{title}]")
    top = retrieve(query_vec, chunks, vectors, top_k=top_k)
    for i, score in top:
        c = chunks[i]
        snippet = c["text"][:40] + ("…" if len(c["text"]) > 40 else "")
        print(f"  相似度 {score:.3f} | {c['source']} 第{c['para']}段 | {snippet}")
    print()


def run_rag(client, model, question, top_k=3, folder=None, no_llm=False, mode="auto", compare=False):
    print("=" * 60)
    print("【离线阶段】加载语料并建索引……")
    chunks = load_chunks(folder)
    print(f"  共 {len(chunks)} 个 chunk（来自 {len(set(c['source'] for c in chunks))} 篇文档）")

    # TF-IDF 索引总是建好（对比模式要同时用两种）
    vocab, tfidf_vecs = build_tfidf(chunks)
    print(f"  TF-IDF 词表 {len(vocab)} 个词（字面检索就绪）")

    # 语义向量：尝试加载 bge 模型；失败则回退
    sem_ok, sem_vecs = False, None
    if mode in ("auto", "embed") or compare:
        try:
            sem_vecs = embed_texts([c["text"] for c in chunks])
            sem_ok = True
            print(f"  bge 语义向量就绪：{len(sem_vecs)} 个 chunk，维度 {len(sem_vecs[0])}")
        except Exception as e:   # 没装依赖/模型下载失败都会走到这
            print(f"  bge 语义向量不可用（{type(e).__name__}），字面检索仍可用")
            if mode == "embed":
                raise SystemExit("强制语义模式失败。请先安装：pip install torch sentence-transformers")

    print("【在线阶段】处理问题……")
    print(f"[用户] {question}\n")

    # ── 对比模式：同一问题，两种检索各打一份，不调 LLM ──
    if compare:
        _print_top("TF-IDF 字面检索", chunks, vectorize_query(question, vocab), tfidf_vecs, top_k)
        if sem_ok:
            _print_top("bge 语义检索", chunks, embed_texts([question])[0], sem_vecs, top_k)
        print("=" * 60)
        return

    # ── 选一种检索向量 ──
    if mode == "embed" and sem_ok:
        query_vec, vectors, tag = embed_texts([question])[0], sem_vecs, "bge 语义检索"
    else:
        query_vec, vectors, tag = vectorize_query(question, vocab), tfidf_vecs, "TF-IDF 字面检索"

    top = retrieve(query_vec, chunks, vectors, top_k=top_k)
    retrieved = []
    print(f"[检索结果·{tag}] top-{top_k}：")
    for i, score in top:
        c = chunks[i]
        snippet = c["text"][:40] + ("…" if len(c["text"]) > 40 else "")
        print(f"  相似度 {score:.3f} | {c['source']} 第{c['para']}段 | {snippet}")
        retrieved.append((score, c["text"]))

    # 调试模式：只验证检索，不调 LLM（不需要 key）
    if no_llm:
        print("\n（--no-llm 调试模式：跳过 LLM 生成。加 key 后去掉此参数跑完整版）")
        print("=" * 60)
        return

    answer = ask_llm(client, model, question, retrieved)
    print(f"\n[最终回答] {answer}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 最小 demo")
    parser.add_argument("--question", default="小明有哪些竞赛经历？", help="要问的问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索多少个 chunk（默认 3）")
    parser.add_argument("--no-llm", action="store_true", help="调试模式：只看检索结果，不调 LLM（无需 key）")
    parser.add_argument("--mode", choices=["auto", "tfidf", "embed"], default="auto",
                        help="检索模式：auto=语义可用则用语义（默认）；tfidf=字面；embed=强制语义")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式：同一问题分别用 TF-IDF 与语义各检索一遍（不调 LLM，无需 key）")
    args = parser.parse_args()

    # 调试/对比模式：不需要 API key
    if args.no_llm or args.compare:
        run_rag(None, None, args.question, top_k=args.top_k,
                no_llm=True, mode=args.mode, compare=args.compare)
        return

    from openai import OpenAI

    # 复用 FC demo 的 key 读取逻辑（DeepSeek 默认）
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("缺少 API key。示例：$env:DEEPSEEK_API_KEY='sk-xxx' 后再运行本脚本")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    run_rag(client, "deepseek-chat", args.question, top_k=args.top_k,
            mode=args.mode, compare=args.compare)


if __name__ == "__main__":
    main()
