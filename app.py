import base64
import lzma
import os
import threading

import gradio as gr
import numpy as np
import pandas as pd
import pybase16384 as b14
import torch
from gradio_leaderboard import Leaderboard
from pandas import DataFrame

# 临时文件目录
tmp_dir = os.path.join(os.getcwd(), "tmp")
os.makedirs(tmp_dir, exist_ok=True)

# ChatTTS Hugging Face 模型缓存目录。
# 可以在启动程序前通过 HF_ENDPOINT 切换 Hugging Face 镜像源，例如：
# HF_ENDPOINT=https://hf-mirror.com python app.py
MODEL_DIR = os.path.join(os.getcwd(), "models")
HF_CACHE_DIR = os.path.join(MODEL_DIR, "huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# 嵌入式设备音色试听固定文本。
# 每个 seed_id 都使用完全相同的文本，便于横向比较音色。
PREVIEW_TEXT = "你好，我是小智。设备连接成功。正在为您拍照。任务执行完成。"

# ChatTTS 模型采用懒加载：
# 启动 Gradio 时不加载；第一次点击 seed_id 时才下载/加载并缓存。
_chat_model = None
_chat_model_load_lock = threading.Lock()
_tts_infer_lock = threading.Lock()


def _encode_spk_emb(spk_emb: torch.Tensor) -> str:
    with torch.no_grad():
        arr: np.ndarray = spk_emb.to(dtype=torch.float16, device="cpu").numpy()
        s = b14.encode_to_string(
            lzma.compress(
                arr.tobytes(),
                format=lzma.FORMAT_RAW,
                filters=[
                    {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}
                ],
            ),
        )
        del arr
    return s


def pt2str(pt_path):
    spk_emb = torch.load(pt_path, map_location="cpu")
    return _encode_spk_emb(spk_emb)


def base64_to_file(base64_str, output_path):
    with open(output_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def convert_to_markdown(percentage_str):
    """
    将百分比字符串转换为 markdown 格式
    :param percentage_str:
    :return:
    """
    if not percentage_str:
        return ""
    if not isinstance(percentage_str, str):
        return percentage_str
    items = percentage_str.split(";")
    markdown_str = "  ".join(
        [f"**{item.split(':')[0]}** {item.split(':')[1]}%" for item in items]
    )
    return markdown_str


def convert_to_str(percentage_str):
    """
    将百分比字符串转换为 str
    :param percentage_str:
    :return:
    """
    if not percentage_str or not isinstance(percentage_str, str):
        return "未知"
    items = percentage_str.split(";")
    items.sort(key=lambda x: float(x.split(":")[1]), reverse=True)
    keys = [item.split(":")[0] for item in items]
    if keys and keys[0]:
        return keys[0]
    return "未知"


# Load
df = pd.read_csv("evaluation_results.csv", encoding="utf-8")

df["rank_long"] = df["rank_long"].apply(lambda x: round(x, 2))
df["rank_multi"] = df["rank_multi"].apply(lambda x: round(x, 2))
df["rank_single"] = df["rank_single"].apply(lambda x: round(x, 2))
df["gender_filter"] = df["gender"].apply(convert_to_str)
df["gender"] = df["gender"].apply(convert_to_markdown)
df["age_filter"] = df["age"].apply(convert_to_str)
df["age"] = df["age"].apply(convert_to_markdown)
df["feature"] = df["feature"].apply(convert_to_markdown)
df["score"] = df["score"].apply(lambda x: round(x, 2))


def restore_pt_file(seed_id):
    """
    根据给定的 seed_id 从 evaluation_results.csv 恢复 PT 音色文件。
    """
    row = df[df["seed_id"] == seed_id]
    if row.empty:
        return None
    row = row.iloc[0]
    emb_data = row["emb_data"]
    output_path = os.path.join(tmp_dir, f"{row['seed_id']}_restored_emb.pt")
    base64_to_file(emb_data, output_path)
    return output_path


def get_chat_model():
    """
    懒加载并缓存 ChatTTS 模型。

    第一次调用时通过 Hugging Face 下载 2Noise/ChatTTS 到项目的 models/huggingface 目录；
    可通过启动前设置 HF_ENDPOINT 切换镜像源，例如 https://hf-mirror.com。
    后续选择其他 seed_id 时复用同一个已加载模型。
    """
    global _chat_model

    if _chat_model is not None:
        return _chat_model

    with _chat_model_load_lock:
        if _chat_model is not None:
            return _chat_model

        try:
            import ChatTTS
        except ImportError as exc:
            raise RuntimeError(
                "未安装 ChatTTS，请先执行：pip install -r requirements.txt"
            ) from exc

        hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
        print("Preparing ChatTTS model from Hugging Face (first preview only)...")
        print(f"HF endpoint: {hf_endpoint}")
        print(f"HF cache directory: {HF_CACHE_DIR}")

        chat = ChatTTS.Chat()
        loaded = chat.load(
            source="huggingface",
            custom_path=HF_CACHE_DIR,
            compile=False,
        )
        if not loaded:
            raise RuntimeError(
                "ChatTTS 模型下载或加载失败。请检查 HF_ENDPOINT、网络以及 Hugging Face 缓存目录。"
            )

        _chat_model = chat
        print("ChatTTS model loaded from Hugging Face cache.")
        return _chat_model


def synthesize_preview(spk_emb_str):
    """
    使用当前 speaker embedding 实时合成固定的嵌入式设备播报测试语音。

    返回 Gradio Audio 支持的 (sample_rate, numpy_array)。
    """
    import ChatTTS

    chat = get_chat_model()
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=spk_emb_str,
        temperature=0.3,
        top_P=0.7,
        top_K=20,
        manual_seed=42,
    )

    # ChatTTS 的同一个模型实例不并发执行推理，避免多个 Gradio 请求互相影响。
    with _tts_infer_lock:
        wavs = chat.infer(
            PREVIEW_TEXT,
            skip_refine_text=True,
            params_infer_code=params_infer_code,
        )

    if not wavs:
        raise RuntimeError("ChatTTS 未返回音频数据。")

    audio = np.asarray(wavs[0], dtype=np.float32).squeeze()
    if audio.size == 0:
        raise RuntimeError("ChatTTS 返回了空音频。")

    return 24000, audio


def seed_change(evt: gr.SelectData, value=None):
    """
    处理 seed_id 选择事件：
    1. 恢复并提供 .pt 文件下载；
    2. 显示 speaker embedding；
    3. 使用 ChatTTS 实时合成固定测试语音。
    """
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    # Leaderboard 的搜索/筛选控件也会触发 select。
    # 只有点第一列 seed_id 时才真正生成试听音频。
    if not isinstance(evt.index, list) or evt.index[1] != 0:
        return [
            None,
            gr.DownloadButton(
                value=None, label="Download .pt File", visible=False
            ),
            gr.Audio(None, visible=False),
            "",
        ]

    assert isinstance(value, DataFrame), "Expected value to be a DataFrame"

    seed_id = evt.value
    print(f"Selected seed_id: {seed_id}")

    down_file = restore_pt_file(seed_id)
    if not down_file:
        print(f"Unable to restore speaker embedding for {seed_id}")
        return [
            evt.index,
            gr.DownloadButton(
                value=None, label=f"PT file unavailable [{seed_id}]", visible=True
            ),
            gr.Audio(
                None,
                label="TTS试听不可用：未找到音色数据",
                visible=True,
            ),
            "",
        ]

    spk_emb_str = pt2str(down_file)

    try:
        audio = synthesize_preview(spk_emb_str)
        audio_component = gr.Audio(
            value=audio,
            label=f"TTS实时试听 [{seed_id}]",
            visible=True,
        )
        print(f"TTS preview generated for {seed_id}")
    except Exception as exc:
        print(f"Failed to generate TTS preview for {seed_id}: {exc}")
        audio_component = gr.Audio(
            None,
            label="TTS试听生成失败，请查看终端日志",
            visible=True,
        )

    return [
        evt.index,
        gr.DownloadButton(
            value=down_file,
            label=f"Download .pt File [{seed_id}]",
            visible=True,
        ),
        audio_component,
        spk_emb_str,
    ]


with gr.Blocks() as demo:
    gr.Markdown("# 🥇 ChatTTS Speaker Leaderboard ")
    gr.Markdown("""
    ### 🎤 [ChatTTS](https://github.com/2noise/ChatTTS): 稳定音色查找与音色打标（实验性）

    点选 **seed_id** 后，将直接使用 ChatTTS 和该音色实时生成固定设备播报语音，无需外部 WAV 试听资源。

    ChatTTS 模型首次试听时从 **Hugging Face / 2Noise/ChatTTS** 下载并缓存到项目目录；可通过 `HF_ENDPOINT` 切换下载镜像。

    评估基于通义实验室：[eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
    """)

    with gr.Tab(label="🏆Leaderboard"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
### 参数解释

- **rank_long**: 长句文本的音色稳定性评分。
- **rank_multi**: 多句文本的音色稳定性评分。
- **rank_single**: 单句文本的音色稳定性评分。

这三个参数用于衡量不同音色在生成不同类型文本时的一致性，数值越高表示音色越稳定。

- **score**: 音色性别、年龄、特征的可能性，越高越准确。
- **gender age feature**: 音色的性别、年龄、特征。（特征准确度不高，仅供参考）

### 如何试听 / 下载音色

- 点选表格第一列的 **seed_id**。
- 程序会使用 ChatTTS 实时合成固定测试句。
- 第一次试听会从 Hugging Face 下载并加载 ChatTTS 模型；以后会复用本机缓存和已加载模型。
- 如需使用镜像，可运行：`HF_ENDPOINT=https://hf-mirror.com python app.py`。
- 点击 **Download .pt File** 可下载对应音色文件。
                """)
            with gr.Column(scale=3, min_width=800):
                leaderboard = Leaderboard(
                    value=df,
                    datatype=["markdown"] * 12,
                    select_columns=[
                        "seed_id",
                        "rank_long",
                        "rank_multi",
                        "rank_single",
                        "score",
                        "gender",
                        "age",
                        "feature",
                    ],
                    search_columns=["gender", "age"],
                    filter_columns=[
                        "rank_long",
                        "rank_multi",
                        "rank_single",
                        "gender_filter",
                        "age_filter",
                    ],
                    hide_columns=["emb_data", "gender_filter", "age_filter"],
                )
                stats = gr.State(value=[1])
                download_button = gr.DownloadButton(
                    "Download .pt File", visible=True
                )
                spk_emb_str = gr.Textbox(
                    "",
                    label="音色码/speaker embedding",
                    lines=10,
                )
                test_audio = gr.Audio(
                    visible=True,
                    label="TTS实时试听",
                )
                gr.Markdown(
                    f"固定试听文本：**{PREVIEW_TEXT}**"
                )
                leaderboard.select(
                    seed_change,
                    inputs=[leaderboard],
                    outputs=[
                        stats,
                        download_button,
                        test_audio,
                        spk_emb_str,
                    ],
                )

    with gr.Tab(label="📊Details"):
        gr.Markdown("""
    # 音色稳定性初步评估

    ## 原理

    利用通义实验室开源的 [eres2netv2_sv_zh-cn](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)
    SERes2NetV2 说话人识别模型，对同一个音色进行测评，评估其在不同语音样本中的一致性。

    - **rank_single**：单句文本稳定性。
    - **rank_multi**：多句文本稳定性。
    - **rank_long**：长句文本稳定性。

    当前页面的 TTS 试听与上述历史评估数据相互独立：试听直接使用 evaluation_results.csv 中保存的音色 embedding，
    由本机 ChatTTS 实时生成固定测试文本。
        """)


if __name__ == "__main__":
    demo.launch()
