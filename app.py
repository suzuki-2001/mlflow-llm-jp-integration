import os
import re
import uuid
import time
from typing import Dict, Optional, Any, Tuple

import pandas as pd
import torch
import mlflow
from mlflow.entities import SpanType
from loguru import logger
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFacePipeline


# MLflow configuration (used for tracing in the backend)
MLFLOW_TRACKING_DIR = os.path.abspath("./mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
MLFLOW_DB = f"sqlite:///{MLFLOW_TRACKING_DIR}/mlflow.db"
MLFLOW_ARTIFACT_DIR = f"file://{MLFLOW_TRACKING_DIR}"


# System prompt (Debug)
SYTEM_PROMPT = """あなたは高性能な日本語大規模言語モデルです。以下のルールに従い、ユーザーの入力に対して正確かつ論理的な回答を提供してください。

【ルール】
1. ユーザーの意図や文脈を十分に考慮し、的確な回答や補足説明を行うこと。
2. 回答が不明瞭な場合は、必要に応じて追加の質問や確認を促すこと。
3. 倫理的かつ中立的な姿勢を保ち、誤情報や不適切な表現を避けること。

【出力形式】
1. 2,3行程度の文章で簡潔に回答すること。
2. 常に日本語で回答すること。

これらの指示に基づき、常に最適な回答を生成してください。
"""


# Available LLM-JP models
# https://huggingface.co/collections/llm-jp/llm-jp-3-fine-tuned-models-672c621db852a01eae939731
LLM_JP_MODELS = [
    "llm-jp/llm-jp-3-1.8b",
    "llm-jp/llm-jp-3-1.8b-instruct",
    "llm-jp/llm-jp-3-3.7b",
    "llm-jp/llm-jp-3-3.7b-instruct",
    "llm-jp/llm-jp-3-13b",
    "llm-jp/llm-jp-3-13b-instruct",
    "llm-jp-3-172b-beta1",
    "llm-jp/llm-jp-3-172b-beta1-instruct",
    "llm-jp/llm-jp-3-150m-instruct2",
    "llm-jp/llm-jp-3-150m-instruct3",  # default
    "llm-jp/llm-jp-3-440m-instruct2",
    "llm-jp/llm-jp-3-440m-instruct3",
    "llm-jp/llm-jp-3-980m-instruct2",
    "llm-jp/llm-jp-3-980m-instruct3",
    "llm-jp/llm-jp-3-1.8b-instruct2",
    "llm-jp/llm-jp-3-1.8b-instruct3",
    "llm-jp/llm-jp-3-3.7b-instruct2",
    "llm-jp/llm-jp-3-3.7b-instruct3",
    "llm-jp/llm-jp-3-7.2b-instruct",
    "llm-jp/llm-jp-3-7.2b-instruct2",
    "llm-jp/llm-jp-3-7.2b-instruct3",
    "llm-jp/llm-jp-3-13b-instruct2",
    "llm-jp/llm-jp-3-13b-instruct3",
    "llm-jp/llm-jp-3-172b-instruct2",
    "llm-jp/llm-jp-3-172b-instruct3",
]


# Output parser: remove unnecessary prefixes
class OutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        splits = re.split(
            r"(?:Assistant:|assistant:|AI:|ai:)", text, flags=re.IGNORECASE
        )
        if len(splits) >= 2:
            return splits[-1].strip()
        return text.strip()


# Chat processing with MLflow tracing
def chat_with_mlflow_tracing(
    model: HuggingFacePipeline,
    model_params: Dict[str, Any],
    system_prompt: str,
    query: str,
    chat_history: ChatMessageHistory,
    sess_id: str,
    interaction_count: int,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_params["model_name"])
    prompt_template = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("ai", ""),
        ]
    )

    start_time = time.time()

    with mlflow.start_span(
        name="RunnableWithMessageHistory",
        span_type=SpanType.CHAT_MODEL,
        attributes={
            "session_id": sess_id,
            "model_name": model_params["model_name"],
            "model_parameters": model_params,
        },
    ) as span:

        # add session id tags
        mlflow.update_current_trace(tags={"session_id": sess_id})

        try:
            chain = prompt_template | model | OutputParser()
            chat = RunnableWithMessageHistory(
                chain,
                lambda: chat_history,
                input_messages_key="input",
                history_messages_key="history",
            )

            model_call_params = {
                k: v for k, v in model_params.items() if k != "model_name"
            }

            response = chat.invoke(
                {"input": query, **model_call_params},
                config={"configurable": {"session_id": sess_id}},
            )

            duration = time.time() - start_time

            # trace input-output
            span.set_inputs(
                {
                    "system_prompt": system_prompt,
                    "user_input": query,
                    "chat_history": chat_history.messages,
                }
            )
            span.set_outputs({"model_output": response})
            span.set_attributes(
                {
                    "system_prompt_tokens": len(tokenizer.encode(system_prompt)),
                    "user_input_tokens": len(tokenizer.encode(query)),
                    "model_output_tokens": len(tokenizer.encode(response)),
                    "response_time_sec": duration,
                }
            )
            return response

        except Exception as e:
            span.set_outputs({"error": str(e), "status": "error"})
            raise


# Model loading
def load_model(
    model_repo: str, model_params: Optional[Dict] = None
) -> Tuple[HuggingFacePipeline, Dict]:
    if model_params is None:
        model_params = {
            "max_new_tokens": 120,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_repo).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        **model_params,
    )

    return HuggingFacePipeline(pipeline=pipe), model_params


# Initialize session state
def initialize_session_state():
    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "sess_id" not in st.session_state:
        st.session_state.sess_id = ""
    if "llm_jp" not in st.session_state:
        st.session_state.llm_jp = None
    if "model_params" not in st.session_state:
        st.session_state.model_params = None
    if "experiment_id" not in st.session_state:
        st.session_state.experiment_id = None
    if "run_name" not in st.session_state:
        st.session_state.run_name = "LLMJP_Run"
    if "experiment_name" not in st.session_state:
        st.session_state.experiment_name = None


# Main application
def main():
    st.set_page_config(page_title="LLM-jp Chatbot", layout="wide")

    # sidebar & box container
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            padding: 20px;
        }
        .box-container {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        .sidebar-header {
            text-align: center;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .sidebar-description {
            text-align: center;
            font-size: 0.95rem;
            color: #555555;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Sidebar: Header, Session Settings, Parameter Settings, Control Buttons
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-header'>💬 LLM-jp Chatbot</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-description'>This chatbot is created using the open-source LLM-jp model</div>",
            unsafe_allow_html=True,
        )

        # Session Settings form
        with st.form(key="session_settings_form"):
            st.subheader("Session Settings")
            default_model = "llm-jp/llm-jp-3-150m-instruct3"

            try:
                default_index = LLM_JP_MODELS.index(default_model)
            except ValueError:
                default_index = 0

            selected_model = st.selectbox(
                "Select LLM Model",
                LLM_JP_MODELS,
                index=default_index,
                format_func=lambda x: x.split("/")[-1],
                help="Choose the model to use.",
            )

            experiment_name = st.text_input(
                "Experiment Name",
                value="LLM-JP-RUN",
                help="Enter the experiment name.",
            )

            run_name = st.text_input(
                "Run Name",
                # value=st.session_state.run_name,
                value=str(selected_model),
                help="Name this chat session.",
            )

            session_submitted = st.form_submit_button("Start Chat Session")

            if session_submitted:
                if not experiment_name or not run_name:
                    st.error("Please fill in all required fields.")
                else:
                    st.session_state.run_name = run_name
                    st.session_state.experiment_name = experiment_name
                    st.session_state.sess_id = str(uuid.uuid4())  # create session

                    mlflow.set_tracking_uri(MLFLOW_DB)
                    experiment = mlflow.get_experiment_by_name(experiment_name)

                    if experiment is None:
                        logger.debug(f"Creating new experiment: {experiment_name}")
                        experiment_id = mlflow.create_experiment(
                            name=experiment_name, artifact_location=MLFLOW_ARTIFACT_DIR
                        )
                    else:
                        experiment_id = experiment.experiment_id

                    st.session_state.experiment_id = experiment_id

                    with st.spinner("Loading model..."):
                        llm_jp, loaded_model_params = load_model(
                            selected_model, st.session_state.model_params
                        )

                    loaded_model_params["model_name"] = selected_model

                    st.session_state.llm_jp = llm_jp
                    st.session_state.model_params = loaded_model_params

                    run = mlflow.start_run(
                        experiment_id=experiment_id, run_name=run_name
                    )

                    st.session_state.mlflow_run = run
                    st.session_state.run_active = True
                    st.success(
                        f"Chat session started! Session ID: {st.session_state.sess_id}"
                    )

        # Parameter Settings form
        with st.form(key="parameter_settings_form"):
            st.subheader("Parameter Settings")
            current_params = st.session_state.model_params or {
                "max_new_tokens": 120,
                "temperature": 0.7,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "do_sample": True,
            }

            new_max_new_tokens = st.slider(
                "Max New Tokens",
                min_value=50,
                max_value=500,
                value=current_params.get("max_new_tokens", 120),
                step=10,
                help="Set the maximum number of tokens to generate.",
            )

            new_temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.5,
                value=current_params.get("temperature", 0.7),
                step=0.1,
                help="Control the randomness of generation.",
            )

            new_top_p = st.slider(
                "Top p",
                min_value=0.1,
                max_value=1.0,
                value=current_params.get("top_p", 0.95),
                step=0.05,
                help="Probability threshold for sampling tokens.",
            )

            new_repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=0.5,
                max_value=2.0,
                value=current_params.get("repetition_penalty", 1.05),
                step=0.05,
                help="Penalty for repeated tokens.",
            )

            new_do_sample = st.checkbox(
                "Do Sample",
                value=current_params.get("do_sample", True),
                help="Enable sampling for generation.",
            )

            parameter_submitted = st.form_submit_button("Update Parameters")

            if parameter_submitted:
                if st.session_state.model_params is not None:
                    st.session_state.model_params.update(
                        {
                            "max_new_tokens": new_max_new_tokens,
                            "temperature": new_temperature,
                            "top_p": new_top_p,
                            "repetition_penalty": new_repetition_penalty,
                            "do_sample": new_do_sample,
                        }
                    )
                    st.success("Parameters updated.")
                else:
                    st.session_state.model_params = {
                        "max_new_tokens": new_max_new_tokens,
                        "temperature": new_temperature,
                        "top_p": new_top_p,
                        "repetition_penalty": new_repetition_penalty,
                        "do_sample": new_do_sample,
                    }
                    st.success("Parameters set.")

        # Control Buttons
        with st.container():
            if st.button("End Chat Session"):
                mlflow.end_run()
                st.session_state.run_active = False
                st.success("Chat session ended.")

            if st.button("Reset Chat History"):
                st.session_state.chat_history = ChatMessageHistory()
                st.session_state.interaction_count = 0
                st.success("Chat history reset.")

    # Main Screen: Chat Interface
    if st.session_state.run_active:
        # initial message
        if not st.session_state.chat_history.messages:
            st.session_state.chat_history.add_ai_message("How may I assist you today?")

        user_input = st.chat_input("Type your message")
        if user_input:
            system_prompt = SYTEM_PROMPT
            # system_prompt = "You are LLM-jp. Please answer in Japanese."
            st.session_state.interaction_count += 1

            try:
                with st.spinner("Generating response..."):
                    _ = chat_with_mlflow_tracing(
                        st.session_state.llm_jp,
                        st.session_state.model_params,
                        system_prompt,
                        user_input,
                        st.session_state.chat_history,
                        st.session_state.sess_id,
                        st.session_state.interaction_count,
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")

        for msg in st.session_state.chat_history.messages:
            msg_type = msg.type if hasattr(msg, "type") else msg.get("type", "system")
            msg_content = (
                msg.content if hasattr(msg, "content") else msg.get("content", "")
            )
            with st.chat_message(msg_type):
                st.write(msg_content)

    else:
        st.info("Please start a chat session from the sidebar.")


if __name__ == "__main__":
    main()
