import shutil
import subprocess
import sys
import mlflow.pyfunc
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFTextGenModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_repo=None):
        self.model_repo = model_repo
        self.model = None
        self.tokenizer = None

    def load_context(self, context):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_repo).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo)

    def predict(self, context, model_input):
        prompts = self.tokenizer.apply_chat_template(model_input, tokenize=False)
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.05,
            )[0]

        generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return ["".join(generated_text)]


def save_model(save_dir, model_repo):
    shutil.rmtree(save_dir, ignore_errors=True)  # remove save path dir
    mlflow.pyfunc.save_model(
        path=save_dir,
        python_model=HFTextGenModel(model_repo),
        conda_env=None,
    )
    logger.success(f"saved model: {model_repo} -> {save_dir}")
    return save_dir


def serve_model(model_path, port=5001):
    logger.debug(f"Launching MLflow server: Port {port}")
    cmd = f"mlflow models serve -m {model_path} --no-conda --port {port}"

    try:
        process = subprocess.Popen(cmd, shell=True)
        logger.info(
            f"Launched MLflow server (URL: http://localhost:{port}/invocations)"
        )
        process.wait()

    except KeyboardInterrupt:
        process.terminate()
        logger.warning("MLflow server terminated")


if __name__ == "__main__":
    port = 5001
    save_dir = "./saved_model"
    model_repo = "llm-jp/llm-jp-3-3.7b-instruct"

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            print(f"Use default port: {port}")

    model_path = save_model(save_dir, model_repo)
    serve_model(model_path, port)
