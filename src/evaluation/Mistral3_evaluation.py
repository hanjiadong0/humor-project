import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr

from src.humor_classifier.multitask_mistral import MistralForHumorMultiTask

@torch.no_grad()
def main(model_dir="outputs/mistral_multitask", data_path="data/processed/unified.parquet"):
    df = pd.read_parquet(data_path)
    # quick test subset
    df = df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)

    ds = Dataset.from_pandas(df[["text","y_cls","y_reg"]])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(x):
        return tokenizer(x["text"], truncation=True, max_length=256, padding="max_length")

    ds = ds.map(tok)

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    base = MistralForHumorMultiTask.from_pretrained(model_dir, device_map="auto", quantization_config=bnb_cfg)
    model = base.eval()

    # inference
    preds_cls, gold_cls = [], []
    preds_reg, gold_reg = [], []

    for ex in ds:
        input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
        attn      = torch.tensor([ex["attention_mask"]]).to(model.device)
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out["logits_cls"].float().cpu().numpy()[0]
        reg    = out["pred_reg"].float().cpu().numpy()[0]

        if ex["y_cls"] != -1:
            preds_cls.append(int(np.argmax(logits)))
            gold_cls.append(int(ex["y_cls"]))
        if ex["y_reg"] >= 0:
            preds_reg.append(float(reg))
            gold_reg.append(float(ex["y_reg"]))

    if len(gold_cls) > 0:
        print("CLS accuracy:", accuracy_score(gold_cls, preds_cls))

    if len(gold_reg) > 0:
        rmse = mean_squared_error(gold_reg, preds_reg, squared=False)
        corr = pearsonr(gold_reg, preds_reg)[0]
        print("REG RMSE:", rmse)
        print("REG Pearson:", corr)

if __name__ == "__main__":
    main()
