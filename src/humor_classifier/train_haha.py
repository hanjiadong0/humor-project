import numpy as np
import torch
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.common.utils import load_config, load_env
from src.humor_classifier.multitask_mistral import MistralForHumorMultiTask



def tokenize_fn(ex, tokenizer, max_length):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

class MultiTaskTrainer(Trainer):
    def __init__(self, *args, cls_weight=1.0, reg_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
        self.mse = torch.nn.MSELoss(reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        y_cls = inputs.pop("y_cls")
        y_reg = inputs.pop("y_reg")

        outputs = model(**inputs)
        logits_cls = outputs["logits_cls"]
        pred_reg   = outputs["pred_reg"]

        cls_mask = (y_cls != -1)
        reg_mask = (y_reg >= 0)

        loss = 0.0

        if cls_mask.any():
            cls_loss = self.ce(logits_cls[cls_mask], y_cls[cls_mask]).mean()
            loss = loss + self.cls_weight * cls_loss

        if reg_mask.any():
            pr = pred_reg[reg_mask].view(-1).float()
            yr = y_reg[reg_mask].view(-1).float()
            reg_loss = self.mse(pr, yr).mean()
            loss = loss + self.reg_weight * reg_loss

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    # We compute metrics inside evaluate.py typically,
    # but here's a minimal example.
    return {}

def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    env = load_env()
    model_name = cfg["model"]["base_model"]

    df = pd.read_csv(r"C:\Users\Anwender\humor-project\data\labels\hahackathon_train.csv")

    df = df.rename(columns={"is_humor": "y_cls", "humor_rating": "y_reg"})

    # 1) Drop rows where text missing
    df = df.dropna(subset=["text"])

    # 2) y_cls: if missing -> -1 (mask)
    df["y_cls"] = df["y_cls"].fillna(-1).astype(int)

    # 3) y_reg: if missing -> -1.0 (mask)
    df["y_reg"] = df["y_reg"].fillna(-1.0).astype(float)

    # Ensure plain scalar floats/ints (no lists)
    df["y_reg"] = df["y_reg"].astype(float)
    df["y_cls"] = df["y_cls"].astype(int)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=cfg["project"]["seed"])

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=env["HF_TOKEN"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = cfg["model"]["max_length"]
    train_ds = train_ds.map(lambda x: tokenize_fn(x, tokenizer, max_len), batched=True)
    val_ds   = val_ds.map(lambda x: tokenize_fn(x, tokenizer, max_len), batched=True)

    cols = ["input_ids", "attention_mask", "y_cls", "y_reg"]
    train_ds = train_ds.select_columns(cols)
    val_ds   = val_ds.select_columns(cols)


    use_4bit = False  # <- set True only if bitsandbytes works

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_cfg = None


    kwargs = {}
    if bnb_cfg is not None:
        kwargs["quantization_config"] = bnb_cfg
        kwargs["device_map"] = "auto"   # nur bei 4-bit/quant

    model = MistralForHumorMultiTask.from_pretrained(
        model_name,
        token=env["HF_TOKEN"],
        torch_dtype=torch.float16,   # wichtig für Mistral
        **kwargs,
    )

    if bnb_cfg is None:
        model = model.to("cuda")

    if bnb_cfg is not None:
        model = prepare_model_for_kbit_training(model)

    if cfg["lora"]["enabled"]:
        lora_cfg = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=cfg["project"]["output_dir"],

        num_train_epochs=float(cfg["train"]["num_train_epochs"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        max_grad_norm= 1.0,

        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["train"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),

        warmup_ratio=float(cfg["train"]["warmup_ratio"]),
        weight_decay=float(cfg["train"]["weight_decay"]),

        logging_steps=int(cfg["train"]["logging_steps"]),
        logging_strategy="epoch",          # ✅ print pro epoch
        eval_strategy="no",               # ✅ erstmal aus (schneller)
        save_strategy="no",               # ✅ erstmal aus (schneller)

        # diese beiden sind dann egal, aber müssen int sein falls gesetzt
        eval_steps=int(cfg["train"]["eval_steps"]),
        save_steps=int(cfg["train"]["save_steps"]),

        fp16=bool(cfg["train"]["fp16"]),
        bf16=bool(cfg["train"]["bf16"]),

        report_to=[],
        remove_unused_columns=False,
     )



    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        cls_weight=cfg["loss"]["cls_weight"],
        reg_weight=cfg["loss"]["reg_weight"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg["project"]["output_dir"])
    tokenizer.save_pretrained(cfg["project"]["output_dir"])

if __name__ == "__main__":
    main()
