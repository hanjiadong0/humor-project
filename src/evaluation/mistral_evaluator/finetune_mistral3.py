# Generated from: finetune_mistral3.ipynb
# Converted at: 2026-02-08T21:04:35.214Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# === 0) Install (wie im Notebook) ===
# %pip install -q accelerate peft bitsandbytes transformers datasets evaluate scikit-learn
# %pip install transformers==5.0.0rc0
# %pip install mistral-common --upgrade
import os, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, root_mean_squared_error, r2_score


# Model + dataset IDs


base_model_id = "mistralai/Ministral-3-8B-Base-2512"

# Your dataset must contain:
# - "text" (string)
# - "label_cls" (int class id, e.g. 0/1 or 0..K-1)
# - "label_reg" (float, e.g. funniness score)
dataset_path = "YOUR_DATASET_PATH_OR_HF_DATASET_NAME"

num_labels = 2          # <-- set your number of classes
loss_w_reg = 1.0        # <-- weight for regression loss
max_length = 512        # keep smaller first (faster)


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

backbone = AutoModel.from_pretrained(
    base_model_id,
    quantization_config=nf4_config,
    device_map="auto",
    trust_remote_code=True,
)

backbone = prepare_model_for_kbit_training(backbone)


# === 3) Load model backbone in 4-bit (QLoRA wie im Notebook) ===
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)

backbone = AutoModel.from_pretrained(
    base_model_id,
    quantization_config=nf4_config,
    device_map="auto"
)

backbone = prepare_model_for_kbit_training(backbone)


def get_text_config(cfg):
    # Mistral3Config: wrapper => text_config contains the LM config
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        return cfg.text_config
    return cfg

def get_hidden_size(cfg) -> int:
    tc = get_text_config(cfg)

    # tc can be a config object OR a dict
    if isinstance(tc, dict):
        for k in ["hidden_size", "dim", "d_model", "model_dim"]:
            if k in tc and tc[k] is not None:
                return int(tc[k])
        raise ValueError(f"hidden size not found in text_config dict keys={list(tc.keys())}")

    # config object
    for attr in ["hidden_size", "dim", "d_model", "model_dim"]:
        if hasattr(tc, attr) and getattr(tc, attr) is not None:
            return int(getattr(tc, attr))

    # last fallback
    if hasattr(tc, "to_dict"):
        d = tc.to_dict()
        for k in ["hidden_size", "dim", "d_model", "model_dim"]:
            if k in d and d[k] is not None:
                return int(d[k])

    raise ValueError("Could not find hidden size in config / text_config.")

def get_text_backbone(model):
    """
    Only unwrap if this is a multimodal wrapper (has config.text_config).
    Prefer explicit text submodules when present.
    Otherwise return the model itself (safe for AutoModel / CausalLM etc).
    """
    cfg = getattr(model, "config", None)
    is_wrapper = (cfg is not None) and hasattr(cfg, "text_config") and (cfg.text_config is not None)

    if not is_wrapper:
        return model

    # multimodal wrapper: try known text module names
    for name in ["text_model", "language_model"]:
        if hasattr(model, name):
            m = getattr(model, name)
            if m is not None:
                return m

    # if no explicit text submodule, fall back to model itself
    return model

def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Take hidden state of the last *real* token (not padding).
    last_hidden_state: [B, T, H]
    attention_mask:   [B, T] with 1 for tokens, 0 for padding
    """
    if attention_mask is None:
        return last_hidden_state[:, -1, :]

    lengths = attention_mask.long().sum(dim=1)  # [B]
    idx = torch.clamp(lengths - 1, min=0)       # [B]
    batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_idx, idx, :]  # [B, H]


class MultiTaskRegCls(nn.Module):
    def __init__(self, backbone, num_labels: int, loss_w_reg: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config   
        h = get_hidden_size(backbone.config)   # <-- FIX: works for Mistral3Config wrapper
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(h, num_labels)
        self.reg_head = nn.Linear(h, 1)
        self.loss_w_reg = loss_w_reg
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, y_cls=None, y_reg=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        logits_cls = self.cls_head(pooled)               # [B,C]
        pred_reg   = self.reg_head(pooled).squeeze(-1)   # [B]

        # pack outputs for metrics
        logits = torch.cat([logits_cls, pred_reg.unsqueeze(-1)], dim=1)  # [B, C+1]

        loss = None
        if (y_cls is not None) or (y_reg is not None):
            loss = 0.0
            if y_cls is not None:
                loss = loss + F.cross_entropy(logits_cls, y_cls.long())
            if y_reg is not None:
                loss = loss + self.loss_w_reg * F.mse_loss(pred_reg.float(), y_reg.float())

        return {"loss": loss, "logits": logits}

# backbone = AutoModel.from_pretrained(...)
text_backbone = get_text_backbone(backbone)

model = MultiTaskRegCls(
    text_backbone,
    num_labels=num_labels,
    loss_w_reg= 1 ,   # or your variable name
    dropout=0.1
)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

model = MultiTaskRegCls(backbone, num_labels=num_labels, loss_w_reg=loss_w_reg, dropout=0.1)
model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | {trainable/total*100:.4f}%")


import pandas as pd
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

@dataclass
class TrainingDataConfig:
    csv_path: str
    text_col: str = "text"
    cls_col_raw: str = "is_humor"
    reg_col_raw: str = "humor_rating"
    missing_cls_value: int = -1
    missing_reg_value: float = -1.0
    val_size: float = 0.1

class DataModule:
    def __init__(self, cfg: Dict[str, Any], tcfg: TrainingDataConfig):
        self.cfg = cfg
        self.tcfg = tcfg

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.tcfg.csv_path)

        # check columns exist
        needed = [self.tcfg.text_col, self.tcfg.cls_col_raw, self.tcfg.reg_col_raw]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in CSV: {missing}. Found: {df.columns.tolist()[:50]}")

        # rename to internal standard names
        df = df.rename(columns={
            self.tcfg.text_col: "text",
            self.tcfg.cls_col_raw: "y_cls",
            self.tcfg.reg_col_raw: "y_reg",
        })

        # drop empty text
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip().ne("")].reset_index(drop=True)

        # cls label (may have NaN)
        df["y_cls"] = pd.to_numeric(df["y_cls"], errors="coerce").fillna(self.tcfg.missing_cls_value).astype("int64")

        # reg label (may have NaN / strings)
        df["y_reg"] = pd.to_numeric(df["y_reg"], errors="coerce").astype("float32")

        # enforce your consistency rule:
        # - cls=0 => reg=-1
        df.loc[df["y_cls"] == 0, "y_reg"] = self.tcfg.missing_reg_value
        # - cls=1 but missing reg => -1
        df.loc[(df["y_cls"] == 1) & (df["y_reg"].isna()), "y_reg"] = self.tcfg.missing_reg_value

        # clip only valid humor ratings
        humor = df["y_reg"] >= 0
        df.loc[humor, "y_reg"] = df.loc[humor, "y_reg"].clip(0.0, 4.0)

        # final strict rule: cls derived from reg sign
        df["y_cls"] = (df["y_reg"] >= 0).astype("int64")

        return df[["text", "y_cls", "y_reg"]]

    @staticmethod
    def tokenize_fn(batch, tokenizer, max_length: int):
        # IMPORTANT: no padding here (faster). Collator will pad dynamically.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    def build_datasets(self, tokenizer) -> Tuple[Dataset, Dataset]:
        seed = int(self.cfg["project"]["seed"])
        max_len = int(self.cfg["model"]["max_length"])

        df = self.load_dataframe()
        train_df, val_df = train_test_split(df, test_size=self.tcfg.val_size, random_state=seed, shuffle=True)

        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

        train_ds = train_ds.map(lambda b: self.tokenize_fn(b, tokenizer, max_len), batched=True, remove_columns=["text"])
        val_ds   = val_ds.map(lambda b: self.tokenize_fn(b, tokenizer, max_len), batched=True, remove_columns=["text"])

        # keep labels + inputs
        cols = ["input_ids", "attention_mask", "y_cls", "y_reg"]
        train_ds = train_ds.select_columns(cols)
        val_ds   = val_ds.select_columns(cols)

        return train_ds, val_ds




def test_dataset():
    cfg: Dict[str, Any] = {
        "project": {"seed": 42},
        "model": {"max_length": 64},
    }

    tcfg = TrainingDataConfig(
        csv_path=r"C:\Users\Anwender\humor-project\data\labels\hahackathon_train.csv",
        text_col="text",
        cls_col_raw="is_humor",
        reg_col_raw="humor_rating",
        missing_cls_value=-1,
        missing_reg_value=-1.0,
        val_size=0.1,
    )

    dm = DataModule(cfg, tcfg)

    # 1) dataframe checks
    df = dm.load_dataframe()
    print(df.head(2))
    print("columns:", df.columns.tolist())
    print("y_cls unique sample:", df["y_cls"].unique()[:10])
    print("y_reg min/max:", df["y_reg"].min(), df["y_reg"].max())

    assert {"text", "y_cls", "y_reg"} <= set(df.columns)

    # property checks
    bad1 = ((df["y_cls"] == 0) & (df["y_reg"] >= 0)).sum()
    bad2 = ((df["y_cls"] == 1) & (df["y_reg"] < 0)).sum()
    print("(cls=0 but reg>=0):", bad1)
    print("(cls=1 but reg<0):", bad2)
    assert bad1 == 0
    assert bad2 == 0

    # 2) datasets
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds, val_ds = dm.build_datasets(tok)

    print("train rows:", len(train_ds), "val rows:", len(val_ds))
    sample = train_ds[0]
    print("keys:", sample.keys())

    assert set(sample.keys()) == {"input_ids", "attention_mask", "y_cls", "y_reg"}

    # NOTE: because we removed padding="max_length", lengths may vary now!
    # so don't assert fixed length here.
    assert len(sample["input_ids"]) >= 1
    assert len(sample["attention_mask"]) == len(sample["input_ids"])

if __name__ == "__main__":
    test_dataset()



def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # ---- preds: could be array, tuple, list ----
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds)

    # Expect packed logits: [N, C+1]
    if preds.ndim != 2 or preds.shape[1] < 2:
        raise ValueError(f"Unexpected preds shape: {preds.shape}. Expected [N, C+1].")

    cls_logits = preds[:, :num_labels]   # [N,C]
    reg_pred   = preds[:, -1]            # [N]

    # ---- labels: could be tuple/list (y_cls, y_reg) or array ----
    if isinstance(labels, (tuple, list)) and len(labels) == 2:
        y_cls, y_reg = labels
    else:
        labels = np.asarray(labels)
        # If Trainer stacked them as [N,2]
        if labels.ndim == 2 and labels.shape[1] == 2:
            y_cls, y_reg = labels[:, 0], labels[:, 1]
        else:
            raise ValueError(f"Unexpected labels shape/type: {type(labels)} {getattr(labels,'shape',None)}")

    y_cls = np.asarray(y_cls).astype(int)
    y_reg = np.asarray(y_reg).astype(float)

    cls_pred = cls_logits.argmax(axis=1)

    acc = accuracy_score(y_cls, cls_pred)
    f1  = f1_score(y_cls, cls_pred, average="macro")

    mae  = mean_absolute_error(y_reg, reg_pred)
    rmse = root_mean_squared_error(y_reg, reg_pred)
    r2   = r2_score(y_reg, reg_pred)

    return {"cls_acc": acc, "cls_f1": f1, "reg_mae": mae, "reg_rmse": rmse, "reg_r2": r2}


training_args = TrainingArguments(
    output_dir="./checkpoints_ministral3_multitask",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    bf16=True,
    fp16=False,
    max_grad_norm=1.0,
    warmup_steps= 136,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    dataloader_num_workers=4
)




data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

cfg: Dict[str, Any] = {
    "project": {"seed": 42},
    "model": {"max_length": 64},
}

tcfg = TrainingDataConfig(
    csv_path=r"C:\Users\Anwender\humor-project\data\labels\hahackathon_train.csv",
    text_col="text",
    cls_col_raw="is_humor",
    reg_col_raw="humor_rating",
    missing_cls_value=-1,
    missing_reg_value=-1.0,
    val_size=0.1,
)


dm = DataModule(cfg, tcfg)

train_ds, val_ds = dm.build_datasets(tokenizer)

# 1) cache off for training
model.config.use_cache =  False  

# 2) gradient checkpointing (optional)
if hasattr(text_backbone, "gradient_checkpointing_enable"):
    try:
        text_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        text_backbone.gradient_checkpointing_enable()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.label_names = ["y_cls", "y_reg"]


# trainer.train()