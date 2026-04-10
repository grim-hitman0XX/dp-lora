import argparse
import time

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

from dp_lora import DPLoRAEngine
from dp_lora.data.virtual_batch import VirtualBatchManager

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------
LOGICAL_BATCH_SIZE = 256
PHYSICAL_BATCH_SIZE = 32
EPOCHS = 3
LR = 5e-4
MAX_GRAD_NORM = 1.0
MAX_SEQ_LENGTH = 128
SEED = 42

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DP-LoRA SST-2 benchmark")
    p.add_argument("--method", choices=["ffa", "vanilla", "none"], default="ffa",
                    help="DP method: ffa, vanilla, or none (no DP baseline)")
    p.add_argument("--epsilon", type=float, default=8.0,
                    help="Target privacy budget epsilon (ignored if method=none)")
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--device", type=str, default=None,
                    help="Device: 'mps', 'cpu', or 'cuda'. Auto-detected if omitted.")
    return p.parse_args()


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sst2(tokenizer):
    dataset = load_dataset("glue", "sst2")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    def tokenize(batch):
        return tokenizer(
            batch["sentence"], padding="max_length",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    return train_ds, val_ds


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    correct = total = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])
    return correct / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_dp(model, train_loader, dp_optimizer, device, engine, epoch, epochs):
    """DP training with virtual batching."""
    model.train()
    total_loss = 0
    n_micro_batches = 0

    with VirtualBatchManager(
        data_loader=train_loader,
        max_physical_batch_size=PHYSICAL_BATCH_SIZE,
        optimizer=dp_optimizer,
    ) as vb_loader:
        pbar = tqdm(vb_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            dp_optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()

            per_sample_grads = engine.grad_sample_module.get_per_sample_grads()
            dp_optimizer.step(per_sample_grads)
            engine.grad_sample_module.clear_per_sample_grads()

            total_loss += loss.item()
            n_micro_batches += 1
            pbar.set_postfix(loss=f"{total_loss/n_micro_batches:.4f}")

    return total_loss / max(n_micro_batches, 1)


def train_one_epoch_nodp(model, train_loader, optimizer, device, epoch, epochs):
    """Standard (non-DP) training."""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = get_device(args.device)
    use_dp = args.method != "none"

    print("=" * 50)
    print("DP-LoRA Experiment")
    print("=" * 50)

    # Load model + tokenizer
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        target_modules=["query", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load data
    train_ds, val_ds = load_sst2(tokenizer)
    # DP uses logical batch (256) split into physical micro-batches (32).
    # No-DP baseline uses physical batch directly (no virtual batching needed).
    batch_size = LOGICAL_BATCH_SIZE if use_dp else PHYSICAL_BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, collate_fn=collate_fn)

    N = len(train_ds)
    delta = N ** (-1.1)

    # Set up DP (or not)
    engine = DPLoRAEngine()
    dp_optimizer = None 

    if use_dp:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
        model, dp_optimizer, train_loader = engine.make_private_with_epsilon(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.epsilon,
            target_delta=delta,
            epochs=EPOCHS,
            max_grad_norm=MAX_GRAD_NORM,
            method=args.method,
            poisson_sampling=True,
        )
        # Recount trainable after FFA may have frozen lora_A
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
        engine = None

    model.to(device)

    print(f"Model: {model_name}")
    print(f"Dataset: SST-2 ({N} train / {len(val_ds)} val)")
    if use_dp:
        print(f"Method: {args.method} | Rank: {args.rank} | "
              f"Target eps: {args.epsilon} | delta: {delta:.2e}")
        print(f"Noise multiplier: {dp_optimizer.noise_multiplier:.4f}")
        print(f"Batch: logical={LOGICAL_BATCH_SIZE}, physical={PHYSICAL_BATCH_SIZE}")
    else:
        print(f"Method: none (no DP) | Rank: {args.rank}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of {total_params:,})")
    print("-" * 50)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        if use_dp:
            avg_loss = train_one_epoch_dp(model, train_loader, dp_optimizer, device, engine, epoch, EPOCHS)
        else:
            avg_loss = train_one_epoch_nodp(model, train_loader, optimizer, device, epoch, EPOCHS)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        if use_dp:
            eps = engine.get_epsilon()
            print(f"Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f}, "
                  f"val_acc={100*val_acc:.2f}%, eps={eps:.2f} "
                  f"({elapsed:.0f}s)")
        else:
            print(f"Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f}, "
                  f"val_acc={100*val_acc:.2f}% ({elapsed:.0f}s)")

    print("-" * 50)
    final_acc = evaluate(model, val_loader, device)
    if use_dp:
        final_eps = engine.get_epsilon()
        print(f"Final: val_acc={100*final_acc:.2f}%, "
              f"eps={final_eps:.2f}, delta={delta:.2e}")
    else:
        print(f"Final: val_acc={100*final_acc:.2f}% (no DP)")


if __name__ == "__main__":
    main()
