import argparse
import json
import time
from pathlib import Path

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
EPSILONS = [1, 2, 4]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DP-LoRA epsilon sweep on SST-2")
    p.add_argument("--method", choices=["ffa", "vanilla"], default="ffa")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON (optional)")
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
# Data (cached across runs)
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
# Eval
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
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, dp_optimizer, device, engine, epoch):
    model.train()
    total_loss = 0
    n_micro = 0

    with VirtualBatchManager(
        data_loader=train_loader,
        max_physical_batch_size=PHYSICAL_BATCH_SIZE,
        optimizer=dp_optimizer,
    ) as vb_loader:
        pbar = tqdm(vb_loader, desc=f"  Epoch {epoch}/{EPOCHS}", leave=True)
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
            n_micro += 1
            pbar.set_postfix(loss=f"{total_loss/n_micro:.4f}")

    return total_loss / max(n_micro, 1)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single_epsilon(epsilon, method, rank, train_ds, val_ds, device, tokenizer):
    """Run one complete training for a given epsilon. Returns result dict."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_name = "roberta-base"
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    )
    lora_config = LoraConfig(
        r=rank, lora_alpha=2 * rank,
        target_modules=["query", "value"],
        lora_dropout=0.0, bias="none", task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)

    train_loader = DataLoader(
        train_ds, batch_size=LOGICAL_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=PHYSICAL_BATCH_SIZE, collate_fn=collate_fn)

    N = len(train_ds)
    delta = N ** (-1.1)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    engine = DPLoRAEngine()
    model, dp_optimizer, train_loader = engine.make_private_with_epsilon(
        model=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        method=method,
        poisson_sampling=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)

    print(f"\n  Noise multiplier: {dp_optimizer.noise_multiplier:.4f}")
    print(f"  Trainable params: {trainable_params:,}")

    epoch_results = []
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, dp_optimizer, device, engine, epoch)
        val_acc = evaluate(model, val_loader, device)
        eps_spent = engine.get_epsilon()
        elapsed = time.time() - t0

        epoch_results.append({
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "epsilon_spent": round(eps_spent, 2),
            "time_s": round(elapsed),
        })
        print(f"  Epoch {epoch}/{EPOCHS}: loss={avg_loss:.4f}, "
              f"val_acc={val_acc*100:.2f}%, eps={eps_spent:.2f} ({elapsed:.0f}s)")

    total_time = time.time() - t_start
    final_acc = evaluate(model, val_loader, device)
    final_eps = engine.get_epsilon()

    return {
        "method": method,
        "rank": rank,
        "target_epsilon": epsilon,
        "delta": delta,
        "noise_multiplier": round(dp_optimizer.noise_multiplier, 4),
        "trainable_params": trainable_params,
        "final_val_acc": round(final_acc * 100, 2),
        "final_epsilon": round(final_eps, 2),
        "total_time_s": round(total_time),
        "epochs": epoch_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device(args.device)

    print("=" * 60)
    print(f"Epsilon Sweep: method={args.method}, rank={args.rank}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load tokenizer + data once
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds, val_ds = load_sst2(tokenizer)

    all_results = []

    for eps in EPSILONS:
        print(f"\n{'='*60}")
        print(f"Running eps={eps}, method={args.method}, rank={args.rank}")
        print(f"{'='*60}")

        result = run_single_epsilon(
            epsilon=eps, method=args.method, rank=args.rank,
            train_ds=train_ds, val_ds=val_ds,
            device=device, tokenizer=tokenizer,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Method: {args.method} | Rank: {args.rank} | Epochs: {EPOCHS}")
    print(f"Batch: logical={LOGICAL_BATCH_SIZE}, physical={PHYSICAL_BATCH_SIZE}")
    print(f"{'─'*60}")
    print(f"{'Epsilon':>8} {'Noise σ':>10} {'Val Acc':>10} {'Time':>10}")
    print(f"{'─'*60}")
    for r in all_results:
        mins = r['total_time_s'] // 60
        print(f"{r['target_epsilon']:>8.1f} {r['noise_multiplier']:>10.4f} "
              f"{r['final_val_acc']:>9.2f}% {mins:>8d}m")
    print(f"{'─'*60}")

    # Save JSON if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
