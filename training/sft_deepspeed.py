"""Full-parameter SFT of Qwen3-8B on agent-turn samples with DeepSpeed ZeRO-3.

Each line of the data file is one assistant turn rendered by training/build_sft_data.py:
{"prompt_text": <chat-template text up to the assistant turn>, "response_text": <the turn>}.
Loss is computed on the response tokens only. A plain DeepSpeed loop is used instead of the
HF Trainer because the shared env pairs transformers 4.57 with accelerate 0.34, which the
Trainer does not support.

The loss is a per-token average over the global batch, not a per-sample average: each
micro-batch returns sum(token losses) / C with C the mean number of target tokens per sample,
and DeepSpeed averages micro-batches and ranks. A per-sample mean gives a 30-token `ls` turn
the same weight as a 1000-token `write_file` turn, so the single tokens that choose the action
weigh ~30x more in short turns; the first model trained that way learned to avoid writing code.

Launch (8 GPUs): bash training/run_sft_deepspeed.sh <data.jsonl> <output-dir>
"""

import argparse
import json
import math
import os
import time

import deepspeed
import torch
import torch.distributed as dist
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          set_seed)

IGNORE_INDEX = -100


class TurnDataset(torch.utils.data.Dataset):
    """Tokenised (prompt, response) pairs; prompt tokens are masked out of the loss."""

    def __init__(self, path, tokenizer, max_len):
        self.items, self.dropped = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt = tokenizer(row["prompt_text"], add_special_tokens=False)["input_ids"]
                response = tokenizer(row["response_text"], add_special_tokens=False)["input_ids"]
                if len(prompt) + len(response) > max_len or not response:
                    self.dropped += 1
                    continue
                self.items.append((prompt, response))

    def __len__(self):
        return len(self.items)

    def mean_target_tokens(self):
        return sum(len(r) for _p, r in self.items) / max(1, len(self.items))

    def __getitem__(self, i):
        prompt, response = self.items[i]
        ids = torch.tensor(prompt + response)
        labels = torch.tensor([IGNORE_INDEX] * len(prompt) + response)
        return {"input_ids": ids[None], "labels": labels[None]}


class ResponseOnlyLoss(torch.nn.Module):
    """Causal-LM loss that projects only the hidden states of target positions to the vocabulary.

    The stock forward builds float32 logits for every position (seq_len x 151936), which for a
    30k-token prompt is ~18 GB and caused the OOM of the first long-context run. The loss only
    needs the positions whose next token is a label, so the result is identical.
    """

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, labels, norm):
        hidden = self.lm.model(input_ids=input_ids).last_hidden_state[0, :-1]
        targets = labels[0, 1:]
        keep = targets != IGNORE_INDEX
        logits = self.lm.lm_head(hidden[keep]).float()
        return torch.nn.functional.cross_entropy(logits, targets[keep], reduction="sum") / norm


def lr_lambda(total_steps, warmup_steps):
    """Linear warm-up, then cosine decay to 0 (the schedule used in the training plan)."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def ds_config(path, accum, world, lr):
    cfg = json.load(open(path))
    cfg.update(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=accum,
               train_batch_size=accum * world)
    cfg["optimizer"] = {"type": "AdamW", "params": {"lr": lr, "weight_decay": 0.0,
                                                    "betas": [0.9, 0.95], "eps": 1e-8}}
    return cfg


def convert_to_hf(consolidated, base_model, out):
    """Write a 16-bit state dict as a Hugging Face (safetensors) checkpoint with the base
    model's config, generation config and tokenizer."""
    state = torch.load(consolidated, map_location="cpu", weights_only=True)
    state = {k.removeprefix("lm."): v for k, v in state.items()}  # ResponseOnlyLoss wrapper
    config = AutoConfig.from_pretrained(base_model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    GenerationConfig.from_pretrained(base_model).save_pretrained(out)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)


def save_hf(engine, base_model, out):
    """Gather the ZeRO-3 shards on rank 0 and convert them to a Hugging Face checkpoint."""
    engine.save_16bit_model(out, "consolidated.bin")
    dist.barrier()
    if dist.get_rank() == 0:
        consolidated = os.path.join(out, "consolidated.bin")
        convert_to_hf(consolidated, base_model, out)
        os.remove(consolidated)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deepspeed", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--no-save", action="store_true", help="smoke tests only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    deepspeed.init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    set_seed(args.seed)
    accum = max(1, args.global_batch // world)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = TurnDataset(args.data, tokenizer, args.max_len)
    sampler = torch.utils.data.distributed.DistributedSampler(data, world, rank, shuffle=True,
                                                              seed=args.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=None, sampler=sampler, num_workers=2)
    steps_per_epoch = len(loader) // accum
    total = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    warmup = max(1, round(total * args.warmup_ratio))
    # Token-level normaliser (same on every rank, so the effective loss is a token average)
    norm = data.mean_target_tokens()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ResponseOnlyLoss(model)
    # The scheduler is stepped by DeepSpeed at every optimizer step (accumulation boundary)
    engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config=ds_config(args.deepspeed, accum, world, args.lr),
        lr_scheduler=lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, warmup)))
    if rank == 0:
        print(json.dumps({"samples": len(data), "dropped_too_long": data.dropped, "world": world,
                          "grad_accum": accum, "global_batch": accum * world,
                          "optimizer_steps": total, "warmup_steps": warmup,
                          "loss_normaliser_tokens": round(norm, 1)}), flush=True)

    history, step, start = [], 0, time.time()
    window_loss, window_tokens = 0.0, 0
    engine.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            # sum of token losses / norm (same scaling as transformers' num_items_in_batch)
            loss = engine(batch["input_ids"], batch["labels"], norm)
            engine.backward(loss)
            boundary = engine.is_gradient_accumulation_boundary()
            engine.step()
            window_loss += loss.item() * norm  # back to a token-loss sum for logging
            window_tokens += int((batch["labels"] != IGNORE_INDEX).sum())
            if not boundary:
                continue
            step += 1
            stats = torch.tensor([window_loss, window_tokens], device="cuda")
            dist.all_reduce(stats)
            window_loss, window_tokens = 0.0, 0
            if rank == 0:
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                       "loss": round(stats[0].item() / stats[1].item(), 4),  # per target token
                       "lr": engine.get_lr()[0], "target_tokens": int(stats[1].item()),
                       "elapsed_s": round(time.time() - start, 1)}
                history.append(rec)
                print("train_step " + json.dumps(rec), flush=True)
            if step >= total:
                break
        if step >= total:
            break

    if not args.no_save:
        save_hf(engine, args.model, args.out)
    if rank == 0:
        with open(os.path.join(args.out, "log_history.json"), "w") as f:
            json.dump(history, f, indent=1)


if __name__ == "__main__":
    main()
