
from __future__ import absolute_import, division, print_function

import argparse
import gc
import json
import logging
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

from data_structured_evidence import (
    HiDialogDataloaderStructuredEvidence,
    HiDialogDatasetStructuredEvidence,
)
from models.RoBERTa.HiDialog_RoBERTa_SEA import HiDialog_RoBERTa_SEA
from models.RoBERTa.configuration_roberta import RobertaConfig
from models.RoBERTa.tokenization_roberta import RobertaTokenizer
from dialogre.evaluate import f1_eval


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

n_classes = {"DialogRE": 36}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_cache_file(data_dir, split_name, encoder_type, evidence_file, max_seq_length, max_support_turns):
    evidence_tag = "none"
    if evidence_file:
        evidence_tag = os.path.splitext(os.path.basename(evidence_file))[0]
    cache_name = f"{split_name}_{encoder_type}_sea_{evidence_tag}_ms{max_support_turns}_seq{max_seq_length}.pkl"
    return os.path.join(data_dir, cache_name)


def forward_model(model, batch, device, with_labels=True, output_attentions=False):
    return model(
        input_ids=batch["input_ids"].to(device),
        token_type_ids=batch["segment_ids"].to(device),
        attention_mask=batch["input_masks"].to(device),
        speaker_ids=batch["speaker_ids"].to(device),
        graphs=batch["graphs"],
        labels=batch["label_ids"].to(device) if with_labels else None,
        output_attentions=output_attentions,
        cls_indices=batch["cls_indices"].to(device),
        hidialog_mask=batch["hidialog_masks"].to(device),
        turn_ids=batch["turn_ids"].to(device),
        evidence_turn_mask=batch["evidence_turn_mask"].to(device),
        evidence_turn_scores=batch["evidence_turn_scores"].to(device),
        relation_prior=batch["relation_prior"].to(device),
        evidence_global_confidence=batch["evidence_global_confidence"].to(device),
        evidence_valid=batch["evidence_valid"].to(device),
    )


def evaluate_dialogre(model, dataloader, device):
    model.eval()
    logits_all = []
    labels_all = []
    losses = []
    consistency_losses = []
    residual_gates = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            loss, logits, extras = forward_model(model, batch, device, with_labels=True, output_attentions=False)
            losses.append(loss.mean().item() if loss is not None else 0.0)
            consistency_losses.append(float(extras["consistency_loss"].detach().cpu().item()))
            residual_gates.extend([float(g.detach().cpu().item()) for g in extras.get("residual_gates", [])])

            logits_np = logits.detach().cpu().numpy()
            labels_np = batch["label_ids"].cpu().numpy()
            logits_all.extend(list(logits_np))
            labels_all.extend(list(labels_np))

    eval_f1, eval_t2 = f1_eval(logits_all, labels_all)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "f1": eval_f1,
        "T2": eval_t2,
        "consistency_loss": float(np.mean(consistency_losses)) if consistency_losses else 0.0,
        "mean_residual_gate": float(np.mean(residual_gates)) if residual_gates else 0.0,
    }


def write_logits(model, dataloader, device, savefile):
    model.eval()
    logits_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Write {os.path.basename(savefile)}"):
            logits, _, _ = forward_model(model, batch, device, with_labels=False, output_attentions=False)
            logits_np = logits.detach().cpu().numpy()
            logits_all.extend(list(logits_np))

    with open(savefile, "w", encoding="utf-8") as f:
        for row in logits_all:
            f.write(" ".join(str(x) for x in row) + "\n")


def build_dataset(args, tokenizer, n_class, split_name):
    cache_file = make_cache_file(
        args.data_dir,
        split_name,
        args.encoder_type,
        args.evidence_file,
        args.max_seq_length,
        args.max_support_turns,
    )
    return HiDialogDatasetStructuredEvidence(
        src_file=args.data_dir,
        save_file=cache_file,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        n_class=n_class,
        encoder_type=args.encoder_type,
        dataset=args.data_name,
        split_name=split_name,
        evidence_cache_path=args.evidence_file,
        max_support_turns=args.max_support_turns,
        overwrite_cache=args.overwrite_cache,
    )


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def build_scheduler(optimizer, warmup_steps, total_steps):
    if total_steps <= 0:
        return LambdaLR(optimizer, lambda _: 1.0)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def load_state_partial(model, checkpoint_path, description):
    if not checkpoint_path:
        return
    if os.path.isdir(checkpoint_path):
        candidate = os.path.join(checkpoint_path, "pytorch_model.bin")
        checkpoint_path = candidate if os.path.isfile(candidate) else None
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        logger.warning("%s checkpoint not found: %s", description, checkpoint_path)
        return
    logger.info("Loading %s checkpoint from %s", description, checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    load_dict = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            load_dict[key] = value
        elif (
            value.dim() == 2
            and value.shape[1] == target.shape[1]
            and target.shape[0] > value.shape[0]
        ):
            merged = target.clone()
            merged[: value.shape[0]] = value
            load_dict[key] = merged
            logger.info("Partial-load %s (embedding %s -> %s)", key, tuple(value.shape), tuple(target.shape))
        else:
            skipped.append(key)
    if skipped:
        logger.info("Skipped %d shape-mismatched keys in %s checkpoint: %s",
                    len(skipped), description, skipped[:8])
    if load_dict:
        model.load_state_dict(load_dict, strict=False)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str, help="Path to the dataset directory")
    parser.add_argument("--data_name", required=True, type=str, default="DialogRE", help="Dataset name")
    parser.add_argument("--encoder_type", required=True, type=str, default="RoBERTa", help="Encoder type")
    parser.add_argument("--config_file", required=True, type=str, help="Path to the RoBERTa config file")
    parser.add_argument("--vocab_file", required=True, type=str, help="Path to the vocab file")
    parser.add_argument("--merges_file", required=True, type=str, help="Path to the merges file")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation")
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="Backbone learning rate")
    parser.add_argument("--adapter_learning_rate", default=1.5e-5, type=float, help="Adapter learning rate")
    parser.add_argument("--num_train_epochs", default=5.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--init_checkpoint", default=None, type=str, help="Path to the initial checkpoint")
    parser.add_argument("--base_checkpoint", default=None, type=str, help="Path to the base backbone checkpoint")
    parser.add_argument("--gtn_num_layers", default=2, type=int)
    parser.add_argument("--gtn_num_channels", default=3, type=int)

    parser.add_argument("--evidence_file", type=str, default=None, help="Path to the structured evidence file")
    parser.add_argument("--max_support_turns", type=int, default=6, help="Maximum number of supporting turns")
    parser.add_argument("--evidence_bias_scale", type=float, default=0.7, help="Scale of the evidence bias")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.02, help="Weight of the consistency loss")
    parser.add_argument("--relation_prior_weight", type=float, default=0.05, help="Weight of the relation prior")
    parser.add_argument("--no_lambda_schedule", action="store_true", help="Disable the epoch-wise lambda schedule (default: +0.01 every 2 epochs)")
    parser.add_argument("--lambda_schedule_interval", type=int, default=2, help="Lambda schedule interval in epochs")
    parser.add_argument("--lambda_schedule_increment", type=float, default=0.01, help="Lambda schedule increment per interval")
    parser.add_argument("--no_structured_evidence", action="store_true", help="Disable structured evidence")
    parser.add_argument("--no_confidence_gate", action="store_true", help="Disable the confidence gate")
    parser.add_argument("--no_consistency_loss", action="store_true", help="Disable the consistency loss")
    parser.add_argument("--no_relation_prior", action="store_true", help="Disable the relation prior")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite cached features")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    time.sleep(args.sleep)

    if args.data_name not in n_classes:
        raise ValueError(f"Unsupported dataset: {args.data_name}")
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    n_class = n_classes[args.data_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    set_seed(args.seed)

    config = RobertaConfig.from_json_file(args.config_file)
    tokenizer = RobertaTokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
    os.makedirs(args.output_dir, exist_ok=True)

    train_set = None
    train_loader = None
    num_train_steps = 0

    if args.do_train:
        train_set = build_dataset(args, tokenizer, n_class, "train")
        effective_train_batch = max(1, int(args.train_batch_size / args.gradient_accumulation_steps))
        train_loader = HiDialogDataloaderStructuredEvidence(
            dataset=train_set,
            batch_size=effective_train_batch,
            shuffle=True,
            relation_num=n_class,
            max_length=args.max_seq_length,
            dataset_name=args.data_name,
        )
        num_train_steps = int(
            len(train_set) / effective_train_batch / args.gradient_accumulation_steps * args.num_train_epochs
        )

    dev_set = build_dataset(args, tokenizer, n_class, "dev")
    test_set = build_dataset(args, tokenizer, n_class, "test")
    dev_loader = HiDialogDataloaderStructuredEvidence(
        dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False,
        relation_num=n_class, max_length=args.max_seq_length, dataset_name=args.data_name,
    )
    test_loader = HiDialogDataloaderStructuredEvidence(
        dataset=test_set, batch_size=args.eval_batch_size, shuffle=False,
        relation_num=n_class, max_length=args.max_seq_length, dataset_name=args.data_name,
    )

    model = HiDialog_RoBERTa_SEA(
        config=config,
        num_labels=n_class,
        gtn_out=1024,
        activation="relu",
        gtn_num_layers=args.gtn_num_layers,
        gtn_num_channels=args.gtn_num_channels,
        dataset=args.data_name,
        evidence_bias_scale=args.evidence_bias_scale,
        consistency_loss_weight=args.consistency_loss_weight,
        relation_prior_weight=args.relation_prior_weight,
        use_structured_evidence=not args.no_structured_evidence,
        use_confidence_gate=not args.no_confidence_gate,
        use_consistency_loss=not args.no_consistency_loss,
        use_relation_prior=not args.no_relation_prior,
    )

    load_state_partial(model, args.init_checkpoint, "init")
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else tokenizer.get_vocab_size()
    model.roberta.resize_token_embeddings(vocab_size)
    load_state_partial(model, args.base_checkpoint, "base")
    model.to(device)

    adapter_keywords = ["evidence_adapter", "relation_prior_gate"]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    named_params = list(model.named_parameters())

    backbone_decay = [p for n, p in named_params if not any(k in n for k in adapter_keywords) and not any(nd in n for nd in no_decay)]
    backbone_no_decay = [p for n, p in named_params if not any(k in n for k in adapter_keywords) and any(nd in n for nd in no_decay)]
    adapter_decay = [p for n, p in named_params if any(k in n for k in adapter_keywords) and not any(nd in n for nd in no_decay)]
    adapter_no_decay = [p for n, p in named_params if any(k in n for k in adapter_keywords) and any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {"params": backbone_decay, "weight_decay": args.weight_decay, "lr": args.learning_rate},
        {"params": backbone_no_decay, "weight_decay": 0.0, "lr": args.learning_rate},
        {"params": adapter_decay, "weight_decay": args.weight_decay, "lr": args.adapter_learning_rate},
        {"params": adapter_no_decay, "weight_decay": 0.0, "lr": args.adapter_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    warmup_steps = int(args.warmup_proportion * num_train_steps)
    scheduler = build_scheduler(optimizer, warmup_steps, num_train_steps)

    best_metric = -1.0
    best_result = None

    if args.do_train:
        logger.info("***** Running SEA training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        global_step = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            if not args.no_lambda_schedule:
                inc = args.lambda_schedule_increment * (epoch // max(1, args.lambda_schedule_interval))
                if model.use_consistency_loss:
                    model.consistency_loss_weight = args.consistency_loss_weight + inc
                if model.use_relation_prior:
                    model.relation_prior_weight = args.relation_prior_weight + inc
                if epoch > 0 and epoch % args.lambda_schedule_interval == 0:
                    logger.info(
                        "Epoch %d: lambda_cons=%.3f, lambda_r=%.3f",
                        epoch, model.consistency_loss_weight, model.relation_prior_weight,
                    )
            model.train()
            tr_loss = 0.0
            nb_tr_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                loss, _, _ = forward_model(model, batch, device, with_labels=True, output_attentions=False)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            dev_result = evaluate_dialogre(model, dev_loader, device)
            test_result = evaluate_dialogre(model, test_loader, device)

            epoch_summary = {
                "epoch": epoch,
                "lambda_cons": model.consistency_loss_weight,
                "lambda_r": model.relation_prior_weight,
                "train_loss": tr_loss / max(nb_tr_steps, 1),
                "dev": dev_result,
                "test": test_result,
                "global_step": global_step,
            }
            logger.info("Epoch summary: %s", json.dumps(epoch_summary, ensure_ascii=False))

            if dev_result["f1"] >= best_metric:
                best_metric = dev_result["f1"]
                best_result = epoch_summary
                torch.save(unwrap_model(model).state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                save_json(os.path.join(args.output_dir, "best_dev_summary.json"), epoch_summary)

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt"), map_location=device))
        torch.save(unwrap_model(model).state_dict(), os.path.join(args.output_dir, "model.pt"))

    if os.path.isfile(os.path.join(args.output_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt"), map_location=device))

    final_dev = evaluate_dialogre(model, dev_loader, device)
    final_test = evaluate_dialogre(model, test_loader, device)
    eval_summary = {
        "dev": final_dev,
        "test": final_test,
        "best_epoch_summary": best_result,
        "args": vars(args),
    }
    save_json(os.path.join(args.output_dir, "eval_summary.json"), eval_summary)

    if args.do_eval:
        del dev_loader, test_loader, dev_set, test_set
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for split_name, output_name in [
            ("dev", "logits_dev.txt"),
            ("test", "logits_test.txt"),
        ]:
            split_set = build_dataset(args, tokenizer, n_class, split_name)
            split_loader = HiDialogDataloaderStructuredEvidence(
                dataset=split_set, batch_size=args.eval_batch_size, shuffle=False,
                relation_num=n_class, max_length=args.max_seq_length, dataset_name=args.data_name,
            )
            write_logits(model, split_loader, device, os.path.join(args.output_dir, output_name))
            del split_loader, split_set
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
