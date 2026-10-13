# RoBERTa-base

Download `roberta-base` from Hugging Face:

- https://huggingface.co/FacebookAI/roberta-base

Place the following files under this directory:

```
pretrained/roberta-base/config.json
pretrained/roberta-base/vocab.json
pretrained/roberta-base/merges.txt
pretrained/roberta-base/pytorch_model.bin
```

The `pytorch_model.bin` file is optional but required to initialize the
backbone encoder when training the SEA adapter (`--base_checkpoint`).

You can download the full directory with:

```bash
python -m transformers.cli.download FacebookAI/roberta-base --local-dir pretrained/roberta-base
```

or manually download the files listed above.
