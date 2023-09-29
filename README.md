## Transformers for generating a text with Shakespeare style.

### Setup
```
# Clone the repo
git clone https://github.com/garipovroma/shakespeare-gen
cd shakespeare-gen

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Prepare data
python3 scripts/prepare_data.py

```

### Train model
There are a few options to train the model:
```
usage: train.py [-h] [--data_folder DATA_FOLDER] [--learning_rate LEARNING_RATE]
                [--use_scheduler USE_SCHEDULER] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                [--max_seq_len MAX_SEQ_LEN] [--device DEVICE] [--seed SEED] [--num_workers NUM_WORKERS]
                [--out_folder OUT_FOLDER] [--train_verbose TRAIN_VERBOSE] [--push_flag PUSH_FLAG]
                [--huggingface_token HUGGINGFACE_TOKEN] [--use_wandb USE_WANDB]

options:
  -h, --help            show this help message and exit
  --data_folder DATA_FOLDER
                        path to the data folder
  --learning_rate LEARNING_RATE
                        learning rate
  --use_scheduler USE_SCHEDULER
                        use scheduler flag
  --num_epochs NUM_EPOCHS
                        number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --max_seq_len MAX_SEQ_LEN
                        max sequence length
  --device DEVICE       device to run the model on
  --seed SEED           seed for random number generators
  --num_workers NUM_WORKERS
                        number of workers for data loaders
  --out_folder OUT_FOLDER
                        folder to save the model and tokenizer
  --train_verbose TRAIN_VERBOSE
                        number of batches after which to print the training loss
  --push_flag PUSH_FLAG
                        push to hub flag
  --huggingface_token HUGGINGFACE_TOKEN
                        huggingface token
  --use_wandb USE_WANDB
                        use wandb flag
```

Example:
``` 
python3 scripts/train.py --use_wandb True
```
Output:
```
python3 scripts/train.py --use_wandb True
wandb: Currently logged in as: garipovroma. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.15.11
wandb: Run data is saved locally in /Users/garipovroma/work/omnicode-proj/wandb/run-20230929_224515-49p8kko3
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run gpt2
wandb: ⭐️ View project at https://wandb.ai/garipovroma/shakespeare-transformers
wandb: 🚀 View run at https://wandb.ai/garipovroma/shakespeare-transformers/runs/49p8kko3
Tokenizing data/train_df.csv: 100%|█████████████████████████████████████████| 100/100 [00:00<00:00, 1823.98it/s]
Tokenizing data/val_df.csv: 100%|███████████████████████████████████████████| 100/100 [00:00<00:00, 2180.22it/s]
Epoch 1: 100%|██████████████████████████████████████████████████████████████████| 13/13 [01:43<00:00,  8.00s/it]
Epoch [1/1] - Training Loss: 10.9173 - Perplexity: 55123.5162
Evaluation after epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:28<00:00,  2.21s/it]
Epoch [1/1] - {'train_loss': 10.91733169555664, 'train_perplexity': 55123.5162097329, 'val_loss': 9.00640715085543, 'val_perplexity': 8155.168286687939}
wandb: Waiting for W&B process to finish... (success).
wandb:
wandb: Run history:
wandb:       train_loss ▁
wandb: train_perplexity ▁
wandb:         val_loss ▁
wandb:   val_perplexity ▁
wandb:
wandb: Run summary:
wandb:       train_loss 10.91733
wandb: train_perplexity 55123.51621
wandb:         val_loss 9.00641
wandb:   val_perplexity 8155.16829
wandb:
wandb: 🚀 View run gpt2 at: https://wandb.ai/garipovroma/shakespeare-transformers/runs/49p8kko3
wandb: ️⚡ View job at https://wandb.ai/garipovroma/shakespeare-transformers/jobs/QXJ0aWZhY3RDb2xsZWN0aW9uOjEwMjUwMjEwOA==/version_details/v1
wandb: Synced 6 W&B file(s), 0 media file(s), 2 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20230929_224515-49p8kko3/logs
```

# Generate text

```