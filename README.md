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

``` 
python3 scripts/train.py
```

# Generate text

For generating text you can use the following script:

```
usage: generate.py [-h] [--load_mode {huggingface,torch}] --model MODEL --tokenizer TOKENIZER [--device DEVICE] --prompt PROMPT [--out_prompts OUT_PROMPTS] [--result_length RESULT_LENGTH]
                   [--top_p TOP_P] [--temperature TEMPERATURE] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --load_mode {huggingface,torch}
                        where from to load the model and tokenizer
  --model MODEL         path to the model or the model name on huggingface
  --tokenizer TOKENIZER
                        path to the tokenizer or the tokenizer name on huggingface
  --device DEVICE       device to run the model on
  --prompt PROMPT       prompt to generate from
  --out_prompts OUT_PROMPTS
                        number of prompts to generate
  --result_length RESULT_LENGTH
                        approximate length of the generated text
  --top_p TOP_P         top p parameter for nucleus sampling
  --temperature TEMPERATURE
                        temperature parameter for softmax
  --seed SEED           seed for random number generators
```

Example
```
python3 scripts/generate.py --model garipovroma/gpt_2_shakespeare_finetuned --tokenizer garipovroma/gpt_2_shakespeare_finetuned --prompt "To be or not to be" --out_prompts 5 --result_length 200

python3 scripts/generate.py --model garipovroma/gpt_2_shakespeare_finetuned --tokenizer garipovroma/gpt_2_shakespeare_finetuned --prompt "To be or not to be" --out_prompts 5 --result_length 200
Generating tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 16.69it/s]
Generating tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 16.99it/s]
Generating tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 16.31it/s]
Generating tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 16.32it/s]
Generating tokens: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.90it/s]
Generating text: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:15<00:00,  3.04s/it]
0:
To be or not to be as the lord is gone. I'll go. and the whole and I have gone. and I'll have gone. and no, if you'll have gone? And, sir is dead. And we'll have gone: I'd the good good

Number of words: 42
Length of the text: 185



1:
To be or not to be of him, the whole and I'll and be to be done. I am no of thee to be done. he hath gone. he must die to be gone. I'll not? he will not? And, if I'll not for a

Number of words: 45
Length of the text: 176



2:
To be or not to be in a good king to die of him, if it was gone from him, if they must have gone. For, if the whole earth is dead. I was in a man of a thousand dead and be gone to be gone. I was gone

Number of words: 50
Length of the text: 199



3:
To be or not to be in haste: I'll go, to be to be dead. I'll have gone. I say: I'll leave him, I'll go, I'll go, I'll go, I'll go, if, the good and he will not

Number of words: 38
Length of the text: 159



4:
To be or not to be as the lord, the good, I have, I shall not, I'll have to be to be done the world. I shall not to be. I have, if it was done. he was done. I have, if they will have

Number of words: 45
Length of the text: 182
```
