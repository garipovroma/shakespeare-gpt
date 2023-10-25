# GPT-2 Fine-tuning on Shakespeare dataset

ShakespeareGPT is a [GPT2](https://huggingface.co/gpt2) model fine-tuned on the [Shakespeare dataset](https://www.kaggle.com/kingburrito666/shakespeare-plays). 

In this repository you can find the code for training the model and generating text from it.

The fine-tuned model is available on the [🤗 HuggingFace](https://huggingface.co/garipovroma/gpt_2_shakespeare_finetuned). You can infer it right in your browser!

## Metrics:
Model was trained for 1 epoch
```
Epoch [1/1] - {'train_loss': 1.1098246806401264, 'train_perplexity': 3.033826459294963, 'val_loss': 1.0202441342251818, 'val_perplexity': 2.7738718783693135}
```

## Examples of generated text:
Prompt: I am the king, that's why
```
0:
I am the king, that's why I have the lord of he must. I have gone. he hath no good and no'' he hath no of good. he must, I to be gone. I, I have to be to be of good and you are not to be. and the whole and no good and he hath gone to be in a whole. I have, I to and thou know. And, I and and I are dead. And, I is. And, the good and I is not to good
Number of words: 89
Length of the text: 349


1:
I am the king, that's why I am no man to to be to be done the whole king: he must go? and it must, he will not be no. I, he will, if the power to be in a thousand. I have, I have done? And be no king of king of no. and be done of a man of king of king. he will not? he must, I was in, I have to be. and I was not to be of good and I have no good king.
Number of words: 90
Length of the text: 351
```

Prompt: Once upon a time
```
0:
Once upon a time to the head, I am gone. I'll go, if I'll go, to be good and you are gone? I am good and to be no man to be of good. I'll have done of him, if the whole. I'll have no. he must have no good. and I shall, the whole. then. he must have no good king of him. and the head and be dead. and I, for. and I have no of a man of a head
Number of words: 83
Length of the text: 340


1:
Once upon a time to the sword and and to be dead: for you will not the heart, I am no man. I shall not be gone. I'll, sir is gone. he will not. he hath no. the head to be done. he now. and I'll say the sword of him in him in a head of him to him to him. I was not. he was dead. he was done to him in, he was no. I shall, he was in a king of him
Number of words: 86
Length of the text: 344
```

Prompt: Such a beautiful day! Such a
```
0:
Such a beautiful day! Such a thousand and I will not. and he was no king of king and be the royal majesty, I have to be the royal head. and be dead and I'll make to be no, for and be gone. and I was, if it was of and of a good and be dead and he hath gone. I'd be to a whole. I hath the head of no of all. I the like. And and he, I the and he hath gone. thou must the'd to, he is in, the king to, he must. thou? thou thou is in, thou is not? the whole.
Number of words: 108
Length of the text: 452


1:
Such a beautiful day! Such a world. he was no. the head in a man of a head of a head and of a whole and I will not to be gone. I am dead. the head and I will, I have, he will, for he must, he will not be no king. and be dead. he was done in no more of you. and be dead. and I. And, if the good king of no more and he will, I, he hath to and he was not to, he no and and he will. he will, the head to be in no of and he will the world. I have
Number of words: 111
Length of the text: 441
```

Prompt: Christ
```
0:
Christ, and the man of his own hand: I'll not, no. he must, the whole the whole and he was dead to be no man and I'll make, I'll not of no, the head. the whole and I have, if the world of good and be of good. he hath not, if they will the head. the world and I was. and I shall the world and be no good. the lord of and I have, the good and I shall not, I to be, I shall not, he must, the head the head and be of and he will not, if we
Number of words: 103
Length of the text: 435


1:
Christ, sir is to him. I must go, sir is not the lord and let the whole. I am gone. I am dead. I shall not. and the whole world is not to be done. I was, sir is, I'll have no. he will go. he must go. I shall go. and the world and be in a good and I. and be done the good. I have, I, I have no. And, sir is, he will go. and go. and I am not. he must the head to be in a dead. I have, sir. let it
Number of words: 98
Length of the text: 394
```
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

## Train model
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

## Generate text
The nucleus sampling is used for text generation.

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
