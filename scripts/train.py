import argparse
import torch
from src.dataset import ShakespeareDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm
import os

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_folder", type=str, help="path to the data folder", default="data")
    argparser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    argparser.add_argument("--use_scheduler", type=bool, default=True, help="use scheduler flag")
    argparser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    argparser.add_argument("--batch_size", type=int, default=8, help="batch size")
    argparser.add_argument("--max_seq_len", type=int, default=300, help="max sequence length")
    argparser.add_argument("--device", type=str, default="cpu", help="device to run the model on")
    argparser.add_argument("--seed", type=int, default=42, help="seed for random number generators")
    argparser.add_argument("--num_workers", type=int, default=2, help="number of workers for data loaders")
    argparser.add_argument("--out_folder", type=str, default="shakespeare_finetuned", help="folder to save the model and tokenizer")
    argparser.add_argument("--train_verbose", type=int, default=500, help="number of batches after which to print the training loss")
    argparser.add_argument("--push_flag", type=bool, default=False, help="push to hub flag")
    argparser.add_argument("--huggingface_token", type=str, help="huggingface token")
    argparser.add_argument("--use_wandb", type=bool, default=False, help="use wandb flag")

    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.init(project="shakespeare-transformers", name="gpt2")
        wandb.config.learning_rate = args.learning_rate
        wandb.config.num_epochs = args.num_epochs
        wandb.config.batch_size = args.batch_size
        wandb.config.max_seq_len = args.max_seq_len

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = ShakespeareDataset(os.path.join(args.data_folder, 'train_df.csv'), tokenizer=tokenizer, args=args)
    val_dataset = ShakespeareDataset(os.path.join(args.data_folder, 'val_df.csv'), tokenizer=tokenizer, args=args)

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers)

    # Define optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    if args.use_scheduler:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=200,
            num_training_steps=-1,
        )

    device = torch.device(args.device)

    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_sum = 0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}'):
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            if args.use_scheduler:
                lr_scheduler.step()
            
            train_loss_sum += loss.item()
            
            if idx % args.train_verbose == args.train_verbose - 1:
                print(f'Epoch [{epoch + 1}/{num_epochs}] - Batch [{idx + 1}/{len(train_loader)}] - Training Loss: {loss.item():.4f}')
        
        train_loss = train_loss_sum / len(train_loader)
        train_perplexity = np.exp(train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {train_loss:.4f} - Perplexity: {train_perplexity:.4f}")
        
        # Validation
        model.eval()
        val_loss_sum = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc=f'Evaluation after epoch {epoch + 1}'):
                batch = batch.to(device)
                outputs = model(batch, labels=batch)

                loss = outputs.loss
                val_loss_sum += loss.item()
        
        val_loss = val_loss_sum / len(val_loader)
        val_perplexity = np.exp(val_loss)

        full_metrics_dict = {
            'train_loss': train_loss,
            'train_perplexity': train_perplexity,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
        }
        print(f"Epoch [{epoch + 1}/{num_epochs}] - {full_metrics_dict}")
        if args.use_wandb:
            wandb.log(full_metrics_dict)

    # Optionally, save the trained model
    model.save_pretrained(args.out_folder)

    # Finish the wandb run
    if args.use_wandb:
        wandb.finish()

    if args.push_flag:
        model.push_to_hub("gpt_2_shakespeare_finetuned", use_auth_token=args.huggingface_token)
        tokenizer.push_to_hub("gpt_2_shakespeare_finetuned", use_auth_token=args.huggingface_token)