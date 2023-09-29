import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ShakespeareDataset(Dataset):
    
    def __init__(self, df_path, tokenizer, args):
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenized_text = []
        self.df = pd.read_csv(df_path)
        for ind, data in tqdm(self.df.iterrows(), total=self.df.shape[0], desc=f'Tokenizing {df_path}'):
            encoded = tokenizer.encode(data['Text'], max_length=args.max_seq_len, padding='max_length', truncation=True)
            self.tokenized_text.append(torch.tensor(encoded))
            
    def __len__(self):
        return len(self.tokenized_text)
    
    def __getitem__(self, index):
        return self.tokenized_text[index]