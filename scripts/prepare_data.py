import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_folder", type=str, help="path to the data folder", default="data")
    argparser.add_argument("--val_size", type=float, default=0.15, help="size of the val dataset")
    argparser.add_argument("--seed", type=int, default=42, help="seed for random number generators")
    argparser.add_argument("--head", type=bool, default=False, help="head flag just to save only 100 head elements and use them in train")

    args = argparser.parse_args()

    df_path = os.path.join(args.data_folder, 'Shakespeare_data.csv')
    df = pd.read_csv(df_path)
    df = df.dropna() # drop lines which doesn't correspond to player, lines like "ACT I"
    df = df.reset_index()

    def concat(group):
        return ' '.join(group['PlayerLine'].tolist())

    df = df.groupby(['PlayerLinenumber', 'Player']).apply(concat).reset_index()
    df['Text'] = df[0]
    df = df.drop(columns=[0])

    df['TextLen'] = df['Text'].apply(lambda x: len(x))

    print(f'Total number of samples: {len(df)}')

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    
    print(f'Train size: {len(train_df)}')
    print(f'Val size: {len(val_df)}')

    if args.head:
        train_df = train_df.head(100)
        val_df = val_df.head(100)
        
    train_df.to_csv(os.path.join(args.data_folder, 'train_df.csv'), index=None)
    val_df.to_csv(os.path.join(args.data_folder, 'val_df.csv'), index=None)