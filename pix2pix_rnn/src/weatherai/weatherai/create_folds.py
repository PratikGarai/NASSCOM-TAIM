import pandas as pd
from sklearn import model_selection
from src.config import config

if __name__ == '__main__':
    kfold = model_selection.KFold(n_splits=config.num_folds)

    df = pd.read_csv(config.train_path)
    df = df.sample(frac=1).reset_index(drop=True)

    df['kfold'] = -1

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=df, y=df['results'])):
        df.loc[valid_idx, 'kfold'] = fold
      
    df.to_csv('../data/csvs/train_folds.csv', index=False)