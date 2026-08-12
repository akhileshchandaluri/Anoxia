import pandas as pd

df = pd.read_parquet('data/argo/feature_matrix.parquet')
print('Feature Matrix Summary')
print('='*60)
print(f'Shape: {df.shape}')
print(f'Rows: {len(df)}')
print()
print('Columns (9 total):')
for i, col in enumerate(df.columns, 1):
    print(f'  {i}. {col:15s} ({df[col].dtype})')
print()
print('Statistics:')
print(f'  Hypoxic (DO<2.0): {(df["label"]==1).sum()} ({100*(df["label"]==1).sum()/len(df):.1f}%)')
print(f'  Oxic (DO>=2.0):   {(df["label"]==0).sum()} ({100*(df["label"]==0).sum()/len(df):.1f}%)')
print()
print('Data Quality:')
print(f'  Missing values: {df.isnull().sum().sum()} (0%)')
print(f'  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB')
