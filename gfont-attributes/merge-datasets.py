import pandas as pd

attributes = pd.read_csv('fonts-attributes.csv')
zhats = pd.read_csv('zhats.csv')

# BIG PROBLEM: DUPLICATE FONT STYLES
merged = pd.merge(zhats, attributes, on='font', how='outer')
merged = merged.drop_duplicates() # this is super iffy, currently arbitrary style choice
merged = merged.dropna() # we lose 514 fonts with this, these are fonts which were not identified in our torch dataset

merged.to_csv("merged.csv", index=False)