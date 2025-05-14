import pandas as pd

df = pd.read_csv("SmallReviews.csv")  

df = df.head(50000)

df.to_csv("reduced_dataset.csv", index=False)


print(df.shape)
print(df.head())
