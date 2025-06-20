from fetch import load_data, view
import restructure
import tokenization
from datasets import Dataset, NamedSplit, DatasetDict

if __name__ == "__main__":
    # Downloading the data using polars, restructuring it and converting to Pandas dataframe.
    train_df, val, test = load_data()
    trainData = restructure.toPandas(train_df)
    valData = restructure.toPandas(val)
    testData = restructure.toPandas(test)

    # Saving the dataframes as csv files.
    restructure.save(trainData, dir="./train.csv")
    restructure.save(valData, dir="./val.csv")
    restructure.save(testData, dir="./test.csv")

    # test_df = pd.read_csv("./train.csv")
    # print(test_df.iloc[0]['reasoning'])
    
    train_dataset = Dataset.from_pandas(trainData, split=NamedSplit('train'))
    val_dataset = Dataset.from_pandas(valData, split=NamedSplit('validation'))
    test_dataset = Dataset.from_pandas(testData, split=NamedSplit('test'))

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    tokenized_dataset = dataset_dict.map(tokenization.tokenize, batched=True)
    print(dataset_dict)
