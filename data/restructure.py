import random

def restructureData(example):
  '''
    The purpose of "restructuring" is to have a clear format for the input and output data for the model.
    Adding 'think' and 'answer' tags.
  '''
  options = [f"{example[i]}" for i in range(1, 5)]
  random.shuffle(options)
  re_input = f"{example[0]}\nOptions: {options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"
  re_reason = f"<think>{example[5]}</think>"
  re_output = f"{example[4]}"
  return re_input, re_reason, re_output


def toPandas(df):
    '''
        Renaming the columns and then converting to Pandas dataframe.
    '''
    newData = df.map_rows(lambda x: restructureData(x))
    print("Restructured ")
    newData.columns = ['questions', 'reasoning', 'answers']
    # Interestingly enough Dataset.from_polars() is not supported now.
    # So, I convert the newly formed polars dataset to pandas, and then in the cell below into an distinct Dataset object based on the splits.
    # Then, combining them into a master 'dataset_dict' object.
    Data_pd = newData.to_pandas()
    return Data_pd

def save(df, dir=""):
    df.to_csv(dir, index=False)
    print(f"Dataframe saved as csv at {dir}")