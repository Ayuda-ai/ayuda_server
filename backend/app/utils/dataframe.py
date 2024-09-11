import pandas as pd

'''
    Utility functions to play with dataframes
'''

def make_dataframe(data):
    df = pd.DataFrame(data)
    return df

def make_threefield_dataframe(name1, data1, name2, data2, name3, data3):
    df = pd.DataFrame({name1: data1, name2: data2, name3: data3})
    return df

def sort_dataframe(dataframe, sortBy, ascending):
    df = dataframe.sort_values(by=sortBy, ascending=ascending)
    return df