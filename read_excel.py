import pandas as pd

def read_file(filename):
    file = pd.read_excel(filename)
    file = file.values.tolist()
    return file

#data = read_file('training.xlsx')
#print(data[:5])
