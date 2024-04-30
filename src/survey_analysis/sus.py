import pandas as pd
import sys

infile = sys.argv[1]
outfile = sys.argv[2]
# infile = 'sus-input-data.csv'
# outfile = 'sus-results-tool1.csv'


# Load data from csv file
data = pd.read_csv(infile)

# Test the loaded data in Python
# data

# Create a list to store the data
sus = []

def str_to_int(str_list):
    return list(map(int, str_list))

# For each row in the column,
for index, row in data.iterrows() :
    odd_int = str_to_int([row['SUS1'], row['SUS3'], row['SUS5'], row['SUS7'], row['SUS9']])
    odd = sum(odd_int) - 5
    even_int = str_to_int([row['SUS2'], row['SUS4'], row['SUS6'], row['SUS8'], row['SUS10']])
    even = 25 - sum(even_int)

    total = (odd + even) * 2.5
    sus.append(total)
    print(total)

# Create a column from the list
data['total'] = sus

# Write results to a csv file
data['total'].to_csv(outfile, header=["total"])
