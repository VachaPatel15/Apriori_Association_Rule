import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
# header equals to none states that the first row is not the names of the columns and hence do not skip it

transactions=[]
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# 20 is max size of bucket, or no. of columns
# 7501 rows

from apyori import apriori
rules = apriori(transactions= transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2 )
# min_support = (3*7)/7501 , 3 as 3 same products, 7 as calculating in a week
# min_confidence hit and trial
# 3-9 is good min_lift

# visualising results
results = list(rules)
print(results)

# putting results in well organised manner
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# displaying the results non sorted
print(resultsinDataFrame)

# displaying results by descending lift
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))
