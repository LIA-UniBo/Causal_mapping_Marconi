import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


node = "merged"
distance = 4
extreme = True

if extreme:
    xls_file = "./results/Validation/{}_{}_0.8_extreme.xlsx".format(node,distance)
else:
    xls_file = "./results/Validation/{}_{}_0.8.xlsx".format(node, distance)

df = pd.read_excel(xls_file)

count_less = 0
count_greater = 0
for index, row in df.iterrows():
    if row[1] < row[3]:
        count_less += 1
    else:
        count_greater += 1

list_comparison = [count_less,count_greater]
list_label = ["The loss of the causes is lower","The loss of the ind. features is lower"]

plt.bar(list_label,list_comparison)
plt.title("Comparison of the two losses for merged dataset and distance = {}".format(distance))
plt.ylabel("Amount")
plt.savefig("generic.png")
plt.show()
list_diff = [0,0,0,0]
for index, row in df.iterrows():
    if row[1] < row[3]:
        diff = row[3] - row[1]
        if 0.001 < diff <= 0.01:
            list_diff[0]+=1
        elif 0.01 < diff <= 0.1:
            list_diff[1] += 1
        elif 0.1 < diff <= 1:
            list_diff[2] += 1
        else:
            list_diff[3] += 1

list_label = ["0.001 - 0.01","0.01 - 0.1","0.1 - 1","1. - 10."]
plt.bar(list_label,list_diff)
plt.title("Differences between the losses")
plt.ylabel("Amount")
plt.xlabel("Intervals")
plt.savefig("focus.png")
plt.show()




