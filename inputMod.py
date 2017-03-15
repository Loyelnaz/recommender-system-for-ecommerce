import csv
from utils import ratingFor
import pandas as pd
import json
import xlrd
df = pd.read_excel("Online Retail.xlsx")
outputFile = open("OnlineMod.csv","w+")
with open("cust-to-id.json", "r") as custD:
    custDict = json.load(custD)
with open("item-to-id.json", "r") as itemD:
    itemDict = json.load(itemD)

lineCount = 0
print "Starting"
for each in df.iterrows():
    lineCount+=1
    item = str(each[1].StockCode)
    cust = str(each[1].CustomerID)
    outputFile.write(str(itemDict[item]) + "\t" + str(custDict[cust]) + "\t" + str(ratingFor(lineCount)) + "\n")

outputFile.close()
# inp.close()
