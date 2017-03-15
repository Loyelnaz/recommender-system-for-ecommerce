import pandas as pd
import json
import random
import operator
import numpy as np
def cust():
    print "Customer Dictionary .... Creating"
    print "r"
    df = pd.read_excel("Online Retail.xlsx")
    print "s"
    out1 = open("cust-to-id.json","w")
    out2 = open("id-to-cust.json","w")
    unique = 0
    custDict = {}
    revCDict = {}
    for row in df.iterrows():
        if str(row[1].CustomerID) not in custDict:
            custDict[str(row[1].CustomerID)] = unique
            revCDict[unique] = str(row[1].CustomerID)
            unique +=1
    json.dump(custDict, out1)
    json.dump(revCDict, out2)
    print "Customer Dictionary .... Done"
    return custDict

def item():
    df = pd.read_excel("Online Retail.xlsx")
    print "Item Dictionary .... Creating"
    out1 = open("item-to-id.json","w")
    out2 = open("id-to-item.json","w")
    unique = 0
    itemDict = {}
    revIDict = {}
    for row in df.iterrows():
        if str(row[1].StockCode) not in itemDict:
            itemDict[str(row[1].StockCode)] = unique
            revIDict[unique] = str(row[1].StockCode)
            unique +=1
    json.dump(itemDict, out1)
    json.dump(revIDict, out2)
    print "Item Dictionary .... Done"
    return itemDict

def ratingFor(num):
    allRatings = [0.0, 0.2, 0.5, 0.7, 0.8, 0.9, -0.1, -0.2] #Implementing negative scores
    lenRatings = len(allRatings)
    random.seed(num)
    return allRatings[random.randint(0,lenRatings-1)]

def topPredictions(data):
    User_Pred = {}
    count = 0
    for each in data:
        User_Pred[count] = []
        for i in range(5):
            index, value = max(enumerate(each), key=operator.itemgetter(1))
            User_Pred[count].append(index+i)
            each = np.delete(each,[index])
        #print User_Pred[count]
        count += 1
    outJson = open("Predictions-user-user.json", "w")
    json.dump(User_Pred, outJson)
    outJson.close()
#print ratingFor(0)

# cust()
# item()
