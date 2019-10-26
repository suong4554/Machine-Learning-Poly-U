import pandas as pd

def createSubmission(df, dir):
    dir = dir + "//submission//submission.csv"
    #df = df.reset_index()
    #df.columns[0] = "id"
    df.to_csv(index=True)
    print("submission written to file")
    #print(df)