import pandas

randomvalues = pandas.read_csv('./files/randomvalues.csv')
sample = pandas.read_csv('./files/sampleSubmission.csv')
sample.Sentiment = randomvalues
sample.to_csv('./outputs/random_output.csv', index=False)