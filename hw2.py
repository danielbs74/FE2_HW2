import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import string
import re
import csv
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


nlp = spacy.load("en_core_web_sm")
nlp2 = spacy.load("es_core_news_sm")
nlp3 = spacy.load("de_core_news_sm")

df = pd.read_csv('C:/Users/batista1/OneDrive/PhD/Financial Econometrics 2/B_Homeworks/Hwk2/all_ECB_speeches_Dec21.csv', sep = '|')

df = df.dropna()

df.tail()
df.columns
df['speakers'].unique()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by = 'date')
df = df.reset_index(drop=True)

df['contents'] = df['contents'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))

df['Count_Column'] = df['date'].map(df.date.value_counts().to_dict()) 

df['Last_Name'] = df.speakers

for i in range(0, len(df)):
    df['Last_Name'][i] = df['speakers'][i].split()[-1]
    
df.Last_Name = pd.Categorical(df.Last_Name)

plt.figure(figsize=(8, 4.8), dpi=300)
plt.scatter(df.date, df.Last_Name, s = 0.5)
plt.yticks(fontsize = 6)
plt.show()

words = []

for i in range(0, len(df)):
    words.extend([word.lower() for word in df.contents[i].split()])



all_stopwords = nlp.Defaults.stop_words.union(nlp2.Defaults.stop_words.union(nlp3.Defaults.stop_words))

words_without_sw = [word for word in words if not word in all_stopwords]



WordsCounter = Counter(words)
WordsCounter.most_common(25)

WordsCounter_without_sw = Counter(words_without_sw)
WordsCounter_without_sw.most_common(25)


def freq_words(x):
        
    words_aux = []
    df_aux = df[df.Last_Name == x]
    df_aux = df_aux.reset_index(drop=True)
    
    for i in range(0, len(df_aux)):
        words_aux.extend([word.lower() for word in df_aux.contents[i].split()])
    
    words_aux = [word for word in words_aux if not word in all_stopwords]

        
    WordsCounter_aux = Counter(words_aux)
    d = {x : WordsCounter_aux.most_common(25)}
    return(d)


common = []

for speaker in df.Last_Name.unique():
    common.append(freq_words(speaker))
    
    
a_file = open("common.csv", "w")

writer = csv.writer(a_file)
for j in common:
    for key, value in j.items():
        writer.writerow([key, value])

a_file.close()


df6D = df[df.Last_Name == 'Draghi']
df6L = df[df.Last_Name == 'Lagarde']
df6D = df6D.reset_index(drop=True)
df6L = df6L.reset_index(drop=True)

words6D = ''

for i in range(0, len(df6D)):
    words6D = words6D + ' '.join([word.lower() for word in df6D.contents[i].split() if not word in all_stopwords])


words6D = words6D.replace('“', "")
words6D = words6D.replace('”', "")

wordcloud = WordCloud(width=800, height=800).generate(str(words6D))
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



words6L = ''

for i in range(0, len(df6L)):
    words6L = words6L + ' '.join([word.lower() for word in df6L.contents[i].split() if not word in all_stopwords])


words6L = words6L.replace('“', "")
words6L = words6L.replace('”', "")

wordcloud = WordCloud(width=800, height=800).generate(str(words6L))
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()




def freq_words_year(x):
    words_aux = []
    df_aux = df[df.date.dt.year == x]
    df_aux = df_aux.reset_index(drop=True)
    
    for i in range(0, len(df_aux)):
        words_aux.extend([word.lower() for word in df_aux.contents[i].split()])
    
    words_aux = [word for word in words_aux if not word in all_stopwords]

        
    WordsCounter_aux = Counter(words_aux)
    d = WordsCounter_aux.most_common(100)
    a = [word[0] for word in d]
    return(a)



benchmark = freq_words_year(1997)
dyn = []

for i in range(1997, 2022):
    dyn.append(len(set(freq_words_year(i)) & set(benchmark)))

plt.scatter(range(1997, 2022), dyn); plt.title('Similarity of Top 100 - 1997');plt.show()



benchmark = freq_words_year(2007)
dyn = []

for i in range(1997, 2022):
    dyn.append(len(set(freq_words_year(i)) & set(benchmark)))

plt.scatter(range(1997, 2022), dyn); plt.title('Similarity of Top 100 - 2007');plt.show()



benchmark = freq_words_year(2017)
dyn = []

for i in range(1997, 2022):
    dyn.append(len(set(freq_words_year(i)) & set(benchmark)))

plt.scatter(range(1997, 2022), dyn); plt.title('Similarity of Top 100 - 2017');plt.show()



benchmark = freq_words_year(2021)
dyn = []

for i in range(1997, 2022):
    dyn.append(len(set(freq_words_year(i)) & set(benchmark)))

plt.scatter(range(1997, 2022), dyn); plt.title('Similarity of Top 100 - 2021');plt.show()


