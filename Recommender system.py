#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(3)


# In[4]:


credits.head(3)


# In[5]:


movies=movies.merge(credits,on="title")


# In[6]:


movies.shape


# In[7]:


#columns use
#genres
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.duplicated().sum()


# In[12]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L


# In[13]:


movies["genres"]=movies["genres"].apply(convert)


# In[14]:


movies["keywords"]=movies["keywords"].apply(convert)


# In[15]:


movies.head()


# In[16]:


import ast
def convert3(obj): 
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i["name"])
            counter+=1
        else:
            break
    return L


# In[17]:


movies["cast"]=movies["cast"].apply(convert3)


# In[18]:


movies.head()


# In[19]:


movies["crew"].values


# In[20]:


import ast
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i["job"]=="Director":
            L.append(i["name"])
    return L


# In[21]:


movies["crew"]=movies["crew"].apply(fetch_director)


# In[22]:


movies.head()


# In[23]:


movies["overview"]=movies["overview"].apply(lambda x:x.split())


# In[24]:


movies.head(2)


# In[25]:


# remove space b/w words
movies["genres"]=movies["genres"].apply(lambda x:[i.replace(" ","") for i in x])
movies["keywords"]=movies["keywords"].apply(lambda x:[i.replace(" ","") for i in x])
movies["cast"]=movies["cast"].apply(lambda x:[i.replace(" ","") for i in x])
movies["crew"]=movies["crew"].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


movies.sample(5)


# In[27]:


# make new columns as tags
movies["tags"]=movies["overview"]+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]


# In[28]:


movies.head(2)


# In[29]:


new_df=movies[["movie_id","title","tags"]]


# In[30]:


new_df.head(2)


# In[31]:


new_df["tags"]=new_df["tags"].apply(lambda x:" ".join(x))


# In[32]:


new_df.head()


# In[33]:


new_df["tags"]=new_df["tags"].apply(lambda x:x.lower())


# In[34]:


new_df.sample(10)


# # Apply tags cleaning

# In[40]:


import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new_df["tags"]=new_df["tags"].apply(stem)


# In[41]:


new_df.head()


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=5000,stop_words="english")


# In[43]:


vector=cv.fit_transform(new_df["tags"]).toarray()


# In[44]:


vector[0]


# In[45]:


# use cosine distance
from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


similarity=cosine_similarity(vector)


# In[47]:


vector.shape


# In[48]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x: x[1])[1:6]


# In[49]:


def recommend(movie):
    movie_index=new_df[new_df["title"]==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[57]:


# if you want to select movies here
print(new_df["title"].sample(10))


# In[56]:


# that only work those movies in that dataset
recommend("Men in Black II")


# In[ ]:




