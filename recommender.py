#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# In[2]:


# Read data:
df_movies = pd.read_excel('data/persian-movie-info.xlsx')
df = df_movies.copy()
df


# # EDA

# In[3]:


df.head(30)


# In[113]:


df.shape


# In[114]:


df.tail(25)


# In[115]:


df.dtypes


# In[116]:


df.info()


# In[117]:


df.describe().T


# In[4]:


df[df.duplicated()]


# In[4]:


# Delete NAN rows:
df = df.dropna(axis=0, how='all')
df.tail(10)


# In[5]:


df.isnull().sum()


# In[6]:


df['title'].value_counts().to_frame()


# In[7]:


df[df['title']=='سریال گیسو Gisoo']


# In[8]:


df[df['title']=='سریال هم گناه Hamgonah']


# In[9]:


df = df.drop(axis=0, index=df.index[[93,203]])


# In[10]:


df[df['title']=='سریال هم گناه Hamgonah']


# ###  اضافه کردن ستونی تحت عنوان movie_id برای استفاده در سیستم توصیه گر:

# In[11]:


df = df.reset_index(drop=True)
df['movie_id'] = df.index
df


# In[12]:


df['year'].value_counts().to_frame()


# In[10]:


df['country'].value_counts().to_frame()


# In[11]:


df['time'].value_counts().to_frame()


# In[12]:


df['director'].value_counts().to_frame()


# ### در ستون director دو داده نال وجود دارد، همچنین در ستون cast داده نویز وجود دارد که با مقادیر مناسب جاگذاری میشود:

# In[13]:


df[df['director'].isnull()]


# In[13]:


df[df['genre'] == 'انیمیشن']


# In[15]:


print(df.at[96,'director'])
print(df.at[219,'director'])


# ### چون تنها دو داده نال وجود دارد، با مقادیر متناسب جایگذاری میشود:

# In[14]:


df.at[96,'director'] = "مجید محمودی"
df.at[219,'director'] = "علی درخشی"
df.at[96,'cast'] = ""
df.at[219,'cast'] = ""


# In[15]:


df[df.genre.str.contains("انیمیشن")]


# In[16]:


df['cast'].value_counts().to_frame()


# In[17]:


df['genre'].value_counts().to_frame()


# In[18]:


df['post_image_link'].value_counts().to_frame()


# ### پاک کردن کاراکترهای اضافی در ستون های title، director، cast، genre:
# ### منظور از کاراکترهای اضافه، حروف انگلیسی، خط تیره، فاصله های قبل و بعد نام فیلم، اعداد و ... میباشد.
# ### لزوم انجام این کار این است که بتوان این ستون ها را بدون کاراکترهای اضافی ترکیب کرد و با استفاده شباهت آن ها سیستم توصیه گر را ساخت. 

# In[19]:


df1 = df.copy()


# In[20]:


df1['title'] = df1['title'].str.replace(r'[A-Z,a-z,0-9]+', '',regex=True)
df1['title'] = df1['title'].str.replace(r'[-,:,&,.,_]+', '', regex=True)
df1['title'] = df1['title'].str.replace("'", '')


# In[21]:


df1[df1.title.str.contains("-")]


# In[22]:


df1['title'] = df1['title'].str.strip()
df1


# In[23]:


df1['director'] = df1['director'].str.replace(r'[0-9,-]+', '', regex=True)
df1['director'] = df1['director'].str.replace(' ', '')
df1[df1.director.str.contains(" ")]


# In[24]:


df1[df1.cast.str.contains("-")]


# In[25]:


df['cast'] = df['cast'].str.replace('\d+', '', regex=True)
df1['cast'] = df1['cast'].str.replace('-', '')
df1['cast'] = df1['cast'].str.replace(' ', '')
# df1[~df1.cast.str.contains("-")]
df1


# In[26]:


df['genre'] = df['genre'].str.replace('\d+', '', regex=True)
# df1['genre'] = df1['genre'].str.replace(' ', '')
df1


# ### فاصله بین اسامی برای عملکرد بهتر سیستم توصیه گر حذف شد.
# 
# ### انتخاب ستون های مورد نیاز برای سیستم توصیه گر:

# In[28]:


df_final = df1[['title','director','cast', 'genre', 'movie_id', 'post_link','post_image_link']]
df_final


# ### ترکیب ویژگی های title, director, cast, genre به ازای هر سطر و ساخت یک ستون جدید تحت عنوان selected_features:
# ### این کار برای انجام CountVectorizer انجام میشود (تعداد هر کلمه از ستون جدید به ازای هر سطر محاسبه میشود) و سپس با استفاده از اعداد بدست آمده از CountVectorizer و شباهت کسینوسی، شباهت هر فیلم با دیگر فیلم ها بدست میآید. (مسلم است که قطر اصلی ماتریس شباهت عدد یک قرار دارد زیرا هر فیلم شبیه خودش است.) 

# In[29]:


def combine_features(data):
    selected_features = []
    for i in range(0, data.shape[0]):
        selected_features.append(data['title'][i]+ ' '+data['director'][i]+' '+data['cast'][i]+' '+data['genre'][i])
    return selected_features
df_final['selected_features'] = combine_features(df_final)
df_final['selected_features'][0]


# In[30]:


cv = CountVectorizer().fit_transform(df_final['selected_features'])


# In[31]:


cs = cosine_similarity(cv)
cs


# In[32]:


# Check dimention of similarity matrix:
cs.shape


# ### با استفاده از شباهت های بدست آمده میتوان سیستم توصیه گر را پیاده سازی کرد؛ به این صورت که با در نظر گرفتن یک حد آستانه و یا یک تعداد خاصی از فیلم ها، شبیه ترین فیلم ها به فیلم انتخاب شده، توصیه میشود.
# 
# ### ذخیره دیتافریم برای استفاده در streamlit:

# In[33]:


import pickle
pickle.dump(df_final.to_dict(),open('streamlit/data/movies.pkl','wb'))


# In[34]:


pickle.dump(cs,open('streamlit/data/similarity.pkl','wb'))


# ### تست یک نمونه:
# ### با توجه به اینکه این سیستم از شباهت کسینوسی استفاده میکند، میتوان به راحتی با دیدن اعداد مربوط به شباهت سیستم را ارزیابی کرد.

# In[35]:


name = 'انیمیشن حسنا کوچولو'
df1[df1['title'] == name]


# In[36]:


movie_id = df1[df1.title == name]['movie_id'].values[0]
movie_id


# In[37]:


scores = list(enumerate(cs[movie_id]))
scores


# In[38]:


sorted_scores= sorted(scores,key=lambda x:x[1],reverse=True)
sorted_scores = sorted_scores[1:]
sorted_scores


# In[39]:


j=0
print('پنج مورد از شبیه ترین ها:')
for item in sorted_scores:
    movie_title = df1[df1.movie_id == item[0]]['title'].values[0]
    print(j+1,movie_title)
    j = j+1
    if j > 5:
        break

