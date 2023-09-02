import streamlit as st
import pandas as pd
import pickle
import webbrowser
#-----------------------------------------------
# Read data:
movies = pickle.load(open('data/movies.pkl', 'rb'))
df = pd.DataFrame(movies)

similarity = pickle.load(open('data/similarity.pkl', 'rb'))
#----------------------------------------------------------------
st.title('!فیلم های مورد علاقتو پیدا کن')
st.text('.بعد از انتخاب ژانر و سپس فیلم مورد علاقه، دکمه "نمایش فیلم های مشابه" را کلیک کنید')
#********************************************************************************************
# Select genre and movie:
genre_list = df['genre'].unique().tolist()
selected_genre = st.selectbox(':ژانر مورد علاقه', genre_list)

movie_list = df[df['genre'] == selected_genre]['title']
selected_movie = st.selectbox(':فیلم مورد علاقه', movie_list)
#-------------------------------------------------------
# Get image of movies with url in dataframe:
def get_image(selected_movie):
    image_url = df[df.title == selected_movie]['post_image_link'].values[0]
    return image_url
#-------------------------------------------------------------
# Get link of movies:
def get_download_link(selected_movie):
    download_url = df[df.title == selected_movie]['post_link'].values[0]
    return download_url
#------------------------------------------------------------
# Get recommender movies:
def recommend(selected_movie):
    movie_id = df[df.title == selected_movie]['movie_id'].values[0]
    scores = similarity[movie_id]
    movies = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])[1:5]
    recommended_movies = []
    recommended_movie_images = []
    recommended_link_movie = []
    for i in movies:
        movie_name = df.iloc[i[0]].title
        recommended_movies.append(movie_name)
        recommended_movie_images.append(get_image(movie_name))
        recommended_link_movie.append(get_download_link(movie_name))
    return recommended_movies,recommended_movie_images,recommended_link_movie
#---------------------------------------------------------------
# Create a button for showing movies:
if st.button('نمایش فیلم های مشابه'):
    names, images, links = recommend(selected_movie)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.text(names[0])
        st.image(images[0])
        st.write("[لینک دانلود](%s)" % links[0])


    with col2:
        st.text(names[1])
        st.image(images[1])
        st.write("[لینک دانلود](%s)" % links[1])

    with col3:
        st.text(names[2])
        st.image(images[2])
        st.write("[لینک دانلود](%s)" % links[2])

    with col4:
        st.text(names[3])
        st.image(images[3])
        st.write("[لینک دانلود](%s)" % links[3])
#---------------------------------------------------