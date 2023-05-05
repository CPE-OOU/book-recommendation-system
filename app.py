
import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('model.pkl','rb'))
final_rating = pickle.load(open('booksMerge.pkl','rb'))
genrelabels = pickle.load(open('genrelabels.pkl','rb'))

genre = st.selectbox('Select a genre:', sorted(genrelabels))
rating = st.slider('Select a rating:', min_value=0.0, max_value=5.0, step=0.1)

#genre_labels = final_rating['genres'].astype('category').cat.categories.tolist()
genre_mapping = {'genres': {k: v for k,v in zip(genrelabels,list(range(1,len(genrelabels)+1)))}}
final_rating.replace(genre_mapping, inplace=True)


genre_num = genre_mapping['genres'][genre]
book = [[genre_num, rating]]


if st.button('Recommend Books'):

    distances, indices = model.kneighbors(book)

    for i in indices[0]:
        st.write(final_rating.loc[i]['Title'])

    # st.text(recommended_books[5])
       # st.image(poster_url[5])