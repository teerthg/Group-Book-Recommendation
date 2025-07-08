import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load saved files:
model = pickle.load(open('model.pkl', 'rb'))
book_names = pickle.load(open('book_names', 'rb'))
Book_pivot = pickle.load(open('user_item_matrix', 'rb'))
Book_pivot_sparse = pickle.load(open('user_item_matrix_sparse', 'rb'))
Books = pickle.load(open('books_df.pkl', 'rb'))


def recommend_book(book_title, model, Book_pivot, Books):
    try:
        book_id = np.where(Book_pivot.index == book_title)[0][0]
        distances, suggestions = model.kneighbors(Book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
        recommendation = []
        for i in range(len(suggestions)):
            books_indices = suggestions[i]
            for j in books_indices:
                recommendation.append(Books.iloc[j]['Book-Title'])
        return recommendation
    except Exception as e:
        return [f"An error occurred: {e}"]


st.title("Book Recommendation System")
book_input = st.selectbox("Choose a book title:", list(book_names))

if st.button('Recommend'):
    recommendation = recommend_book(book_input, model, Book_pivot, Books)
    st.subheader("Recommended Books you may also like")
    for book in recommendation:
        st.write(book)
