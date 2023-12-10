from search_engine import ranker, pipeline, bm25_qe, final_pipeline
import csv
import streamlit as st
import pandas as pd
from movieSearchLib.constants import *


# Streamlit app
def search_app():
    with st.sidebar:
        st.title("Movie Search Engine")
        st.write("""
            This is a search engine for movies. 
            You can search for movies using the search bar below.
            The search engine uses BM25 and LTR to rank the search results.
            The search engine also uses query expansion to improve search results.
            The search engine will support some movie scenario-based queries.
        """)
        st.write("Here are some example queries to try out:")
        st.write("1. Vampire, werewolf, and love.")
        st.write("2. Movies with epic space battles.")
        st.write("3. War films set during World War II.")
        st.write("4. Adaptations of classic literature into film.")
        st.write("5. Movies about superheroes.")
        selected_pipeline = st.selectbox("Select a pipeline", ["BM25", "L2R", "L2R + DLM"], index=0)
        if selected_pipeline == "BM25":
            selected_pipeline = bm25_qe
        elif selected_pipeline == "LTR":
            selected_pipeline = pipeline
        else:
            selected_pipeline = final_pipeline

    st.title("Movie Search Engine Interface")

    # Search box
    query = st.text_input("Enter your search query:")


    if query:
        # Assuming you have a search function that returns a DataFrame
        # results_df = search_function(query, df) 
        # For now, just displaying the whole DataFrame

        if len(bm25_qe.search(query)) != 0:
            results_df = selected_pipeline.search(query)
            ids = results_df["docno"].tolist()
            ids = list(map(int, ids))
            movies = []
            for id in ids:
                movie = {}
                movie["title"] = title[id]
                movie["content"] = synopsis[id]
                movie["tmdbid"] = tmdbid[id]
                movie["category"] = categories[id]
                movies.append(movie)

            results_df = pd.DataFrame(movies)

            for index, row in results_df.iterrows():
                with st.expander(f"{index+1}\. {row['title']}"):
                    category = row['category']
                    category = category.replace("|", " | ")
                    st.write(f"Category: {category}")
                    st.write(row['content'])
                    hyperlink = f"https://www.themoviedb.org/movie/{row['tmdbid']}"
                    st.markdown(f"[{hyperlink}]({hyperlink})", unsafe_allow_html=True)
        else:
            st.write("No results found")

if __name__ == "__main__":
    synopsis = {}
    tmdbid = {}
    with open(OVERVIEW_PATH, 'rt') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for line in csv_reader:
            synopsis[int(line[0])] = line[2]
            tmdbid[int(line[0])] = line[1]

    title = {}
    categories = {}
    with open(MOVIE_PATH, 'rt') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for line in csv_reader:
            title[int(line[0])] = line[1]
            categories[int(line[0])] = line[2]
    search_app()
