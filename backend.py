import csv

from flask import Flask, request, jsonify
from flask_cors import CORS
from movieSearchLib.constants import *
from search_engine import final_pipeline, ranker


app = Flask(__name__)
CORS(app)

synopsis = {}
tmdbid = {}
with open(OVERVIEW_PATH, 'rt') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)

    for line in csv_reader:
        synopsis[int(line[0])] = line[2]
        tmdbid[int(line[0])] = line[1]

title = {}
with open(MOVIE_PATH, 'rt') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)

    for line in csv_reader:
        title[int(line[0])] = line[1]


@app.route('/search', methods=['GET'])
def search_movies():
    query = request.args.get('query', '')
    page = int(request.args.get('page', 1))
    per_page = 10
    if len(ranker.bm25_qe.search(query)) == 0:
        return jsonify([])
    results = final_pipeline.search(query)
    ids = results["docno"].tolist()
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    ids = ids[start_idx:end_idx]
    movies = []
    for id in ids:
        movie = {}
        movie["title"] = title[int(id)]
        movie["synopsis"] = synopsis[int(id)]
        movie["tmdbid"] = tmdbid[int(id)]
        movies.append(movie)

    return jsonify(movies)

if __name__ == '__main__':
    app.run(debug=True)
