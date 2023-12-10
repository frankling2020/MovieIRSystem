import csv
import requests

from tqdm import tqdm


api_key = '44060b42222980b4285273e0e724128b'

def getOverview(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)

    overview = ""
    if response.status_code == 200:
        movie_data = response.json()
        overview = movie_data['overview']
    
    return overview


def main():
    movies = []
    with open("data/links.csv", 'rt') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for line in tqdm(csv_reader, total=86537):
            movie = {}
            movie['id'] = line[0]
            movie['tmdbId'] = line[2]
            movie['overview'] = getOverview(line[2])
            movies.append(movie)
    
    csv_file_path = 'data/overview.csv'
    csv_header = ['id', 'tmdbId', 'overview']

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(movies)


if __name__ == "__main__":
    main()