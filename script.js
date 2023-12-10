const baseUrl = 'http://localhost:5000'; // Replace with your actual backend URL
let page = 1;
let loading = false;

function searchMovies() {
    const query = document.getElementById('searchInput').value;

    // Reset page and clear existing results
    page = 1;
    clearResults();

    if (query.trim() !== '') {
        // Fetch movies from the backend only if the query is not empty
        fetchMovies(query);
    }
}

function fetchMovies(query) {
    if (loading) return; // Do not fetch movies if already loading
    loading = true;

    // Fetch movies from the backend
    fetch(`${baseUrl}/search?query=${query}&page=${page}`)
        .then(response => response.json())
        .then(data => {
            displayMovies(data);

            // Allow loading for the next set of movies
            loading = false;
        })
        .catch(error => {
            console.error('Error fetching movies:', error);
            loading = false;
        });
}

function displayMovies(movies) {
    const movieListContainer = document.getElementById('movieList');

    movies.forEach(movie => {
        const movieDiv = document.createElement('div');
        movieDiv.classList.add('movie');

        const tmdbLink = document.createElement('a');
        tmdbLink.href = `https://www.themoviedb.org/movie/${movie.tmdbid}`;
        tmdbLink.target = '_blank'; // Open the link in a new tab
        tmdbLink.textContent = movie.title;

        const synopsis = document.createElement('p');
        synopsis.textContent = movie.synopsis;

        movieDiv.appendChild(tmdbLink);
        movieDiv.appendChild(synopsis);

        movieListContainer.appendChild(movieDiv);
    });

    // Increment the page number for the next set of movies
    page++;
}

function clearResults() {
    const movieListContainer = document.getElementById('movieList');
    movieListContainer.innerHTML = '';
}

// Event listener for scrolling
window.addEventListener('scroll', () => {
    const { scrollTop, scrollHeight, clientHeight } = document.documentElement;

    // Check if the user has scrolled to the bottom of the page
    if (scrollTop + clientHeight >= scrollHeight - 10) {
        const query = document.getElementById('searchInput').value;
        fetchMovies(query);
    }
});

// Event listener for search button click
document.getElementById('searchButton').addEventListener('click', searchMovies);

// Clear results when the page loads or is refreshed
window.addEventListener('load', clearResults);
