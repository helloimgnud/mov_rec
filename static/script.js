
const API_KEY = 'YOUR_API_KEY_HERE';  // Replace with your TMDB token
const TMDB_BASE_URL = 'https://api.themoviedb.org/3/movie/';

const form = document.getElementById('userForm');
const moviesDiv = document.getElementById('movies');
const loadingSpinner = document.getElementById('loadingSpinner');
const modal = document.getElementById('modal');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');

// Close modal when "X" is clicked
modalClose.onclick = () => {
    modal.style.display = 'none';
};

// Close modal when clicking outside content
modal.onclick = (e) => {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
};

form.onsubmit = async (e) => {
    e.preventDefault();
    moviesDiv.innerHTML = ''; // Clear old movie cards
    modal.style.display = 'none'; // Hide modal in case it was open
    loadingSpinner.style.display = 'block'; // Show spinner

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const res = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const movieIds = await res.json();
        if (!Array.isArray(movieIds)) {
            throw new Error("Expected an array of movie IDs from the backend");
        }

        for (const id of movieIds) {
            try {
                const movieRes = await fetch(`${TMDB_BASE_URL}${id}?language=en-US`, {
                    headers: {
                        accept: 'application/json',
                        Authorization: `Bearer ${API_KEY}`
                    }
                });

                if (!movieRes.ok) {
                    console.error(`Failed to fetch movie ID ${id}: ${movieRes.status}`);
                    continue;
                }

                const movie = await movieRes.json();

                // Create movie card
                const card = document.createElement('div');
                card.className = 'movie';
                card.innerHTML = `
                    <img src="https://image.tmdb.org/t/p/w300/${movie.poster_path}" alt="${movie.title}">
                    <h3>${movie.title}</h3>
                    <p><strong>Release:</strong> ${movie.release_date}</p>
                    <p><strong>Rating:</strong> ${movie.vote_average}</p>
                `;

                // Show modal on card click
                card.addEventListener('click', () => {
                    modalBody.innerHTML = `
                      <h2>${movie.title}</h2>
                      <img src="https://image.tmdb.org/t/p/w500/${movie.poster_path}" alt="${movie.title} Poster">
                      <p><strong>Original Title:</strong> ${movie.original_title}</p>
                      <p><strong>Tagline:</strong> ${movie.tagline || '—'}</p>
                      <p><strong>Overview:</strong> ${movie.overview}</p>
                      <p><strong>Genres:</strong> ${movie.genres.map(g => g.name).join(', ')}</p>
                      <p><strong>Language:</strong> ${movie.spoken_languages.map(l => l.english_name).join(', ')}</p>
                      <p><strong>Countries:</strong> ${movie.production_countries.map(c => c.name).join(', ')}</p>
                      <p><strong>Release Date:</strong> ${movie.release_date}</p>
                      <p><strong>Runtime:</strong> ${movie.runtime} minutes</p>
                      <p><strong>Rating:</strong> ${movie.vote_average} (${movie.vote_count} votes)</p>
                      <p><strong>Production Companies:</strong> ${movie.production_companies.map(p => `${p.name} (${p.origin_country})`).join(', ')}</p>
                      <p><strong>Budget:</strong> $${movie.budget.toLocaleString()}</p>
                      <p><strong>Revenue:</strong> $${movie.revenue.toLocaleString()}</p>
                      <p><strong>IMDB:</strong> <a href="https://www.imdb.com/title/${movie.imdb_id}" target="_blank">${movie.imdb_id}</a></p>
                    `;
                    modal.style.display = 'flex';
                });

                moviesDiv.appendChild(card);
            } catch (error) {
                console.error(`Error fetching movie ${id}:`, error);
            }
        }
    } catch (error) {
        console.error("Error getting recommendations:", error);
    } finally {
        loadingSpinner.style.display = 'none'; // Hide spinner after all done
    }
};
