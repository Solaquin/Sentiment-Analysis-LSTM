/*(function(){
            const fileInput = document.getElementById('file-input');
            const uploadBtn = document.getElementById('upload-btn');
            const textarea = document.getElementById('review-text');
            const reviewsContainer = document.getElementById('reviews-container');
            const analyzeBtn = document.getElementById('analyze-btn');
            const totalEl = document.getElementById('total-reviews');
            const posEl = document.getElementById('positive-reviews');
            const negEl = document.getElementById('negative-reviews');

            function splitReviewsFromText(text){
                // separa por líneas en blanco (una o más)
                return text.split(/\r?\n\s*\r?\n/).map(s => s.trim()).filter(Boolean);
            }

            function renderReviews(reviews){
                reviewsContainer.innerHTML = '';
                reviews.forEach((r, idx) => {
                    const sentiment = (idx % 2 === 0) ? 'positive' : 'negative'; // primera positiva, segunda negativa, etc.
                    const item = document.createElement('div');
                    item.className = `review-item ${sentiment}`;

                    const header = document.createElement('div');
                    header.className = 'review-header';
                    header.innerHTML = `<div class="review-index">#${idx+1}</div>`;

                    const sentimentBlock = document.createElement('div');
                    sentimentBlock.className = 'review-sentiment';
                    const icon = document.createElement('img');
                    icon.className = 'sentiment-icon';
                    icon.src = sentiment === 'positive' ? 'Icons/PositiveIcon.svg' : 'Icons/NegativeIcon.svg';
                    icon.alt = sentiment === 'positive' ? 'Positive Icon' : 'Negative Icon';
                    const label = document.createElement('div');
                    label.className = 'sentiment-label';
                    label.innerHTML = `<strong>${sentiment === 'positive' ? 'Positivo' : 'Negativo'}</strong><div class="sentiment-sub">Esta reseña expresa una experiencia ${sentiment === 'positive' ? 'positiva' : 'negativa'}</div>`;

                    sentimentBlock.appendChild(icon);
                    sentimentBlock.appendChild(label);

                    const textBlock = document.createElement('div');
                    textBlock.className = 'review-text';
                    textBlock.textContent = r;

                    item.appendChild(header);
                    item.appendChild(sentimentBlock);
                    item.appendChild(textBlock);

                    reviewsContainer.appendChild(item);
                });

                // actualizar estadísticas: primera positiva
                const total = reviews.length;
                const positives = Math.ceil(total / 2);
                const negatives = Math.floor(total / 2);
                totalEl.textContent = total;
                posEl.textContent = positives;
                negEl.textContent = negatives;
            }

            uploadBtn.addEventListener('click', () => fileInput.click());

            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files && e.target.files[0];
                if(!file) return;
                const text = await file.text();
                let reviews = [];
                if (file.name.toLowerCase().endsWith('.json')){
                    try{
                        const parsed = JSON.parse(text);
                        if (Array.isArray(parsed)){
                            reviews = parsed.map(p => typeof p === 'string' ? p : JSON.stringify(p));
                        } else if (parsed && typeof parsed === 'object'){
                            // intentar detectar un campo "reviews" o similar
                            if (Array.isArray(parsed.reviews)) reviews = parsed.reviews.map(String);
                            else reviews = [JSON.stringify(parsed)];
                        } else {
                            reviews = [String(parsed)];
                        }
                    } catch(err){
                        // si no es JSON válido, tratar como texto plano
                        reviews = splitReviewsFromText(text);
                    }
                } else {
                    reviews = splitReviewsFromText(text);
                }

                // copiar al textarea separando con una linea en blanco
                textarea.value = reviews.join('\n\n');

                renderReviews(reviews);

                // limpiar el input para permitir subir el mismo archivo otra vez si es necesario
                fileInput.value = '';
            });

            analyzeBtn.addEventListener('click', () => {
                const raw = textarea.value || '';
                const reviews = splitReviewsFromText(raw);
                renderReviews(reviews);
            });

            // Si hay texto inicial en el textarea (ej. por server-side), renderizarlo
            document.addEventListener('DOMContentLoaded', () => {
                if (textarea.value.trim()){
                    const initial = splitReviewsFromText(textarea.value);
                    renderReviews(initial);
                }
            });
        })(); */

const textarea = document.getElementById('review-text');

function splitReviewsFromText(text){
    // separa por líneas en blanco (una o más)
    return text.split(/\r?\n\s*\r?\n/).map(s => s.trim()).filter(Boolean);
}


async function analizarSentimiento() {
    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ reviews: reseñas })
    });

    const data = await response.json();
    console.log("Predicciones:", data.predictions);
}