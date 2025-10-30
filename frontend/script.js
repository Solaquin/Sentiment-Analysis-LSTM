const textarea = document.getElementById('review-text');
const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const reviewsContainer = document.getElementById('reviews-container');

const totalEl = document.getElementById('total-reviews');
const posEl = document.getElementById('positive-reviews');
const negEl = document.getElementById('negative-reviews');

function renderReviews(reviews, data)
{
    reviewsContainer.innerHTML = '';
    let positives = 0;
    let negatives = 0;

    reviews.forEach((r, idx) => {
        const percentage = data[idx];
        const sentiment = percentage >= 0.5 ? 'positive' : 'negative';
        if (sentiment == 'positive') positives +=1 ;
        else negatives +=1;
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

    const total = reviews.length;
    totalEl.textContent = total;
    posEl.textContent = positives;
    negEl.textContent = negatives;
}

function splitReviewsFromText(text){
    // separa por líneas en blanco (una o más)
    return text.split(/\r?\n\s*\r?\n/).map(s => s.trim()).filter(Boolean);
}

async function analizarSentimiento() {

    const raw = textarea.value;
    if (raw === '') return;

    const reviews = splitReviewsFromText(raw);

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ reviews: reviews })
    });

    const data = await response.json();
    renderReviews(reviews, data.predictions)
    console.log("Predicciones:", data.predictions);
}

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if(!file) return;

    const text = await file.text();
    let reviews = [];
    if (file.name.toLowerCase().endsWith('.json'))
    {
        try
        {
            const parsed = JSON.parse(text);

            if (Array.isArray(parsed))
            {
                reviews = parsed.map(p => typeof p === 'string' ? p : JSON.stringify(p));
            } 
            else if (parsed && typeof parsed === 'object')
            {
                // intentar detectar un campo "reviews" o similar
                if (Array.isArray(parsed.reviews)) reviews = parsed.reviews.map(String);
                else reviews = [JSON.stringify(parsed)];
            } 
            else 
            {
                reviews = [String(parsed)];
            }

        } catch(err) {
            // si no es JSON válido, tratar como texto plano
            reviews = splitReviewsFromText(text);
        }
    } 
    else 
    {
        reviews = splitReviewsFromText(text);
    }

    // copiar al textarea separando con una linea en blanco
    textarea.value = reviews.join('\n\n');

    // limpiar el input para permitir subir el mismo archivo otra vez si es necesario
    fileInput.value = '';
});

analyzeBtn.addEventListener('click', analizarSentimiento);