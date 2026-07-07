document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');

    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    submitBtn.disabled = true;

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || 'Server error');
        }

        const result = await response.json();

        const resultDiv = document.getElementById('result');
        const resultIcon = document.getElementById('resultIcon');
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const progressBar = document.getElementById('progressBar');
        const probDetails = document.getElementById('probDetails');

        // Reset progress bar for re-animation
        progressBar.style.width = '0%';
        progressBar.className = 'progress-bar';

        resultDiv.classList.remove('hidden');

        const isApproved = result.prediction === 'Approved';

        resultIcon.textContent = isApproved ? '✅' : '❌';
        predictionText.textContent = `Loan ${result.prediction}!`;
        predictionText.className = isApproved ? 'approved' : 'rejected';
        confidenceText.textContent = `Model Confidence: ${result.confidence}%`;

        progressBar.classList.add(isApproved ? 'approved-bar' : 'rejected-bar');

        probDetails.innerHTML = `
            <span class="prob-approved">Approved: ${result.probability.approved}%</span>
            <span class="prob-rejected">Rejected: ${result.probability.rejected}%</span>
        `;

        // Animate after a tick so the browser picks up the transition
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                progressBar.style.width = `${result.confidence}%`;
            });
        });

        // Scroll result into view
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Prediction error:', error);
    } finally {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        submitBtn.disabled = false;
    }
});
