document.getElementById('loanForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.textContent = 'Predicting...';
    submitBtn.disabled = true;
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const result = await response.json();
        
        const resultDiv = document.getElementById('result');
        const predictionText = document.getElementById('predictionText');
        const confidenceText = document.getElementById('confidenceText');
        const progressBar = document.getElementById('progressBar');
        
        resultDiv.classList.remove('hidden');
        predictionText.textContent = `Loan ${result.prediction}!`;
        confidenceText.textContent = `Confidence: ${result.confidence}%`;
        
        if (result.prediction === 'Approved') {
            predictionText.className = 'approved';
            progressBar.style.backgroundColor = 'var(--success)';
        } else {
            predictionText.className = 'rejected';
            progressBar.style.backgroundColor = 'var(--danger)';
        }
        
        // Small delay to allow CSS transition to work
        setTimeout(() => {
            progressBar.style.width = `${result.confidence}%`;
        }, 100);
        
    } catch (error) {
        alert('Error connecting to the server. Please ensure the backend is running.');
        console.error('Error:', error);
    } finally {
        submitBtn.textContent = 'Predict Eligibility';
        submitBtn.disabled = false;
    }
});
