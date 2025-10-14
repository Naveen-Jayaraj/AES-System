document.addEventListener('DOMContentLoaded', () => {
    const scoreButton = document.getElementById('score-button');
    const essayInput = document.getElementById('essay-input');
    const resultArea = document.getElementById('result-area');
    const scoreValue = document.getElementById('score-value');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');

    scoreButton.addEventListener('click', async () => {
        const essayText = essayInput.value;

        // Reset UI
        resultArea.classList.remove('hidden');
        scoreValue.classList.add('hidden');
        errorMessage.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');

        try {
            const response = await fetch('/score-essay', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ essay: essayText }),
            });
            
            loadingSpinner.classList.add('hidden');

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const data = await response.json();
            scoreValue.textContent = data.score.toFixed(1);
            scoreValue.classList.remove('hidden');

        } catch (error) {
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('hidden');
        }
    });
});