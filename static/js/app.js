function analyzeEmail() {
    const emailInput = document.getElementById('emailInput');
    const emailText = emailInput.value.trim();
    
    if (!emailText) {
        showError('Please enter an email to analyze');
        return;
    }
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    loader.style.display = 'inline-block';
    
    // Hide previous results/errors
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    
    // Send request to server
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email_text: emailText })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during analysis');
        }
    })
    .catch(error => {
        showError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    })
    .finally(() => {
        // Reset button state
        analyzeBtn.disabled = false;
        btnText.textContent = 'Analyze Email';
        loader.style.display = 'none';
    });
}

function displayResults(data) {
    const resultsSection = document.getElementById('results');
    resultsSection.style.display = 'block';
    
    // Update risk level
    const riskLevel = document.getElementById('riskLevel');
    const riskIndicator = document.getElementById('riskIndicator');
    const riskFill = document.getElementById('riskFill');
    
    riskLevel.textContent = data.risk_level;
    riskLevel.className = 'risk-level';
    
    // Set risk level styling
    let riskClass = '';
    let fillWidth = '0%';
    
    if (data.risk_level.includes('High')) {
        riskClass = 'high-risk';
        riskFill.className = 'risk-fill high';
        fillWidth = '90%';
    } else if (data.risk_level.includes('Medium')) {
        riskClass = 'medium-risk';
        riskFill.className = 'risk-fill medium';
        fillWidth = '60%';
    } else {
        riskClass = 'low-risk';
        riskFill.className = 'risk-fill low';
        fillWidth = '30%';
    }
    
    riskLevel.classList.add(riskClass);
    
    // Animate risk meter
    setTimeout(() => {
        riskFill.style.width = fillWidth;
    }, 100);
    
    // Update details
    document.getElementById('prediction').textContent = 
        data.prediction.charAt(0).toUpperCase() + data.prediction.slice(1);
    document.getElementById('confidence').textContent = data.confidence;
    document.getElementById('phishingProb').textContent = data.phishing_probability;
    document.getElementById('legitimateProb').textContent = data.legitimate_probability;
    
    // Update suspicious patterns
    const patternsSection = document.getElementById('patterns');
    const patternsList = document.getElementById('patternsList');
    
    if (data.suspicious_patterns && data.suspicious_patterns.length > 0) {
        patternsSection.style.display = 'block';
        patternsList.innerHTML = '';
        
        data.suspicious_patterns.forEach(pattern => {
            const li = document.createElement('li');
            li.textContent = pattern.charAt(0).toUpperCase() + pattern.slice(1);
            patternsList.appendChild(li);
        });
    } else {
        patternsSection.style.display = 'none';
    }
    
    // Update recommendation
    const recommendation = document.getElementById('recommendation');
    recommendation.textContent = data.recommendation;
    recommendation.className = 'recommendation';
    
    if (data.risk_level.includes('High')) {
        recommendation.classList.add('high-risk');
    } else if (data.risk_level.includes('Medium')) {
        recommendation.classList.add('medium-risk');
    } else {
        recommendation.classList.add('low-risk');
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    
    // Hide error after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

// Allow Enter key to submit (Ctrl+Enter for multiline)
document.getElementById('emailInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        analyzeEmail();
    }
});