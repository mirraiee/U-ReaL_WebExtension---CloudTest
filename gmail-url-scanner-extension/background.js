// Background Service Worker for Gmail URL Scanner

// Backend API configuration
const BACKEND_URL = 'http://3.0.57.177:5000'; // Use your EC2 instance's public IP

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkURL') {
    checkURLWithBackend(request.url)
      .then(result => sendResponse(result))
      .catch(error => {
        console.error('Backend error:', error);
        sendResponse({ is_malicious: false, confidence: 0, error: error.message });
      });
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'openExtension') {
    chrome.action.openPopup();
  }
});

// Check URL with Python backend
async function checkURLWithBackend(url) {
  try {
    const response = await fetch(`${BACKEND_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });

    if (!response.ok) {
      throw new Error(`Backend server error: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    // If backend is not available, return safe default
    console.warn('Backend not available, marking as benign:', error);
    return { is_malicious: false, confidence: 0, error: 'Backend unavailable' };
  }
}

// Handle extension installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('Gmail URL Scanner installed');
  
  // Initialize storage
  chrome.storage.local.set({
    maliciousUrls: [],
    settings: {
      threshold: 0.4,
      enableNotifications: true
    }
  });
});