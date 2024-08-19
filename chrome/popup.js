let endpoint = 'http://localhost:8000/'

document.addEventListener('DOMContentLoaded', function() {
  chrome.tabs.query({
      active: true,
      currentWindow: true
  }, function(tabs) {
      chrome.tabs.sendMessage(tabs[0].id, {
          action: "getSelection"
      }, function(response) {
          if (response && response.selection) {
              sendToEndpoint(response.selection);
          } else {
              document.getElementById('processedText').textContent = "No text selected";
          }
      });
  });
});

function sendToEndpoint(selectionWithContext) {
  fetch(endpoint, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(selectionWithContext),
      })
      .then(response => response.text())
      .then(data => {
          document.getElementById('processedText').textContent = data;
      })
      .catch((error) => {
          console.error('Error:', error);
          document.getElementById('processedText').textContent = 'Error processing text';
      });
}