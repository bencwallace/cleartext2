let contextSize = 4096;

function getSelectionWithContext() {
    const selection = window.getSelection();

    const range = selection.getRangeAt(0);
    const selectedText = range.toString();

    const fullText = document.body.innerText;
    const selectionStart = fullText.indexOf(selectedText);
    const selectionEnd = selectionStart + selectedText.length;

    const contextStart = Math.max(0, selectionStart - contextSize);
    const contextEnd = Math.min(fullText.length, selectionEnd + contextSize);

    const beforeContext = fullText.substring(contextStart, selectionStart);
    const afterContext = fullText.substring(selectionEnd, contextEnd);

    return {
        before: beforeContext,
        selection: selectedText,
        after: afterContext
    };
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "getSelection") {
        const selectionWithContext = getSelectionWithContext();
        sendResponse({
            selection: selectionWithContext
        });
    }
    return true;
});