const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');

async function sendMessage() {
    const message = userInput.value.trim();
    if (message === "") return;

    // 1. Add User Message to Chat
    appendMessage(message, 'user-message');
    userInput.value = '';

    // 2. Send to Backend
    try {
        const response = await fetch('http://127.0.0.1:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message })
        });

        const data = await response.json();

        // 3. Add Bot Response to Chat
        if (data.answer) {
            appendMessage(data.answer, 'bot-message');
        } else {
            appendMessage("Error: Could not get response.", 'bot-message');
        }
    } catch (error) {
        console.error('Error:', error);
        appendMessage("Error: Server is unreachable.", 'bot-message');
    }
}

function appendMessage(text, className) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    chatHistory.appendChild(messageDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Allow pressing "Enter" to send
userInput.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});