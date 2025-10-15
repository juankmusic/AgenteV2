const sendBtn = document.getElementById('sendBtn');
const input = document.getElementById('userInput');
const chatBox = document.getElementById('chat-box');

sendBtn.addEventListener('click', async () => {
  const msg = input.value.trim();
  if (!msg) return;
  addMessage('user', msg);
  input.value = '';

  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message: msg })
  });
  const data = await res.json();
  addMessage('bot', data.reply);
});

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = role;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}
