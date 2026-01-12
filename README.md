- Threaded API calls - GUI stays responsive while waiting for responses
- Conversation history - Maintains context across messages
- Status indicator - Shows "Ready" / "Thinking..."
- Error handling - Displays errors and allows retry
- Reset Conversation - Clears history and starts fresh
- Clear Chat - Clears display only (keeps history)

Uses the gpt-4o-mini model by default.

To run, set your API key first:
```
export OPENAI_API_KEY="your-api-key"
source UpliftAI-venv/bin/activate
python console.py
```
