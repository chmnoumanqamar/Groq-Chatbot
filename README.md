# Grok Chatbot (xAI API)

Same dark UI, but this version talks to the real Grok models from xAI.

## Files
- `app.py` — the chatbot app
- `requirements.txt` — packages needed
- `.env.example` — template for your key

---

## Step 1: Get your xAI (Grok) API key

1. Go to https://console.x.ai
2. Sign up / log in.
3. Go to the **API Keys** section.
4. Click **"Create API Key"**, name it, and copy it.

⚠️ Note: xAI's API is **paid** (usage-based) — check current pricing on their console. It is a different key from Hugging Face and only works with xAI's models.

## Step 2: Add the key to your project

1. Rename `.env.example` to `.env`
2. Paste your key:
```
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
3. Save the file.

## Step 3: Install and run

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Changing the model

Edit this part of `app.py`:
```python
MODEL_OPTIONS = {
    "Grok Fast": "grok-4-fast",
    "Grok Standard": "grok-4",
    "Grok Code": "grok-code-fast-1",
}
```
Check https://docs.x.ai for the current list of available model names.
