# Chat Bot LLM
## How to Run the Project

This guide will help you set up and run the Chat Bot LLM project on your local machine.

### Step 1: Clone the Project
```bash
git clone https://github.com/bon-dawson/chat-bot-llm.git
cd chat-bot-llm
```

### Step 2: Set Up OpenAI API Key
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Step 3: Create a Virtual Environment and Install Dependencies

**Option 1: Using Python's built-in venv**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

**Option 2: Using UV package manager**

If you have the UV package manager installed ([download here](https://github.com/astral-sh/uv)), you can use:
```bash
uv venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Step 4: Run the Backend Server

**With Python:**
```bash
python ./backend/app.py
```

**With UV:**
```bash
uv run ./backend/app.py
```

### Step 5: Run the Frontend Application
Open a new terminal window while keeping the backend running, then:

```bash
cd frontend/
npm install
npm start
```

### Step 6: Access the Application
Open your web browser and navigate to:
http://localhost:3000

You can now interact with the chatbot!
