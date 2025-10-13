# API Autopilot

**Your AI Testing Co-Pilot**

An advanced AI-powered platform that generates comprehensive API test suites with security testing and schema validation. Supports both REST and GraphQL APIs with extensive test coverage including positive, negative, security, and edge case scenarios.

## 🚀 Features

### API Type Support
- ✅ **REST APIs** - Full HTTP method support
- ✅ **GraphQL APIs** - Queries, mutations, and subscriptions

### Multiple Input Methods
- cURL commands
- GraphQL queries/mutations
- HAR (HTTP Archive) files
- Plain text API documentation/contracts
- **Example Response** - Paste actual API response for precise assertions

### Advanced Generation
- ⚡ **Auto-Capture** - Automatically execute API calls to capture real responses
- 🔍 **Smart Validation** - Post-generation code quality checks
- 🔄 **Auto-Regeneration** - Automatically fixes incomplete or placeholder code
- 📋 **Schema Extraction** - Automatic JSON schema generation from responses
- 🎯 **Chunked Generation** - Handles large test suites without token exhaustion

### AI Provider Support
- OpenAI (GPT-4o)
- Anthropic (Claude 3.7 Sonnet)
- Google (Gemini 2.0 Flash)

### Framework Support
- **JavaScript**: Jest, Mocha, Cypress
- **Python**: pytest, requests + unittest
- **Java**: TestNG, JUnit

### Comprehensive Test Coverage (20-30+ test cases)
1. ✅ **Positive Tests** - Success scenarios with complete schema validation
2. ✅ **Negative Tests** - Input validation, malformed data, constraint violations
3. ✅ **Security Tests** - SQL injection, XSS, command injection attempts
4. ✅ **Authentication Tests** - Missing, invalid, expired tokens (401/403)
5. ✅ **Authorization Tests** - Permission validation and role-based access
6. ✅ **Edge Cases** - Boundary testing, special characters, large payloads
7. ✅ **Rate Limiting** - Concurrent requests and throttling (429)
8. ✅ **Resource Tests** - Non-existent resources, invalid endpoints (404, 405)

## 🎯 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB (running instance)
- Yarn package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ApiAutopilot
```

2. **Setup Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup Frontend**
```bash
cd frontend
yarn install
```

4. **Configure Environment**
Create `.env` file in `backend/` directory:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=api_autopilot
CORS_ORIGINS=http://localhost:3000
```

### Running the Application

#### Option 1: Use the restart script (Recommended)
```bash
./restart.sh
```
This will clear cache and start both servers.

#### Option 2: Manual start
**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
yarn start
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 How to Use

1. **Select AI Provider**: Choose from OpenAI, Anthropic, or Gemini
2. **Enter API Key**: Provide your AI provider API key
3. **Choose Test Framework**: Select your preferred testing framework
4. **Input API Details**: Use one of five methods:
   - Paste a cURL command
   - Paste GraphQL query/mutation
   - Upload/paste HAR file content
   - Describe your API in plain text
   - Paste actual API response (for precise assertions)
5. **Optional Enhancements**:
   - ⚡ Enable **Auto-Capture** to automatically execute the API and capture real responses
   - 📋 Provide **Example Response** for more accurate field-level assertions
6. **Generate**: Click "Generate Test Scripts" button
7. **Download/Copy**: Get your ready-to-use test scripts with NO TODOs or placeholders

## ✨ Quality Guarantees

### Zero Placeholders Promise
- ❌ NO TODO comments
- ❌ NO "Add more tests" suggestions
- ❌ NO placeholder values (UPDATE_THIS, REPLACE_ME, etc.)
- ❌ NO generic assertions (toBeTruthy, toBeDefined only)
- ❌ NO incomplete implementations
- ✅ ONLY production-ready, immediately executable code

### Smart Generation Process
1. **Pre-Generation**: Captures actual API responses when enabled
2. **Generation**: Uses AI with explicit anti-TODO instructions
3. **Post-Validation**: Scans for TODOs, placeholders, and generic code
4. **Auto-Regeneration**: Automatically fixes any detected issues
5. **Final Output**: Complete, specific assertions for every response field

### When You Provide Example Response
- Exact field names from your API
- Precise data type validations
- Specific format checks (email, UUID, date, etc.)
- Nested object structure validation
- Array item type checking
- No guesswork or assumptions

## 🔑 API Keys

You need to provide your own API key for one of these services:
- **OpenAI**: Get key from https://platform.openai.com/api-keys
- **Anthropic**: Get key from https://console.anthropic.com/
- **Google AI**: Get key from https://aistudio.google.com/

## 🛠️ Technical Stack

- **Frontend**: React 19 + TailwindCSS + shadcn/ui
- **Backend**: FastAPI (Python)
- **Database**: MongoDB
- **AI Integration**: Multi-provider AI library

## 📝 Project Structure

```
ApiAutopilot/
├── backend/
│   ├── server.py          # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   └── components/   # UI components
│   ├── package.json      # Node dependencies
│   └── public/
├── tests/                # Test files
├── restart.sh           # Cache clear & restart script
└── README.md            # This file
```

## 🧪 Testing

Run backend tests:
```bash
cd backend
pytest
```

Run frontend tests:
```bash
cd frontend
yarn test
```

## 🔧 Maintenance Scripts

### Clear Cache and Restart
```bash
./restart.sh
```
This script:
- Stops all running servers
- Clears Python cache (`__pycache__`, `.pyc`, `.pytest_cache`)
- Clears frontend cache (`build/`, `node_modules/.cache/`)
- Prompts to restart servers
- Shows server URLs and log locations

## 📌 Notes

- No authentication required (stateless generation)
- API keys are not stored
- All processing happens in real-time
- Test scripts are immediately ready to use
- Chunked generation prevents token exhaustion
- Auto-validation ensures code quality

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests pass before submitting
- Documentation is updated

## 📄 License

[Add your license here]

## 🆘 Support

For issues or questions:
- Check API documentation: http://localhost:8000/docs
- Review generated test scripts
- Check logs: `backend.log` and `frontend.log`

---

**API Autopilot** - Your AI Testing Co-Pilot ✈️
