# Master Plan AI Assistant — Online Demo

## Recommended deployment
Use **Streamlit Community Cloud** for the fastest management/demo deployment.

### 1. Create a GitHub repository
Create a repository such as:
`master-plan-ai-assistant`

Upload:
- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `UPPG_Master_Plan_Guide.pdf`

### 2. Deploy
Open Streamlit Community Cloud and choose:
- Repository: your GitHub repository
- Main file: `app.py`

### 3. Add the OpenAI API key
In the deployed app settings, add a secret:

```toml
OPENAI_API_KEY = "YOUR_KEY"
OPENAI_MODEL = "gpt-5-mini"
```

Do NOT put the API key inside `app.py`, GitHub, or the PDF.

### 4. First launch
Open the generated Streamlit URL and select:
**AI + UPPG RAG**

The application will create/use a vector store and ground answers in the uploaded UPPG PDF.

## Public deployment recommendation
For a genuinely public website, keep the source repository private if possible and keep the UPPG PDF on the server rather than exposing it as a downloadable public asset.

The current application also has:
**Demo (no API key)**
which is useful for testing the interface before connecting the AI.

## Management demo flow
1. Requirements
2. Process
3. Permit type
4. Checklist
5. Ask about a modification exceeding the threshold
6. Show source/page references and explain the future completeness-checking phase.
