# Master Plan AI Assistant — POC

A rapid proof-of-concept based on the supplied **Urban Planning Permits Guideline (UPPG)**, focused on the **Master Plan Permit** journey.

## What this POC demonstrates
- Conversational Q&A over the UPPG
- Master Plan Permit process explanation
- Preliminary submission checklist generation
- Dubai 2040 alignment guidance
- Permit-type guidance (MPP / PP / GPP)
- Source/page-oriented guidance
- Two modes: **Demo** (no API key) and **AI + UPPG RAG**

## Run the presentation demo in 2 minutes

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Choose **Demo (no API key)**. Use the four buttons at the top to drive the demo.

## Run the real AI + RAG version

1. Create an OpenAI API key.
2. Set it as an environment variable:

```bash
# macOS/Linux
export OPENAI_API_KEY="..."

# Windows PowerShell
$env:OPENAI_API_KEY="..."
```

3. Start the app:

```bash
streamlit run app.py
```

4. Select **AI + UPPG RAG**.

On first AI run, the app uploads the supplied UPPG PDF to an OpenAI vector store and saves the vector store ID locally in `.poc_state.json`. Subsequent runs reuse it.

## Suggested 3-minute management demo

1. Click **Requirements** → show the required submission package.
2. Click **Process** → show the end-to-end journey.
3. Ask: `What is the difference between a Master Plan Permit and a Planning Permit?`
4. Ask: `Create a preliminary checklist for a new mixed-use master plan.`
5. Ask: `What happens if a modification exceeds 10%?`
6. Emphasize: **AI is grounded in the official UPPG and is designed to show the relevant source/page rather than invent requirements.**

## Next phase

For a production pilot, add:
- project intake questionnaire
- personalized checklist with Required / Conditional / Missing status
- document upload and preliminary completeness check
- structured rules engine for permit-type selection and thresholds
- official portal integration
- governance, versioning, audit trail, access control, and approved-source management

> This POC is preliminary guidance only. It does not issue an approval or replace the competent authority's review.
