# Web Extraction Agent

## Executive Summary
This project is an advanced, AI-powered web extraction pipeline designed to scrape structured data from complex real estate listings. Instead of relying on fragile, hard-coded HTML DOM parsing that breaks whenever a website updates its layout, this agent uses Large Language Models (LLMs) to intelligently understand and extract information directly from raw webpage text. It visits pages securely using stealth-configured browser automation, cleans the noise from the DOM, and leverages schema-driven extraction to reliably output clean, persistent data.

## System Architecture
The extraction pipeline is built on the following core technologies:
- **Playwright (Async):** Drives a headless/headed Chromium browser configured with stealth settings to bypass bot detection, navigate through pagination, and trigger lazy-loaded elements.
- **Groq/Gemini API (LLM Extraction):** Employs fast, state-of-the-art LLMs (like `llama-3.3-70b-versatile` via Groq) to analyze the raw text semantics of the page and dynamically locate the required fields, making the scraper robust against minor CSS/HTML structure changes.
- **Pydantic:** Strictly defines the extraction schema (e.g., fields, data types, and logical rules like currency conversion and boolean flags) to guarantee that the LLM output is consistently structured and validated.
- **PostgreSQL:** A robust relational database (containerized via Docker) used to cleanly persist the extracted and validated schema listings for downstream analysis.

## Prerequisites
- Python 3.10+
- Docker & Docker Compose
- API Keys: A Groq API Key (or Gemini API Key for fallback)
- Git

## Local Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Web_Extraction_Agent
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   *(Ensure you have `pip` updated)*
   ```bash
   pip install playwright pydantic groq python-dotenv asyncpg
   playwright install chromium
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Start the Database Services:**
   Launch the local PostgreSQL database using Docker Compose:
   ```bash
   docker-compose up -d
   ```
   *Note: This spins up a Postgres 15 database on port `5433` with the configured user `Ben` and database `WebExAg`.*

6. **Run the Extraction Agent:**
   Execute the scraping pipeline:
   ```bash
   python agent.py
   ```
   The agent will launch a browser session, securely navigate the target paginated URLs, intelligently construct valid extractions via Groq, and persist them into the local Postgres instance.
