# laptop recommendation project
This file provides an overview of the project, setup instructions, and deployment details.

# Budget-Friendly Laptop Search with AI Agent

A Streamlit web application that uses LangChain and FAISS to recommend laptops based on user queries, with filtering by brand name and price. The app leverages a dataset of laptops (e.g., from Flipkart) and provides an AI-driven conversational interface to suggest the best laptops.

## Features
- **Search and Filter**: Filter laptops by brand name and maximum price.
- **AI-Powered Recommendations**: Uses LangChain and FAISS to provide relevant laptop suggestions based on user queries.
- **Conversational Interface**: Chat with an AI agent to get personalized laptop recommendations.
- **Streamlit UI**: Interactive and user-friendly web interface.

## Dataset
The app expects a CSV file (`flipkart_laptop_cleaned.csv`) with the following columns:
- `Brand_Model`: Name of the laptop (e.g., "Dell Inspiron 14").
- `Prices`: Price in INR (numeric).
- `Description`: Description of the laptop.
- `Reviews`: Number of reviews (numeric).

You can provide your own dataset or use a sample one (not included in this repository for size reasons).

## Setup Instructions

### Prerequisites
- Python 3.11+
- Git
- A Hugging Face account (optional, for embeddings model access)
- Streamlit Community Cloud account (for deployment)

### Local Setup
1. **Clone the Repository**:
   
Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
Add Dataset: Place your flipkart_laptop_cleaned.csv file in the project root directory.
Run the App Locally:

streamlit run app1.py
