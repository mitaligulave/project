import streamlit as st
import pandas as pd
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import FakeListLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os


# Function to load and clean CSV data
def load_csv_data():
    try:
        df = pd.read_csv('flipkart_laptop_cleaned.csv')
        # Clean data
        df['Description'] = df['Description'].fillna('Not available')
        df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce').fillna(0)
        df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        st.error("Error: 'flipkart_laptop_cleaned.csv' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to filter data by name and price
def filter_data(df, brand_name=None, max_price=None):
    filtered_df = df.copy()
    if brand_name:
        filtered_df = filtered_df[filtered_df['Brand_Model'].str.contains(brand_name, case=False, na=False)]
    if max_price:
        filtered_df = filtered_df[filtered_df['Prices'] <= max_price]
    return filtered_df

# Function to create LangChain vector store
@st.cache_resource
def create_vector_store(descriptions, df):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create documents for LangChain
    documents = [
        f"Brand_Model: {row['Brand_Model']}\nPrice: ₹{row['Prices']}\nDescription: {row['Description']}\nReviews: {row['Reviews']}"
        for _, row in df.iterrows()
    ]
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store, embeddings

# Function to initialize LangChain conversational chain
@st.cache_resource
def initialize_conversational_chain(_vector_store):
    # Use a fake LLM for response formatting (since we're relying on retrieval)
    llm = FakeListLLM(responses=["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"])

    # Define prompt template for response formatting
    prompt_template = """Based on the following laptop information, provide a natural language response recommending laptops that match the user's query. Include the brand, model, price, description, reviews, and explain why the top recommendation is the best choice.

    Context: {context}

    User Query: {question}

    Response:
    I found {num_results} laptop(s) that might match your query:

    {results}

    **Recommended Best Choice:**
    {best_laptop}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "num_results", "results", "best_laptop"])

    # Initialize memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

# Function to format results and find best laptop
def format_results(docs, df, query_embedding, embeddings):
    results = []
    for doc in docs:
        # Extract metadata from document
        content = doc.page_content
        brand_model = content.split('\n')[0].split(': ')[1]
        price = float(content.split('\n')[1].split(': ₹')[1])
        description = content.split('\n')[2].split(': ')[1]
        reviews = float(content.split('\n')[3].split(': ')[1])
        
        # Calculate similarity score
        doc_embedding = embeddings.embed_query(content)
        similarity_score = 1 / (1 + np.linalg.norm(np.array(query_embedding) - np.array(doc_embedding)))
        
        results.append({
            'Brand_Model': brand_model,
            'Price': price,
            'Description': description,
            'Reviews': reviews,
            'Similarity_Score': similarity_score
        })
    
    # Find best laptop (same logic as original)
    if not results:
        return None, []
    
    max_reviews = max(result['Reviews'] for result in results) if results else 1
    best_laptop = None
    highest_score = -1
    
    for result in results:
        normalized_reviews = result['Reviews'] / max_reviews if max_reviews > 0 else 0
        composite_score = 0.7 * result['Similarity_Score'] + 0.3 * normalized_reviews
        
        if composite_score > highest_score:
            highest_score = composite_score
            best_laptop = result
    
    # Format results for response
    formatted_results = ""
    for i, result in enumerate(results, 1):
        formatted_results += (f"{i}. **{result['Brand_Model']}** (₹{int(result['Price'])})\n"
                             f"   - Description: {result['Description']}\n"
                             f"   - Reviews: {result['Reviews']}\n"
                             f"   - Similarity Score: {result['Similarity_Score']:.4f}\n\n")
    
    best_laptop_str = (f"**{best_laptop['Brand_Model']}** (₹{int(best_laptop['Price'])})\n"
                       f"   - Description: {best_laptop['Description']}\n"
                       f"   - Reviews: {best_laptop['Reviews']}\n"
                       f"   - Similarity Score: {best_laptop['Similarity_Score']:.4f}\n"
                       f"   - Why: This laptop has the best combination of matching your query and customer reviews.")
    
    return best_laptop, formatted_results, best_laptop_str

# Streamlit app
def main():
    st.title("Budget-Friendly Laptop Search with AI Agent (Powered by LangChain)")
    st.write("Search for laptops or chat with our AI agent about laptop recommendations.")

    # Load data
    df = load_csv_data()
    if df is None:
        return
    
    # Filter options
    st.subheader("Filter Laptops")
    brand_name = st.text_input("Enter Brand Name (e.g., Dell, HP):", placeholder="Leave blank for all brands")
    max_price = st.number_input("Maximum Price (₹):", min_value=0, value=0, step=1000, 
                               help="Enter 0 to include all prices")
    
    # Apply filters
    filtered_df = filter_data(df, brand_name if brand_name.strip() else None, 
                            max_price if max_price > 0 else None)
    
    if filtered_df.empty:
        st.warning("No laptops match the specified filters. Try adjusting the brand or price.")
        return
    
    # Create vector store with filtered data
    with st.spinner("Generating embeddings and vector store..."):
        descriptions = filtered_df['Description'].tolist()
        vector_store, embeddings = create_vector_store(descriptions, filtered_df)
    
    # Initialize conversational chain
    chain = initialize_conversational_chain(vector_store)
    
    # AI Agent chat interface
    st.subheader("Chat with AI Agent")
    user_query = st.text_input("Ask the AI agent about laptops:", 
                              placeholder="E.g., Recommend a lightweight laptop with good battery life.")
    
    if st.button("Ask AI"):
        if user_query.strip():
            with st.spinner("AI is thinking..."):
                # Get retrieved documents
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(user_query)
                
                # Format results
                query_embedding = embeddings.embed_query(user_query)
                best_laptop, formatted_results, best_laptop_str = format_results(docs, filtered_df, query_embedding, embeddings)
                
                if not docs:
                    response = "I'm sorry, I couldn't find any laptops matching your query. Try specifying different requirements or adjusting the filters."
                else:
                    # Simulate chain response
                    response = (f"I found {len(docs)} laptop(s) that might match your query:\n\n"
                                f"{formatted_results}\n"
                                f"**Recommended Best Choice:**\n"
                                f"{best_laptop_str}")
                
                st.markdown(response)
        else:
            st.error("Please enter a valid question or requirement.")

if __name__ == "__main__":
    main()