import ollama

# Load the model
# Note: Make sure the model is running on your local machine with `ollama run nomic-embed-text:latest`
model = "nomic-embed-text:latest"

# Get a sample embedding
sample_text = "This is a test sentence."
response = ollama.embeddings(model=model, prompt=sample_text)
embedding = response["embedding"]

# Print the length of the embedding list
print(f"The dimension of the embedding is: {len(embedding)}")