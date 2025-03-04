# autoshort
automatic shorts generator from gamplay videos. 
#!/bin/bash
# Installation and setup script for Gameplay YouTube Shorts Generator

echo "Starting installation of Gameplay YouTube Shorts Generator - Open Source Version"

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv shorts_env
source shorts_env/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -U pip
pip install streamlit torch torchvision torchaudio transformers spacy pydub moviepy python-dotenv scikit-learn aiohttp toml pyttsx3

# Install spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_md

# Create download_models.py script
echo "Creating model downloader script..."
cat > download_models.py << 'EOF'
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_id, model_type="llm"):
    """
    Download a model from Hugging Face
    """
    print(f"Downloading {model_type} model: {model_id}...")
    
    if model_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        # Save to disk
        model_dir = os.path.join("models", model_id.split("/")[-1])
        os.makedirs(model_dir, exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # LLM models (smaller first for systems with limited resources)
    print("Downloading text generation models...")
    download_model("mistralai/Mistral-7B-Instruct-v0.2", "llm")
    
    # Create gameplay videos directory
    gameplay_dir = "./gameplay"
    if not os.path.exists(gameplay_dir):
        os.makedirs(gameplay_dir)
        print(f"Created gameplay videos directory: {gameplay_dir}")
    else:
        print(f"Gameplay videos directory already exists: {gameplay_dir}")
    
    print("All models downloaded successfully!")
    print("Note: If you have limited resources, you may want to use smaller models like 'gpt2'.")
EOF

# Create requirements.txt file
echo "Creating requirements file..."
cat > requirements.txt << 'EOF'
streamlit==1.29.0
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
transformers==4.35.0
spacy==3.7.1
pydub==0.25.1
moviepy==1.0.3
python-dotenv==1.0.0
scikit-learn==1.3.2
aiohttp==3.8.6
toml==0.10.2
pyttsx3==2.90
EOF

# Install Gentle
echo "Setting up Gentle for subtitle alignment..."
git clone https://github.com/lowerquality/gentle.git
cd gentle
./install.sh
cd ..

# Create gameplay directory
mkdir -p gameplay
echo "Created 'gameplay' directory. Please add your gameplay videos to this folder."

# Create a startup script
echo "Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source shorts_env/bin/activate

# Start Gentle in the background
cd gentle
python serve.py &
GENTLE_PID=$!
cd ..

echo "Gentle server started on port 8765"

# Start the Streamlit app
echo "Starting Gameplay YouTube Shorts Generator..."
streamlit run app.py

# When Streamlit is killed, also kill Gentle
kill $GENTLE_PID
EOF

chmod +x start.sh

echo "Installation complete! To run the app:"
echo "1. Add your gameplay videos to the 'gameplay' folder"
echo "2. Download the language models: python download_models.py"
echo "3. Start the application: ./start.sh"
