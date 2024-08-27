import os
import replicate
import time
import shutil
from zipfile import ZipFile
from dotenv import load_dotenv
from datetime import datetime
from httpx import RemoteProtocolError, ReadTimeout, ConnectTimeout, HTTPStatusError, RequestError
from huggingface_hub import HfApi, create_repo, whoami
from gdrive_large_file_utils import generate_direct_download_link

# Load environment variables from .env file
load_dotenv()

# GLOBAL VARIABLES for easy tweaks
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
REPLICATE_OWNER = "tillo13"  # Replicate username
HUGGING_FACE_OWNER = "andytillo1"  # Hugging Face username
BASE_MODEL_NAME = "flux1-dev-lora"
STEPS = 1000
LORA_RANK = 16
OPTIMIZER = "adamw8bit"
BATCH_SIZE = 1
RESOLUTION = "512, 768, 1024"
AUTOCAPTION = True
AUTOCAPTION_PREFIX = "A photo of andytillo man in his mid 40s with a large full beard starting to gray, "
AUTOCAPTION_SUFFIX = " in the style of andytillo."
TRIGGER_WORD = "andytillo"
LEARNING_RATE = 0.0004

# Define the input and output directories for image preparation
INPUT_DIR = 'initial_images/'
OUTPUT_DIR = 'prepared_images/'
ZIP_FILE_NAME = 'prepared_images.zip'

# Use standard Google Drive link for the images zip file
GOOGLE_DRIVE_PATH_TO_IMAGES_ZIP = (
    "https://drive.google.com/file/d/1JRwuj-fUkGgRlLk78jxnXivqRsNZ3bXP/view?usp=drive_link"
)

HF_REPO_ID = "andytillo1/flux-lora"
VISIBILITY = "public"  # Or "private" for a private model
HARDWARE = "gpu-t4"  # Replicate will override this for fine-tuned models
DESCRIPTION = "A fine-tuned FLUX.1 model via andytillo"
USE_CAPTIONS = False  # Set to True if you want to enable Llava autocaptioning
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
VERSION = "ostris/flux-dev-lora-trainer:7f53f82066bcdfb1c549245a624019c26ca6e3c8034235cd4826425b61e77bec"

# Ensure tokens are correctly loaded
if not REPLICATE_API_TOKEN:
    print("Error: REPLICATE_API_TOKEN is not set in the .env file")
    exit(1)

if not HUGGING_FACE_TOKEN:
    print("Error: HUGGING_FACE_TOKEN is not set in the .env file")
    exit(1)

# Initialize the Replicate client with your API token
def initialize_client():
    print("Initializing Replicate client with API token...")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    print("Replicate client initialized successfully.")
    return client

# Generate a unique model name to avoid conflicts
def generate_model_name(base_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_name}-{timestamp}"

# Create a new model to store your fine-tuned weights
def create_model(client, owner, name, visibility, hardware, description):
    print(f"Creating a new model on Replicate to store fine-tuned weights... (Model Name: {name})")
    try:
        model = client.models.create(
            owner=owner,
            name=name,
            visibility=visibility,
            hardware=hardware,
            description=description
        )
        print(f"Model created: {model.name}")
        print(f"Model URL: https://replicate.com/{model.owner}/{model.name}")
        return model
    except replicate.exceptions.ReplicateError as e:
        print(f"Error creating model on Replicate: {str(e)}")
        if "already exists" in str(e.detail):
            print(f"Model creation failed: {e.detail}")
            exit(1)
        else:
            raise

# Create a Hugging Face repository if it doesn't exist
def create_hf_repo(hf_token, repo_id):
    api = HfApi()
    user = whoami(token=hf_token)
    username = user["name"]

    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    try:
        api.create_repo(repo_id=repo_id, token=hf_token, private=(VISIBILITY == "private"), exist_ok=True)
        print(f"Repository '{repo_id}' created or already exists.")
    except Exception as e:
        print(f"Error creating Hugging Face repository: {e}")
        exit(1)

# Prepare images and zip them
def prepare_images(input_dir, output_dir, token):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    # List all files in the input directory
    files = os.listdir(input_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for i, file_name in enumerate(image_files):
        # Construct the new file name
        new_file_name = f"{i}_A_photo_of_{token}.jpg"

        # Construct the full file paths
        old_path = os.path.join(input_dir, file_name)
        new_path = os.path.join(output_dir, new_file_name)

        # Copy and rename the file
        shutil.copyfile(old_path, new_path)
        print(f"Copied {old_path} to {new_path}")

    print(f"Prepared {len(image_files)} images and saved in '{output_dir}'.")

def zip_images(output_dir, zip_file_name):
    with ZipFile(zip_file_name, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    zipf.write(os.path.join(root, file), file)
                    print(f"Added {file} to {zip_file_name}")
    print(f"Images have been zipped into '{zip_file_name}'.")

# Initialize the training process on Replicate
def start_training(client, model, path_to_images, steps, lora_rank, optimizer, batch_size, resolution,
                   autocaption, trigger_word, learning_rate, hf_token, hf_repo_id, version, retry_delay, max_retries):
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1} to start the training process...")

        try:
            print("Initializing training on Replicate...")
            training = client.trainings.create(
                version=version,
                input={
                    "input_images": path_to_images,
                    "steps": steps,
                    "lora_rank": lora_rank,
                    "optimizer": optimizer,
                    "batch_size": batch_size,
                    "resolution": resolution,
                    "autocaption": autocaption,
                    "trigger_word": trigger_word,
                    "learning_rate": learning_rate,
                    "hf_token": hf_token,
                    "hf_repo_id": hf_repo_id,
                },
                destination=f"{model.owner}/{model.name}"
            )

            print(f"Training initiation payload:\n"
                  f"{{\n"
                  f"  'input_images': '{path_to_images}',\n"
                  f"  'steps': {steps},\n"
                  f"  'lora_rank': {lora_rank},\n"
                  f"  'optimizer': '{optimizer}',\n"
                  f"  'batch_size': {batch_size},\n"
                  f"  'resolution': '{resolution}',\n"
                  f"  'autocaption': {autocaption},\n"
                  f"  'trigger_word': '{trigger_word}',\n"
                  f"  'learning_rate': {learning_rate},\n"
                  f"  'hf_token': '{hf_token}',\n"
                  f"  'hf_repo_id': '{hf_repo_id}'\n"
                  f" }}")

            print("Training has started successfully.")
            print(f"Training ID: {training.id}")
            print(f"Training URL: https://replicate.com/p/{training.id}")
            print("Monitor the training progress using the URL provided above. Refresh the page periodically to check the status.")
            return training

        except (RemoteProtocolError, ReadTimeout, ConnectTimeout, HTTPStatusError, RequestError) as e:
            print(f"Attempt {attempt + 1} failed due to network issue: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                exit(1)
        except replicate.exceptions.ReplicateError as e:
            print(f"Attempt {attempt + 1} failed due to Replicate error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during attempt {attempt + 1}: {e}")
            exit(1)

# Main function to encapsulate the sequence of operations
def main():
    print("Starting main process...")

    client = initialize_client()

    model_name = generate_model_name(BASE_MODEL_NAME)
    print(f"Generated model name: {model_name}")

    model = create_model(client, REPLICATE_OWNER, model_name, VISIBILITY, HARDWARE, DESCRIPTION)

    create_hf_repo(HUGGING_FACE_TOKEN, HF_REPO_ID)

    if USE_CAPTIONS:
        print("Captions are enabled. Configure Llava3 as USE_CAPTIONS is set to True.")
        exit(0)

    print("Preparing images and zipping them...")
    prepare_images(INPUT_DIR, OUTPUT_DIR, TRIGGER_WORD)
    zip_images(OUTPUT_DIR, ZIP_FILE_NAME)

    print(f"Starting training with model: {model.name} at {datetime.now()}")

    # Generate the direct download link using gdrive_large_file_utils
    direct_download_link = generate_direct_download_link(GOOGLE_DRIVE_PATH_TO_IMAGES_ZIP)
    print(f"Direct download link generated: {direct_download_link}")

    start_training(client, model, direct_download_link, STEPS, LORA_RANK, OPTIMIZER, BATCH_SIZE, RESOLUTION,
                   AUTOCAPTION, TRIGGER_WORD, LEARNING_RATE, HUGGING_FACE_TOKEN, HF_REPO_ID, VERSION, RETRY_DELAY, MAX_RETRIES)

    print(f"Training process completed at {datetime.now()}")

if __name__ == "__main__":
    main()