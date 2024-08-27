import os
import replicate
import requests
import random
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables from .env file
load_dotenv()

# GLOBAL VARIABLES for easy tweaks

# API token for authenticating with Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Version of the model to use on Replicate
MODEL_VERSION = "tillo13/flux1-dev-lora-20240826183521:4da855938dfa7424753aaaad5ba1f0e905453413ccbb1e5a8c029046558122bb"

# Specifies which model to use for inference ("dev" or "schnell"). "dev" needs around 28 steps, "schnell" needs around 4 steps.
DEFAULT_MODEL = "dev"  # Default is "dev"

# Beard descriptor to always include in the prompt
ALWAYS_PREPENDED_PROMPT = "andytillo with his epic and nicely manicured volumous beard"

# Custom scene description to append to the always prepended prompt
CUSTOM_PROMPT = "playing beach volleyball, displaying a sign that says 'Millie rules'. Medium shot. High-energy beach sports scene. Sand flying through the air as he jumps. Determination and joy on his face. The beach crowd is buzzing with excitement and cheers. Warm, sunny lighting beaming down. Best quality, vivid colors, 8k resolution, sleek textures, deep blues of the ocean, the golden sands stretching around. Massive beach atmosphere, sweeping view of the beach volleyball court, intense dynamic ambiance, the sound of waves crashing and crowd cheering filling the air."

# Aspect ratio for the generated image.
# Options:
# "1:1" - Square (default, 1024x1024 if aspect ratio is 1:1)
# "16:9" - Standard widescreen (common for modern videos and movies)
# "21:9" - Ultra widescreen (often used for cinematic movies)
# "3:2" - Traditional photo aspect ratio (used in 35mm film)
# "2:3" - Often used for portrait photos
# "4:5" - Common for social media posts
# "5:4" - Professional photo prints
# "3:4" - Older TV shows and computer monitors
# "4:3" - Standard definition TV (SDTV)
# "9:16" - Vertical video format (common for mobile devices)
# "9:21" - Ultra tall vertical format
# "custom" - Use custom width and height
DEFAULT_ASPECT_RATIO = "21:9"  # For a movie screen-like image

# Width of the generated image. Optional, only used when aspect_ratio="custom". Must be a multiple of 16.
DEFAULT_WIDTH = 1024  # Example: 1024 (not used if aspect_ratio is fixed)

# Height of the generated image. Optional, only used when aspect_ratio="custom". Must be a multiple of 16.
DEFAULT_HEIGHT = 1024  # Example: 1024 (not used if aspect_ratio is fixed)

# Number of images to output. Range: 1 to 4.
DEFAULT_NUM_OUTPUTS = 1  # Default is 1

# Determines how strongly the main LoRA should be applied. Sane results between 0 and 1.
DEFAULT_LORA_SCALE = 1  # Default is 0.8

# Number of inference steps to process. Range: 1 to 50.
DEFAULT_NUM_INFERENCE_STEPS = 50  # Default is 28

# Guidance scale for the diffusion process. Range: 0 to 10.
DEFAULT_GUIDANCE_SCALE = 3.5  # Default is 3.5

# Random seed for reproducible generation. If set to 0, a random seed will be generated.
DEFAULT_SEED = 0

# Combine this fine-tune with another LoRA. Supports various formats for specifying the additional model.
DEFAULT_EXTRA_LORA = 'https://civitai.com/api/download/models/753339?type=Model&format=SafeTensor'  # Proper format for CivitAI model URL

# Determines how strongly the extra LoRA should be applied. Range: 0 to 1.
DEFAULT_EXTRA_LORA_SCALE = 0.8  # Default is 0.8

# Format of the output images. Options: "webp", "jpg", "png"
DEFAULT_OUTPUT_FORMAT = "png"  # "png" for best quality, lossless format

# Quality when saving the output images (0 to 100). Only relevant for "jpg", not relevant for "png".
DEFAULT_OUTPUT_QUALITY = 100  # Best quality, max value is 100

# Disable safety checker for generated images. Set to True to disable, False to keep enabled.
DEFAULT_DISABLE_SAFETY_CHECKER = False  # Default is False, for safety

# Ensure tokens are correctly loaded
if not REPLICATE_API_TOKEN:
    print("Error: REPLICATE_API_TOKEN is not set in the .env file")
    exit(1)

# Initialize the Replicate client with your API token
def initialize_client():
    print("Initializing Replicate client with API token...")
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    print("Replicate client initialized successfully.")
    return client

# Run the model using Replicate's API
def run_model(client, model_version, model=DEFAULT_MODEL, prompt=None, aspect_ratio=DEFAULT_ASPECT_RATIO, width=DEFAULT_WIDTH,
              height=DEFAULT_HEIGHT, num_outputs=DEFAULT_NUM_OUTPUTS, lora_scale=DEFAULT_LORA_SCALE, num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
              guidance_scale=DEFAULT_GUIDANCE_SCALE, seed=DEFAULT_SEED, output_format=DEFAULT_OUTPUT_FORMAT, output_quality=DEFAULT_OUTPUT_QUALITY,
              extra_lora_scale=DEFAULT_EXTRA_LORA_SCALE, extra_lora=DEFAULT_EXTRA_LORA, disable_safety_checker=DEFAULT_DISABLE_SAFETY_CHECKER):
    print("Running the model...")

    if seed == 0:
        seed = random.randint(1, 1_000_000)
        print(f"Random seed generated: {seed}")
    
    if prompt is None:
        prompt = CUSTOM_PROMPT

    full_prompt = f"{ALWAYS_PREPENDED_PROMPT} {prompt}"
    
    # Prepare input parameters
    input_params = {
        "model": model,
        "prompt": full_prompt,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "num_outputs": num_outputs,
        "lora_scale": lora_scale,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_format": output_format,
        "output_quality": output_quality,
        "extra_lora_scale": extra_lora_scale,
        "extra_lora": extra_lora,
        "disable_safety_checker": disable_safety_checker
    }
    
    # Run the model
    output = client.run(model_version, input=input_params)
    
    print("Model run successfully.")
    return output

# Save the generated images
def save_images(urls, model_version, output_format):
    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")
        print("Directory 'generated_images' created.")
    
    for url in urls:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"generated_images/{timestamp}_{model_version.split(':')[1]}.{output_format}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Image saved as {file_name}")
        else:
            print(f"Failed to download image from {url}")

# Main function to encapsulate the sequence of operations
def main():
    start_time = time.time()
    print("Starting main process...")

    client = initialize_client()
    
    # Run the model with the combined prompt
    output = run_model(client, MODEL_VERSION)
    
    print("Output URLs:")
    for url in output:
        print(url)
    
    save_images(output, MODEL_VERSION, DEFAULT_OUTPUT_FORMAT)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()