# Training and Querying LoRA Models with Flux and Phlux V1

This guide provides comprehensive steps to train a LoRA model using `train_flux_lora.py` and query it with `query_lora_model.py`. The process integrates with Replicate for model training, translation, and hosting, and Hugging Face for managing model versions and repositories. We will use the Flux models (dev and schnell) and extend them with Phlux V1 LoRA for enhanced photorealism.

## Prerequisites

1. **Python 3.x**: Ensure Python is installed.
2. **Dependencies**: Install required Python packages.
    ```sh
    pip install os replicate time shutil zipfile dotenv datetime httpx huggingface_hub requests urllib beautifulsoup4
    ```
3. **Environment Variables**: Create a `.env` file in the root directory and add your API tokens.
    ```env
    REPLICATE_API_TOKEN=your_replicate_api_token
    HUGGING_FACE_TOKEN=your_hugging_face_token
    ```

## Directory Structure

1. **Initial Images Directory**:
    Place your training images inside the `initial_images` directory.

## High-Level Overview

### Training and Deploying the LoRA Model

1. **Core Concepts**:
    - **Replicate**: A platform that offers scalable machine learning model training and hosting. It allows easy integration of models with various applications through API calls.
    - **Hugging Face**: A hub for managing and sharing machine learning models, particularly useful for version control and integration with the Transformers library.
    - **Flux Models (dev and schnell)**: These serve as the base models. "dev" requires around 28 steps, while "schnell" is faster, needing around 4 steps.
    - **Phlux V1**: An additional LoRA model for improved photorealism, especially focusing on texture and lighting.

2. **Model Training Workflow**:
    - **Initialization**: Load environment variables and tokens required for API access.
    - **Preparing Data**: Preprocess images in the `initial_images` directory and generate a ZIP file for training.
    - **Model Creation**: Use Replicate to create and store a new model, and Hugging Face to manage model versions.
    - **Training**: Start the training process on Replicate with the preprocessed image set and additional Phlux V1 LoRA for enhanced photorealism.
    - **Deployment**: Once training is complete, the model and its versions are available on Replicate and Hugging Face for querying.

### Using the Trained LoRA Model

1. **Image Generation**:
    - **Prompt Customization**: Customize prompts to generate specific scenes utilizing the newly trained LoRA model.
    - **Configuration Options**: Adjust settings like resolution, guidance scale, output format, and safety checks.
    - **API Integration**: Utilize Replicate’s API to run the model and fetch generated images.

## Running the Training Script

1. **Update Global Variables**:
    Modify the settings in `train_flux_lora.py` to match your desired configurations:
    ```python
    # Example configurations
    REPLICATE_OWNER = "your_replicate_username"
    HUGGING_FACE_OWNER = "your_hugging_face_username"
    BASE_MODEL_NAME = "desired_model_name"
    STEPS = 1000
    LORA_RANK = 16
    OPTIMIZER = "adamw8bit"
    ```
2. **Execute the Training Script**:
    ```sh
    python train_flux_lora.py
    ```

## Running the Query Script

1. **Update Global Variables**:
    Adjust the settings in `query_lora_model.py` to suit your needs:
    ```python
    # Example configurations
    MODEL_VERSION = "your_trained_model_version_on_replicate"
    DEFAULT_MODEL = "dev"
    DEFAULT_ASPECT_RATIO = "21:9"
    ```
2. **Execute the Query Script**:
    ```sh
    python query_lora_model.py
    ```

## Scripts Overview

### `train_flux_lora.py`

This script handles the training of a LoRA model using images from the `initial_images` directory. It integrates the Phlux V1 LoRA model for enhanced photorealism. The main steps include:

1. **Environment Setup**: Load API tokens and initialize the Replicate client.
2. **Image Preparation**: Rename and save images in the `prepared_images` directory, then zip them for training.
3. **Model Creation**: Create a new model on Replicate and a corresponding repository on Hugging Face.
4. **Start Training**: Initiate the training process on Replicate using the preprocessed images and Phlux V1 LoRA.

### `query_lora_model.py`

This script queries a trained LoRA model to generate images based on specified configurations. The main steps include:

1. **Environment Setup**: Load API tokens and initialize the Replicate client.
2. **Prompt Configuration**: Set up prompts and other configurations for image generation.
3. **Model Execution**: Run the model on Replicate and retrieve generated images.
4. **Image Saving**: Save the output images to the `generated_images` directory.

## Results and Access

Upon completing the training and querying processes, the trained model can be accessed through the URLs provided by Replicate and Hugging Face. The images generated will be located in the `generated_images` directory, showcasing the capabilities of the fine-tuned LoRA model with added photorealism from Phlux V1.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Questions or Issues?

If you encounter any issues or have questions, please feel free to open an issue or reach out to the repository maintainers.