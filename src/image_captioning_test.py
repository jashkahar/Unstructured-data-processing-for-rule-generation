''' This is a fucntionality test script, this is not part of the actual system.'''
import base64
from openai import OpenAI  # Import the OpenAI client library
import os
from PIL import Image

# Initialize the client (ensure your environment variable is set)
client = OpenAI(api_key='')

def resize_image(image_path, max_width=800):
    """
    Open the image, resize it to a maximum width (while preserving aspect ratio),
    and return the resized image.
    """
    with Image.open(image_path) as img:
        # Calculate the new height to preserve aspect ratio
        width_percent = max_width / float(img.size[0])
        new_height = int(float(img.size[1]) * width_percent)
        # Resize the image
        img = img.resize((max_width, new_height), Image.LANCZOS)
        return img

def encode_image_from_pil(pil_image, image_format="JPEG"):
    """
    Convert a PIL Image object into a Base64-encoded string.
    """
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Define the path to your image file
image_path = "C:/Users/jkahar/Downloads/Test.png"

# Open and resize the image to reduce resolution
resized_image = resize_image(image_path, max_width=800)

# Encode the resized image to Base64 format
base64_image = encode_image_from_pil(resized_image)

# Define a structured prompt that instructs the model what information to extract
structured_prompt = (
    "Please analyze the attached image and provide a structured JSON output with the following keys:\n"
    "description: A detailed description of this pharmaceutical marketing material,\n"
    "layout: The layout and design of the pharmaceutical advertisement,\n"
    "visual_elements: Colors, imagery, and visual elements in the pharmaceutical material."
)

# Create the response by calling the OpenAI client's method with both text and image input,
# and include the response_format to enforce the structured output.
response = client.responses.create(
    model="gpt-4o-mini-2024-07-18",  # Specify the appropriate model that supports structured outputs
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": structured_prompt },
                { "type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}" }
            ]
        }
    ]
)

# Print the structured output from the response (expected to be a valid JSON matching our schema)
print(response.output_text)
