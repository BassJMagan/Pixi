import gradio as gr
import requests
import random
from datetime import datetime, timedelta
from PIL import Image
import io
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import webbrowser

# ONNX Model Details
MODEL_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# Predictor Class for ONNX Inference
class TaggerPredictor:
    def __init__(self):
        self.model = None
        self.model_target_size = None
        self.tag_names = None

    def load_model(self):
        if self.model is not None:
            return  # Already loaded

        print("Loading ONNX model...")
        csv_path = huggingface_hub.hf_hub_download(MODEL_REPO, LABEL_FILENAME)
        model_path = huggingface_hub.hf_hub_download(MODEL_REPO, MODEL_FILENAME)

        # Load tags
        tags_df = pd.read_csv(csv_path)
        self.tag_names = tags_df["name"].tolist()

        # Load ONNX model
        self.model = rt.InferenceSession(model_path)
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

    def prepare_image(self, image):
        # Ensure input is a PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        # Resize and preprocess image
        image = image.convert("RGB")
        image = image.resize((self.model_target_size, self.model_target_size), Image.BICUBIC)
        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # RGB -> BGR
        return np.expand_dims(image_array, axis=0)

    def predict_tags(self, image):
        self.load_model()

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        image_input = self.prepare_image(image)

        predictions = self.model.run([output_name], {input_name: image_input})[0][0]
        tags_with_scores = list(zip(self.tag_names, predictions))
        sorted_tags = sorted(tags_with_scores, key=lambda x: x[1], reverse=True)

        # Return tags above a confidence threshold, formatted without scores
        tags = [tag.replace("_", " ") for tag, score in sorted_tags if score > 0.30]
        return ", ".join(tags)


# Function to Fetch Random Pixiv Ranked Image
def get_random_pixiv_ranked_image():
    modes = ['day', 'week', 'month', 'day_male', 'day_female', 'week_original', 'day_r18', 'day_male_18', 'day_female_r18', 'week_r18']
    mode = random.choice(modes)
    random_date = datetime.now() - timedelta(days=random.randint(0, 730))
    formatted_date = random_date.strftime('%Y-%m-%d')

    try:
        rank_url = f"https://api.obfs.dev/api/pixiv/rank?mode={mode}&date={formatted_date}"
        rank_response = requests.get(rank_url)
        rank_data = rank_response.json()

        # Choose a random image
        illust = random.choice(rank_data['illusts'])
        image_url = (illust['meta_pages'][0]['image_urls']['original']
                     if illust['meta_pages']
                     else illust['meta_single_page']['original_image_url'])

        headers = {
            'Referer': 'https://www.pixiv.net',
            'User-Agent': 'Mozilla/5.0'
        }
        image_response = requests.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content))
        return image

    except Exception as e:
        return None


# Instantiate Tagger Predictor
predictor = TaggerPredictor()

# Gradio UI
with gr.Blocks() as demo:
    with gr.Column():
        random_btn = gr.Button("Fetch Random Ranked Image")
        random_output = gr.Image()
        get_tags_btn = gr.Button("Get Tags")
        tags_output = gr.Textbox(label="Predicted Tags")

        # Fetch and display random image
        def fetch_image():
            return get_random_pixiv_ranked_image()

        random_btn.click(fetch_image, outputs=random_output)

        # Run model inference on the displayed image
        def get_tags(image):
            if image is None:
                return "No image loaded!"
            return predictor.predict_tags(image)

        get_tags_btn.click(get_tags, inputs=random_output, outputs=tags_output)

# Launch App and Open in Default Browser
if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)