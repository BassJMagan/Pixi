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
from bs4 import BeautifulSoup
import pyperclip

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://gelbooru.com",
}

PIXIV_MODES = [
    'day', 'week', 'month', 'day_male', 'day_female',
    'week_original', 'week_rookie', 'day_r18', 
    'day_male_18', 'day_female_r18', 'week_r18'
]

KEYBOARD_JS = """
<script>
document.addEventListener("keydown", function(event) {
    if (event.key === "Enter") { 
        event.preventDefault();
        document.getElementById('gelbooru_btn').click();
    }
    if (event.key.toLowerCase() === "p") { 
        event.preventDefault();
        document.getElementById('fetch_btn').click();
    }
    if (event.key === " ") { 
        event.preventDefault();
        document.getElementById('tags_btn').click();
    }
});
</script>
"""

class TaggerPredictor:
    def __init__(self):
        self.model = None
        self.tag_names = None
        self.model_target_size = None

    def load_model(self):
        if self.model:
            return
        csv_path = huggingface_hub.hf_hub_download(MODEL_REPO, LABEL_FILENAME)
        model_path = huggingface_hub.hf_hub_download(MODEL_REPO, MODEL_FILENAME)
        self.tag_names = pd.read_csv(csv_path)["name"].tolist()
        self.model = rt.InferenceSession(model_path)
        _, height, _, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

    def prepare_image(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = image.convert("RGB").resize((self.model_target_size, self.model_target_size), Image.BICUBIC)
        image_array = np.asarray(image, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(image_array, axis=0)

    def predict_tags(self, image):
        self.load_model()
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        image_input = self.prepare_image(image)
        predictions = self.model.run([output_name], {input_name: image_input})[0][0]
        tags_with_scores = list(zip(self.tag_names, predictions))
        sorted_tags = sorted(tags_with_scores, key=lambda x: x[1], reverse=True)
        tags = [tag.replace("_", " ") for tag, score in sorted_tags if score > 0.30]
        return ", ".join(tags)

def fetch_random_gelbooru_image():
    url = "https://gelbooru.com/index.php?page=post&s=random"
    while True:
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find("img", {"id": "image"})
            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']
                if not image_url.startswith("http"):
                    image_url = f"https:{image_url}"
                image_response = requests.get(image_url, headers=HEADERS)
                return Image.open(io.BytesIO(image_response.content))
        except Exception:
            continue

def fetch_random_pixiv_ranked_image():
    while True:
        try:
            random_date = (datetime.now() - timedelta(days=random.randint(0, 6096))).strftime('%Y-%m-%d')
            mode = random.choice(PIXIV_MODES)
            rank_url = f"https://api.obfs.dev/api/pixiv/rank?mode={mode}&date={random_date}"
            rank_data = requests.get(rank_url).json()
            illust = random.choice(rank_data['illusts'])
            image_url = illust['meta_pages'][0]['image_urls']['original'] \
                if illust['meta_pages'] else illust['meta_single_page']['original_image_url']
            headers = {"Referer": "https://www.pixiv.net", "User-Agent": "Mozilla/5.0"}
            image_response = requests.get(image_url, headers=headers)
            return Image.open(io.BytesIO(image_response.content))
        except Exception:
            continue

predictor = TaggerPredictor()

with gr.Blocks(head=KEYBOARD_JS) as demo:
    with gr.Column():
        with gr.Row():
            fetch_btn = gr.Button("Fetch Random Ranked Image", elem_id="fetch_btn")
            gelbooru_btn = gr.Button("Fetch Random Gelbooru Image", elem_id="gelbooru_btn")
        image_output = gr.Image(elem_id="image_output")
        tags_btn = gr.Button("Get Tags", elem_id="tags_btn")
        tags_output = gr.Textbox(label="Predicted Tags")

        fetch_btn.click(fetch_random_pixiv_ranked_image, outputs=image_output)
        gelbooru_btn.click(fetch_random_gelbooru_image, outputs=image_output)

        def display_tags(image):
            if image is None:
                return "No image loaded!"
            tags = predictor.predict_tags(image)
            pyperclip.copy(tags)
            return tags

        tags_btn.click(display_tags, inputs=image_output, outputs=tags_output)

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)