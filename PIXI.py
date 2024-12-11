import gradio as gr
import requests
import random
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError
import io
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
from bs4 import BeautifulSoup

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

keyboard_js = """
<script>
document.addEventListener("keydown", function(event) {
    if (event.key === "Enter") { 
        event.preventDefault();
        document.getElementById('gelbooru_btn').click(); // Trigger "Fetch Random Gelbooru Image"
    }
    if (event.key === "p" || event.key === "P") { 
        event.preventDefault();
        document.getElementById('fetch_btn').click(); // Trigger "Fetch Random Ranked Image"
    }
        if (event.key === " ") { 
        event.preventDefault();
        document.getElementById('tags_btn').click(); // Trigger "Get Tags"
    }
});
</script>
"""

class TaggerPredictor:
    def __init__(self):
        self.model = None
        self.model_target_size = None
        self.tag_names = None

    def load_model(self):
        if self.model is not None:
            return

        print("Loading ONNX model...")
        csv_path = huggingface_hub.hf_hub_download(MODEL_REPO, LABEL_FILENAME)
        model_path = huggingface_hub.hf_hub_download(MODEL_REPO, MODEL_FILENAME)

        tags_df = pd.read_csv(csv_path)
        self.tag_names = tags_df["name"].tolist()

        self.model = rt.InferenceSession(model_path)
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

    def prepare_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')

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

        tags = [tag.replace("_", " ") for tag, score in sorted_tags if score > 0.30]
        return ", ".join(tags)


predictor = TaggerPredictor()

def get_random_gelbooru_image():
    gelbooru_random_url = "https://gelbooru.com/index.php?page=post&s=random"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://gelbooru.com"
    }

    while True:
        try:
            response = requests.get(gelbooru_random_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            img_tag = soup.find("img", {"id": "image"})

            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']

                if not image_url.startswith("http"):
                    image_url = f"https:{image_url}"

                image_response = requests.get(image_url, headers=headers)
                image_response.raise_for_status()

                image = Image.open(io.BytesIO(image_response.content))
                return image

            else:
                raise Exception("Failed to locate image URL on the page.")

        except Exception as e:
            print(f"Error fetching Gelbooru image: {e}")
            continue

def get_random_pixiv_ranked_image():
    while True:
        modes = ['day', 'week', 'month', 'day_male', 'day_female', 'week_original', 'week_rookie', 'day_r18', 'day_male_18', 'day_female_r18', 'week_r18']
        mode = random.choice(modes)
        random_date = datetime.now() - timedelta(days=random.randint(0, 6096))
        formatted_date = random_date.strftime('%Y-%m-%d')

        try:
            rank_url = f"https://api.obfs.dev/api/pixiv/rank?mode={mode}&date={formatted_date}"
            rank_response = requests.get(rank_url)
            rank_data = rank_response.json()

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
            print(f"Attempt failed: {e}")
            continue


with gr.Blocks(head=keyboard_js) as demo:
    with gr.Column():
        with gr.Row():
            random_btn = gr.Button("Fetch Random Ranked Image", elem_id="fetch_btn")
            gelbooru_btn = gr.Button("Fetch Random Gelbooru Image", elem_id="gelbooru_btn")
        random_output = gr.Image(elem_id="image_output")
        get_tags_btn = gr.Button("Get Tags", elem_id="tags_btn")
        tags_output = gr.Textbox(label="Predicted Tags")

        def fetch_image():
            return get_random_pixiv_ranked_image()

        random_btn.click(fetch_image, outputs=random_output)

        gelbooru_btn.click(get_random_gelbooru_image, outputs=random_output)

        def get_tags(image):
            if image is None:
                return "No image loaded!"
            return predictor.predict_tags(image)

        get_tags_btn.click(get_tags, inputs=random_output, outputs=tags_output)

if __name__ == "__main__":
    demo.launch(server_port=7860, inbrowser=True)