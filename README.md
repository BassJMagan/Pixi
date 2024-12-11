# Pixi
Python script to get random images from Pixiv and then tag them with Smilingwolf's wd-vit-large-tagger-v3 if you like the result &amp; want to use similar tags in an anime-based diffusion model.

To use simply paste this in cmd/terminal/powershell with Python installed (I'm using version 3.12):


```pip install gradio huggingface_hub onnxruntime pandas numpy pillow requests```


Then you're good to go, run the script and it'll open a Gradio window in your browser with two simple buttons.

![{5B949FF5-E71D-46AA-84AE-5636E22351EF}](https://github.com/user-attachments/assets/04579527-7ba6-4656-9475-39533f217891)


Hotkeys:
Enter = Fetch random Gelbooru image
p/P = Fetch random ranked Pixiv image
Space = Get Tags
