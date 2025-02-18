import re
import torch
import streamlit as st
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the dataset
st.title("📄 Donut OCR Viewer")

@st.cache_resource()
def load_model():
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

@st.cache_data()
def load_cord_dataset():
    return load_dataset("naver-clova-ix/cord-v2")

def decode_prediction(image, processor, model, device, task_prompt="<s_cord-v2>"):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return processor.token2json(sequence)

# Load models and dataset
processor, model, device = load_model()
dataset = load_cord_dataset()

# Select an image index
index = st.slider("Select an image index", 0, len(dataset['validation']) - 1, 0)
example = dataset['validation'][index]
image = example['image']

# Display the image
st.image(image, caption=f"Image ID: {index}", use_column_width=True)

# Run prediction
with st.spinner("Extracting text..."):
    predicted_json = decode_prediction(image, processor, model, device)

# Show extracted content
st.subheader("Extracted JSON Output:")
st.json(predicted_json)
