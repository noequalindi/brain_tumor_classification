import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

model_dir = 'models'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]

class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict(image, model_name):
    # Load the selected model
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return class_names[predicted_class]

def clear_inputs():
    return None, None, ""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="indigo",
).set(
    background_fill_primary='#121212',  # Dark background
    background_fill_secondary='#1e1e1e',
    block_background_fill='#1e1e1e',  # Almost black
    block_border_color='#333',
    block_label_text_color='#fffff',
    block_label_text_color_dark = '#fffff',
    block_title_text_color_dark = '#fffff',
    button_primary_background_fill='#4f46e5',  # Violet
    button_primary_background_fill_hover='#2563eb',  # Light blue
    button_secondary_background_fill='#4f46e5',
    button_secondary_background_fill_hover='#2563eb',
    input_background_fill='#333',  # Dark grey
    input_border_color='#444',  # Intermediate grey
    block_label_background_fill='#4f46e5',
    block_label_background_fill_dark='#4f46e5',
    slider_color='#2563eb',
    slider_color_dark='#2563eb',
    button_primary_text_color='#fffff',
    button_secondary_text_color='#fffff',
    button_secondary_background_fill_hover_dark='#4f46e5',
    button_cancel_background_fill_hover='#444',
    button_cancel_background_fill_hover_dark='#444'
)

with gr.Blocks(theme=theme, css="""
    body, gradio-app {
      background-image: url('https://b2928487.smushcdn.com/2928487/wp-content/uploads/2022/04/Brain-inspiredAI-2048x1365.jpeg?lossy=1&strip=1&webp=1'); 
      background-size: cover;
      color: white;
    }

    .gradio-container {
      background-color: transparent;
      background-image: url('https://b2928487.smushcdn.com/2928487/wp-content/uploads/2022/04/Brain-inspiredAI-2048x1365.jpeg?lossy=1&strip=1&webp=1') !important; 
      background-size: cover !important;
      color: white;
    }

    .gradio-container .gr-dropdown-container select::after {
        content: '▼'; 
        color: white; 
        padding-left: 5px; 
    }

    .gradio-container .gr-dropdown-container select:focus {
        outline: none; 
        border-color: #4f46e5;
    }
    .gradio-container select {
      color: white;
    }  
    input, select, span, button, svg, .secondary-wrap {
      color: white; 
    }
               
    h1 {
        color: white;
        font-size: 4em;  
        margin: 20px auto;
    }
    .gradio-container h1 { 
        font-size: 5em;  
        color: white; 
        text-align: center; 
        text-shadow: 2px 2px 0px #8A2BE2,  
                     4px 4px 0px #00000033; 
        text-transform: uppercase;
        margin: 18px auto;
    }
    .gradio-container input { 
        color: white; 
    }
    .gradio-container .output { 
        color: white; 
    }
    .required-dropdown li {
      color: white;
    }
    .button-style {
      background-color: #4f46e5;
      color: white;
    }
    .button-style:hover {
      background-color: #2563eb;
      color: white;
    }
               
    .gradio-container .contain textarea {
      color: white;
      font-weight: 600;
      font-size: 1.5rem;
    }
    .contain textarea {
      color: white;
      font-weight: 600;
      font-size: 1.5rem;
    }     
    textarea {
      color: white;
      font-weight: 600;
      font-size: 1.5rem;
      background-color: lavender;
    }     
    textarea .scroll-hide {
      color: white;
    }
    .scroll-hide svelte-1f354aw {
      color: white;
    }
    """) as demo:

    gr.Markdown("# Brain Tumor Classification 🧠")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Sube la imagen")
            model_input = gr.Dropdown(choices=model_files, label="Selecciona un modelo", elem_classes=['required-dropdown'])
            classify_btn = gr.Button("Clasificar", elem_classes=['button-style'])
            clear_btn = gr.Button("Limpiar")
        with gr.Column():
            prediction_output = gr.Textbox(label="Predicción")
    
    classify_btn.click(predict, inputs=[image_input, model_input], outputs=prediction_output)
    clear_btn.click(clear_inputs, inputs=[], outputs=[image_input, model_input, prediction_output])

demo.launch()
