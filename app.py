import gradio as gr
import os
# import spaces
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)


DESCRIPTION = '''
<div>
<h1 style="text-align: center;">LLaMA-Mesh</h1>
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/nv-tlabs/LLaMA-Mesh"><img src='https://img.shields.io/github/stars/nv-tlabs/LLaMA-Mesh?style=social'/></a>
</div>
<p>LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models. <a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/">[Project Page]</a> <a style="display:inline-block" href="https://github.com/nv-tlabs/LLaMA-Mesh">[Code]</a></p>
<p> Notice: (1) The default token length is 4096. If you observe incomplete generated meshes, try to increase the maximum token length into 8192.</p>
<p>(2) We only support generating a single mesh per dialog round. To generate another mesh, click the "clear" button and start a new dialog.</p>
<p>(3) If the LLM refuses to generate a 3D mesh, try adding more explicit instructions to the prompt, such as "create a 3D model of a table <strong>in OBJ format</strong>." A more effective approach is to request the mesh generation at the start of the dialog.</p>
</div>
'''

LICENSE = """
<p/>

---
Built with Meta Llama 3.1 8B
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">LLaMA-Mesh</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Create 3D meshes by chatting.</p>
</div>
"""


css = """
h1 {
  text-align: center;
  display: block;
}

#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""
# Load the tokenizer and model
model_path = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


from trimesh.exchange.gltf import export_glb
import gradio as gr
import trimesh
import numpy as np
import tempfile
def apply_gradient_color(mesh_text):
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
    Args:
        mesh_text (str): The input mesh in OBJ format as a string.
    Returns:
        str: Path to the GLB file with gradient colors applied.
    """
    # Load the mesh
    temp_file =  tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
    with open(temp_file+".obj", "w") as f:
        f.write(mesh_text)
    # return temp_file
    mesh = trimesh.load_mesh(temp_file+".obj", file_type='obj')

    # Get vertex coordinates
    vertices = mesh.vertices
    y_values = vertices[:, 1]  # Y-axis values

    # Normalize Y values to range [0, 1] for color mapping
    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    # Generate colors: Map normalized Y values to RGB gradient (e.g., blue to red)
    colors = np.zeros((len(vertices), 4))  # RGBA
    colors[:, 0] = y_normalized  # Red channel
    colors[:, 2] = 1 - y_normalized  # Blue channel
    colors[:, 3] = 1.0  # Alpha channel (fully opaque)

    # Attach colors to mesh vertices
    mesh.visual.vertex_colors = colors

    # Export to GLB format
    glb_path = temp_file+".glb"
    with open(glb_path, "wb") as f:
        f.write(export_glb(mesh))
    
    return glb_path

def visualize_mesh(mesh_text):
    """
    Convert the provided 3D mesh text into a visualizable format.
    This function assumes the input is in OBJ format.
    """
    temp_file = "temp_mesh.obj"
    with open(temp_file, "w") as f:
        f.write(mesh_text)
    return temp_file

# @spaces.GPU(duration=120)
def chat_llama3_8b(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int
             ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        #print(outputs)
        yield "".join(outputs)
        

# Gradio block
chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True, css=css) as demo:
    with gr.Column(): 
        gr.Markdown(DESCRIPTION)
        # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
        with gr.Row():
            with gr.Column(scale=3):    
                gr.ChatInterface(
                    fn=chat_llama3_8b,
                    chatbot=chatbot,
                    fill_height=True,
                    additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
                    additional_inputs=[
                        gr.Slider(minimum=0,
                                maximum=1, 
                                step=0.1,
                                value=0.95, 
                                label="Temperature", 
                                render=False),
                        gr.Slider(minimum=128, 
                                maximum=8192,
                                step=1,
                                value=4096, 
                                label="Max new tokens", 
                                render=False),
                        ],
                    examples=[
                        ['Create a 3D model of a wooden hammer'],
                        ['Create a 3D model of a pyramid in obj format'],
                        ['Create a 3D model of a cabinet.'],
                        ['Create a low poly 3D model of a coffe cup'],
                        ['Create a 3D model of a table.'],
                        ["Create a low poly 3D model of a tree."],
                        ['Write a python code for sorting.'],
                        ['How to setup a human base on Mars? Give short answer.'],
                        ['Explain theory of relativity to me like I’m 8 years old.'],
                        ['What is 9,000 * 9,000?'],
                        ['Create a 3D model of a soda can.'],
                        ['Create a 3D model of a sword.'],
                        ['Create a 3D model of a wooden barrel'],
                        ['Create a 3D model of a chair.']
                        ],
                    cache_examples=False,
                                )
                gr.Markdown(LICENSE)
        
            with gr.Column(scale=2): 
                output_model = gr.Model3D(
                            label="3D Mesh Visualization",
                            interactive=False,
                        )
                gr.Markdown("You can copy the generated 3d objects in the left and paste in the textbox below. Put the button and you will see the visualization of the 3D mesh.")
                
                # Add the text box for 3D mesh input and button
                mesh_input = gr.Textbox(
                    label="3D Mesh Input",
                    placeholder="Paste your 3D mesh in OBJ format here...",
                    lines=5,
                )
                visualize_button = gr.Button("Visualize 3D Mesh")
                
                # Link the button to the visualization function
                visualize_button.click(
                    fn=apply_gradient_color,
                    inputs=[mesh_input],
                    outputs=[output_model]
                )
          
if __name__ == "__main__":
    demo.launch()
    
