import gradio as gr
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

sys.stdout = Logger("output.log")

def test(x):
    print("This is a test")
    print(f"Your function is running with input {x}...")
    return x

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

with gr.Blocks() as demo:
    with gr.Row():
        input = gr.Textbox()
        output = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(test, input, output)
    
    logs = gr.Textbox(label='log')
    logs.attach_load_event(read_logs, every=3)
    # demo.load(read_logs, None, logs)
    
demo.launch(server_name='0.0.0.0', server_port=9800)