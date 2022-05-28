from aitextgen import aitextgen
from aitextgen.utils import GPT2ConfigCPU
from aitextgen.tokenizers import train_tokenizer
from aitextgen.TokenDataset import TokenDataset

import streamlit as st
import pandas as pd
from io import StringIO

# Needed otherwise program freezes
from multiprocessing import freeze_support


def main():

    #! streamlit
    # Base
    st.title("aiTextGen GUI version 0.0.1")
    uploaded_file = st.file_uploader("Upload a textfile to train AI")
    col1, col2, = st.columns(2)
    button_generate = col1.button("Generate Dataset")
    button_save = col2.button("Save File")
    
    # Select ai model
    aimodel = st.selectbox("Select AI Model",["deepset/gbert-large","gpt2","oliverguhr/german-sentiment-bert","dbmdz/bert-base-german-cased","dbmdz/bert-base-german-uncased"])
    
    # Parameter
    batch = st.sidebar.number_input("Batch Size",0,value=1)
    num_steps = st.sidebar.number_input("Number of steps",0,value=100)   
    gen_dataset = st.sidebar.number_input("Generate Dataset every",0,value=10)
    save_dataset = st.sidebar.number_input("Save Dataset every",0,value=10)
    block_size = st.sidebar.number_input("Block Size",0,value=32)
    num_lines = st.sidebar.number_input("Create N lines",0,value=10)
    prompt = st.sidebar.text_input("Prompt")
    
    #! ai generation part
    def aiTrainingProcess():
        # Output filename
        global output_file 
        output_file = "affe5.txt"

        # Training file
        file_name = "beleidigungen.txt"
        train_tokenizer(file_name)
        
        tokenizer_file = "aitextgen.tokenizer.json"
     
        # GPU config
        # config = aitextgen.to_gpu(self=0)
        # GPT2Model.cuda(self=GPT2Model,device=0)
        
        # Select training via CPU
        config = GPT2ConfigCPU()
        # Set ai model and token file
        ai = aitextgen(model=aimodel, tokenizer_file=tokenizer_file, config=config)

        # DataSet
        data = TokenDataset(file_name, tokenizer_file=tokenizer_file, block_size=block_size)
        ai.train(data, batch_size=batch, num_steps=num_steps, generate_every=gen_dataset, save_every=save_dataset, enable_model_summary=False)
        
        # Generate file with N lines of generated text
        ai.generate(num_lines, prompt=prompt)
        ai.generate_to_file(num_lines, destination_path=output_file, cleanup_resources=True)
           
    
    #! file handling
    if uploaded_file is not None:

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        file_string_data = stringio.read()
        
        # Add input text
        header = "INPUT TEXT"
        st.header(header)
        st.write(file_string_data)

        #! Generate button pressed TRUE
        if button_generate:
            # Start ai training
            aiTrainingProcess()

            # Add output text
            st.header("OUTPUT TEXT")
            stringioOutput = StringIO(output_file)

            # Load and output file
            file_string_data = open(output_file,"r")
            st.write(file_string_data.read())
            pass
        #! Generate button pressed TRUE
        if button_save:
            # Generate json an aven 
            pass
        else:
            st.write("gdf")
            pass
        
    else:
        warning = st.warning("Still no file upload")
        pass


if __name__ == '__main__':
    main()
    freeze_support()