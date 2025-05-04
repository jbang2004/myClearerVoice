import streamlit as st
from clearvoice import ClearVoice
import os
import tempfile
import librosa
import numpy as np
import soundfile as sf

st.set_page_config(page_title="ClearerVoice Studio", layout="wide")
temp_dir = 'temp'
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Check if temp directory exists, create if not
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Save to temp directory, overwrite if file exists
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return temp_path
    return None

def main():
    st.title("ClearerVoice Studio")
    
    tabs = st.tabs(["Speech Enhancement", "Speech Separation", "Target Speaker Extraction"])
    
    with tabs[0]:
        st.header("Speech Enhancement")
        
        # Model selection
        se_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        selected_model = st.selectbox("Select Model", se_models)
        
        # File upload
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav'], key='se')
        
        if st.button("Start Processing", key='se_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # 保存并转换为单声道输入
                    input_path = save_uploaded_file(uploaded_file)
                    mono_name = f"mono_{os.path.basename(input_path)}"
                    mono_path = os.path.join(temp_dir, mono_name)
                    # 用 librosa 读入并强制为 mono，再写回 wav
                    audio_arr, sr0 = librosa.load(input_path, sr=None, mono=True)
                    sf.write(mono_path, audio_arr, sr0)
                    input_path = mono_path

                    # 输出目录，按模型子目录组织
                    output_dir = os.path.join(temp_dir, "speech_enhancement_output")
                    os.makedirs(output_dir, exist_ok=True)

                    # 在线写入：底层只输出增强音，噪声文件可能不存在
                    myClearVoice = ClearVoice(task='speech_enhancement', model_names=[selected_model])
                    myClearVoice(input_path=input_path, online_write=True, output_path=output_dir)
                    # 构建输出路径
                    model_dir = os.path.join(output_dir, selected_model)
                    wav_name = os.path.basename(input_path)
                    enh_path = os.path.join(model_dir, wav_name)
                    noc_path = enh_path.replace('.wav', '_noise.wav')
                    # 若底层未生成噪声，则手动后处理
                    if not os.path.exists(noc_path):
                        # 读取原始和增强音，再做减法
                        orig, _ = librosa.load(input_path, sr=sr0, mono=True)
                        enh, _ = librosa.load(enh_path, sr=sr0, mono=True)
                        min_len = min(len(orig), len(enh))
                        noise = orig[:min_len] - enh[:min_len]
                        sf.write(noc_path, noise, sr0)
                    # 展示结果
                    st.subheader("增强后音频")
                    st.audio(enh_path)
                    st.subheader("被去除的背景噪声")
                    st.audio(noc_path)
            else:
                st.error("Please upload an audio file first")
    
    with tabs[1]:
        st.header("Speech Separation")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Mixed Audio File", type=['wav', 'avi'], key='ss')
        
        if st.button("Start Separation", key='ss_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)

                    # Extract audio if input is video file
                    if input_path.endswith(('.avi')):
                        import cv2
                        video = cv2.VideoCapture(input_path)
                        audio_path = input_path.replace('.avi','.wav')
                        
                        # Extract audio
                        import subprocess
                        cmd = f"ffmpeg -i {input_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                        subprocess.call(cmd, shell=True)
                        
                        input_path = audio_path
                    
                    # Initialize ClearVoice
                    myClearVoice = ClearVoice(task='speech_separation', 
                                            model_names=['MossFormer2_SS_16K'])
                    
                    # Process audio
                    output_wav = myClearVoice(input_path=input_path, 
                                            online_write=False)
                    
                    output_dir = os.path.join(temp_dir, "speech_separation_output")
                    os.makedirs(output_dir, exist_ok=True)

                    file_name = os.path.basename(input_path).split('.')[0]
                    base_file_name = 'output_MossFormer2_SS_16K_'
                    
                    # Save processed audio
                    output_path = os.path.join(output_dir, f"{base_file_name}{file_name}.wav")
                    myClearVoice.write(output_wav, output_path=output_path)
                    
                    # Display output directory
                    st.text(output_dir)

            else:
                st.error("Please upload an audio file first")
    
    with tabs[2]:
        st.header("Target Speaker Extraction")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi'], key='tse')
        
        if st.button("Start Extraction", key='tse_process'):
            if uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Save uploaded file
                    input_path = save_uploaded_file(uploaded_file)
                    
                    # Create output directory
                    output_dir = os.path.join(temp_dir, "videos_tse_output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Initialize ClearVoice
                    myClearVoice = ClearVoice(task='target_speaker_extraction', 
                                            model_names=['AV_MossFormer2_TSE_16K'])
                    
                    # Process video
                    myClearVoice(input_path=input_path, 
                                 online_write=True,
                                 output_path=output_dir)
                    # Display output folder
                    st.subheader("Output Folder")
                    st.text(output_dir)
                
            else:
                st.error("Please upload a video file first")

if __name__ == "__main__":    
    main()