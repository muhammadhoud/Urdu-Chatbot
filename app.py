import streamlit as st
import torch
import json
import re
import unicodedata
import math
import torch.nn as nn
from model_architecture import Transformer, ModelConfig

st.set_page_config(
    page_title="اردو چیٹ بوٹ",
    page_icon="🤖",
    layout="wide"
)

# RTL CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap');
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 20px;
        line-height: 2;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 75%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model once and cache"""
    with open('model/urdu_vocabulary.json', 'r', encoding='utf-8') as f:
        token_to_id = json.load(f)
        id_to_token = {int(idx): token for token, idx in token_to_id.items()}
    
    checkpoint = torch.load('model/final_transformer_model.pth',
                           map_location='cpu', weights_only=False)
    
    config_dict = checkpoint['config']
    config_dict['start_token_id'] = token_to_id['[START]']
    config_dict['end_token_id'] = token_to_id['[END]']
    config = ModelConfig(config_dict)
    
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, token_to_id, id_to_token, config

def normalize_urdu_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[آأإ]', 'ا', text)
    text = re.sub(r'[ى]', 'ی', text)
    return re.sub(r'\s+', ' ', text).strip()

def text_to_tokens(text, token_to_id, max_len):
    tokens = [token_to_id['[START]']]
    for char in text[:max_len-2]:
        tokens.append(token_to_id.get(char, token_to_id['[UNK]']))
    tokens.append(token_to_id['[END]'])
    
    if len(tokens) < max_len:
        tokens.extend([token_to_id['[PAD]']] * (max_len - len(tokens)))
    else:
        tokens = tokens[:max_len]
        tokens[-1] = token_to_id['[END]']
    return tokens

def tokens_to_text(tokens, id_to_token, end_id):
    text = []
    for token_id in tokens:
        if token_id == end_id:
            break
        token = id_to_token.get(token_id, '')
        if token not in ['[PAD]', '[START]', '[END]', '[UNK]']:
            text.append(token)
    return ''.join(text)

def generate_response(user_input, model, token_to_id, id_to_token, config):
    normalized = normalize_urdu_text(user_input)
    input_tokens = text_to_tokens(normalized, token_to_id, config.max_len)
    input_tensor = torch.tensor([input_tokens], dtype=torch.long)
    
    with torch.no_grad():
        output_tokens = model.greedy_decode(input_tensor, max_len=config.max_len)
        response = tokens_to_text(output_tokens[0].tolist(), id_to_token, 
                                 token_to_id['[END]'])
    
    return response if response.strip() else "میں ابھی سیکھ رہا ہوں۔"

def main():
    st.markdown('<h1 style="text-align: center; color: #4CAF50;">🤖 اردو چیٹ بوٹ</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model, token_to_id, id_to_token, config = load_model()
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "السلام علیکم! میں آپ کا اردو چیٹ بوٹ ہوں۔ 👋"}
        ]
    
    # Display messages
    for msg in st.session_state.messages:
        css_class = "user-message" if msg["role"] == "user" else "bot-message"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{css_class} urdu-text">{icon} {msg["content"]}</div>',
                   unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("اپنا پیغام لکھیں...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🤖 جواب تیار ہو رہا ہے..."):
            response = generate_response(user_input, model, token_to_id, id_to_token, config)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ معلومات")
        st.info(f"📊 Vocabulary: {len(token_to_id)} tokens")
        st.info(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if st.button("🗑️ گفتگو صاف کریں", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "السلام علیکم! 👋"}
            ]
            st.rerun()

if __name__ == "__main__":
    main()
