"""Streamllit application to run Banalata poetry/prose generator."""

import logging
import random
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from indic_transliteration.sanscript import transliterate

# Setup paths relative to this file
APP_DIR = Path(__file__).parent
SRC_DIR = APP_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from src.banalata import s05_generate
from src.banalata.s01_prepare_data import TOK_BOW

# Sample text for random generation when the "Sample Bengali words" button is clicked
SAMPLE_BENGALI_TEXTS = [
    'হাজার বছর ধরে',
    'তুমি কি জানো',
    'আমার হৃদয়',
    'আকাশ ভরা সূর্য তারা',
    'বনলতা কি জানে',
    'সারাদিন মাল্যবানের মনেও ছিল না',
    'তুমি কি শুনতে পাবে',
    'আমার কবিতা',
    'বিদ্রোহের চিঠি',
    'হাজার বছর ধরে',
]


# List of all authors in the dataset read from the authors tokens file
@st.cache_data
def get_authors_list() -> list[str]:
    """Reads the author tokens file and returns a list of author names.

    Returns:
        List of author names extracted from the author tokens file.
    """
    author_tokens_path = SRC_DIR / 'banalata'/ 'data' / 'author_tokens.txt'
    print(author_tokens_path)
    with open(author_tokens_path, encoding='utf-8') as f:
        _ = f.read().strip().split('\n')
        _ = [line.split(':')[1].replace('|>', '').strip() for line in _ if line.strip()]

    return _


AUTHORS = get_authors_list()
CONTENT_TYPE = ['poem', 'prose']


st.set_page_config(
    page_title='বনলতা—Banalata',
    page_icon='📖',
    layout='wide',
    initial_sidebar_state='expanded',
)

_logger = logging.getLogger('banalata_app')
_logger.setLevel(logging.DEBUG)

# Configure logging to output to the terminal (stderr)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)

# Custom CSS to tighten the UI
st.markdown(
    """
<style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    /* Center all images in the app */
    [data-testid="stImageContainer"] {
        display: flex;
        justify-content: center;
        padding: 0;
    }
    [data-testid="stImageContainer"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for widgets that are programmatically updated
if 'prompt_text' not in st.session_state:
    st.session_state.prompt_text = random.choice(SAMPLE_BENGALI_TEXTS)

# Main UI
logo_path = APP_DIR / 'assets' / 'logo_3x2.png'
if logo_path.exists():
    logo = Image.open(logo_path)
else:
    st.error(f'Logo not found at {logo_path}')
    st.stop()
# Display the logo centered using a column for better structure
# This ensures the image container itself is centered
col1, col2 = st.columns([1, 2], border=False, gap='xxsmall', vertical_alignment='center')
with col1:
    st.image(logo, width=400)

with col2:
    st.markdown('### শতাব্দী পারের বাংলা সাহিত্যের প্রতিধ্বনি—Echoes across centuries of Bengali literature')

with st.expander(label='বনলতা সম্পর্কে—About Banalata'):
    st.write(
        'বনলতা হল ২৮ মিলিয়ন প্যারামিটার-বিশিষ্ট একটি বাংলা ভাষা মডেল, যা ৮ম থেকে ২০শ শতাব্দীর মধ্যবর্তী'
        ' সময়ের—কবিতা, গদ্য ও প্রবন্ধসহ—৩৫-এরও অধিক লেখকের সর্বজনীন ডোমেইনভুক্ত (public-domain)'
        ' সাহিত্যকর্মের ওপর ভিত্তি করে একেবারে গোড়া থেকে প্রশিক্ষিত। লেখকের পরিচয় বা সত্তার ওপর ভিত্তি'
        ' করে বিন্যস্ত হওয়ায়, এটি রবীন্দ্রনাথ, বঙ্কিমচন্দ্র, জীবনানন্দ দাশ, সুকান্ত এবং অন্যান্য লেখকের নিজস্ব'
        ' শৈলী ও কণ্ঠস্বরে স্বতন্ত্রধর্মী পাঠ্য বা লেখা তৈরি করতে সক্ষম। এটি সম্পূর্ণভাবে কোনো পূর্ব-প্রশিক্ষিত ওয়েট'
        ' (pretrained weights) ব্যবহার না করেই, বরং সুপরিকল্পিতভাবে বাছাইকৃত প্রায় ১০ মিলিয়ন টোকেন'
        ' সম্বলিত বাংলা সাহিত্যিক পাঠ্যের ওপর ভিত্তি করে গড়ে তোলা হয়েছে।'
    )
    st.write(
        '\nBanalata is a 28M parameter Bengali language model trained from scratch on public-domain'
        ' literary works spanning the 8th to 20th century—poetry, prose, and essays across'
        ' 35+ authors. Conditioned on author identity, it generates stylistically distinct'
        ' text in the voice of Tagore, Bankimchandra, Jibanananda Das, Sukanta, and others.'
        ' Built entirely without pretrained weights, on roughly 10M tokens of curated Bengali'
        ' literary text.'
    )


_device = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
)


@st.cache_resource
def load_model_cached(ckpt_path: Path, device: torch.device):
    """Cached wrapper for loading the model and tokenizer."""
    if not ckpt_path.exists():
        return None, None, None
    return s05_generate.load_model_and_tokenizer(str(ckpt_path), device)


_ckpt_path = SRC_DIR / 'banalata' / 'checkpoints' / 'ckpt_best.pt'
model, sp, tok_config = load_model_cached(_ckpt_path, _device)

if model is None:
    st.error(
        f'Model checkpoint not found at `{_ckpt_path}`.'
        ' Please ensure the project is correctly set up.'
    )
    st.stop()


def on_generate_text():
    """Callback function to generate poetry when the button is clicked."""
    # Access variables from session state to ensure they're available in callbacks
    author = st.session_state.get('author', AUTHORS[12])
    content_type = st.session_state.get('content_type', CONTENT_TYPE[0])
    temperature = st.session_state.get('temperature', 0.85)
    top_p = st.session_state.get('top_p', 0.92)
    max_tokens = st.session_state.get('max_length', 128)
    _logger.debug(
        'author: %s, content_type: %s, temperature: %s, top_p: %s, max_tokens: %s',
        author,
        content_type,
        temperature,
        top_p,
        max_tokens,
    )

    # Call s05_generate_poetry.py to generate poetry based on the prompt_text
    _formatted_prompt = (
        f'{TOK_BOW}<|author:{author}|><|{content_type}|>\n{st.session_state.prompt_text[:10]}'
    )
    _logger.debug('Formatted prompt for generation: %s...', _formatted_prompt)
    generated_poetry = s05_generate.generate(
        model=model,
        sp=sp,
        tok_config=tok_config,
        device=_device,
        prompt=st.session_state.prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        author=author,
        content_type=content_type,
    )
    _logger.debug('Generated poetry: %s', generated_poetry)
    # Change the text in the prompt_text text area to show the generated poetry
    st.session_state.prompt_text = '\n'.join(generated_poetry)


st.text_area(
    label='কবিতা বা গদ্য রচনার জন্য বাংলায় লিখুন—Enter Bengali text to compose poetry/prose:',
    key='prompt_text',
    height=300,
)

col3, col4, col5 = st.columns([2, 1, 4], border=False, gap='small')
with col3:
    st.button(label='✍️ রচনা করুন—Compose', type='primary', on_click=on_generate_text)

with col4:
    st.button(
        label='🗑️ মুছুন—Clear',
        type='secondary',
        on_click=lambda: st.session_state.update(prompt_text=''),
    )

with col5:
    st.button(
        label='🎲 নমুনা—Sample Bengali text',
        type='secondary',
        on_click=lambda: st.session_state.update(prompt_text=random.choice(SAMPLE_BENGALI_TEXTS)),
    )

# Sidebar for settings
with st.sidebar:
    st.selectbox('Author style to use', options=AUTHORS, index=12, key='author')
    st.selectbox('Content type to generate', options=CONTENT_TYPE, index=0, key='content_type')
    st.slider('Max Length', min_value=64, max_value=512, value=128, key='max_length')
    st.slider('Temperature', min_value=0.1, max_value=1.0, value=0.85, key='temperature')
    st.slider('Top-p (nucleus sampling)', min_value=0.0, max_value=1.0, value=0.92, key='top_p')


st.divider()

with st.expander(label='অনলাইন বাংলা কীবোর্ড—Online Bengali keyboard'):
    st.markdown(
        'আপনি যদি বাংলা কীবোর্ড ব্যবহার না করতে চান, তাহলে নিচের অনলাইন বাংলা কীবোর্ড ব্যবহার করে'
        ' বাংলা টেক্সট লিখতে পারেন।'
        '\n\nIf you do not have a Bengali keyboard, you can use the online'
        ' Bengali keyboards below to type Bengali text.'
        '\n- [Google Input Tools online](https://www.google.co.in/inputtools/try/)  '
        '\n- [Ekushey Phonetic Keyboard](https://ekushey.org/projects/browser_ime/qwerty-phonetic-legacy/)  '
    )

st.divider()

st.write('Copyright © 2026 Barun Saha. All rights reserved.')
