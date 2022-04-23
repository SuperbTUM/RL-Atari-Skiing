from PIL import Image, ImageFilter
import numpy as np


def process_state(state, ratio=0.6):
    state = Image.fromarray(state[28:-35, 8:152, :]).convert('L')
    preprocessed_state = np.asarray(state)
    preprocessed_state = np.where(preprocessed_state >= 180, 236, preprocessed_state)
    preprocessed_state = Image.fromarray(preprocessed_state)
    preprocessed_state = preprocessed_state.resize(
        (int(preprocessed_state.size[0] * ratio), int(preprocessed_state.size[1] * ratio)), Image.LANCZOS)
    assert preprocessed_state.size[0] >= 80
    preprocessed_state = preprocessed_state.filter(ImageFilter.EDGE_ENHANCE_MORE)
    state = np.expand_dims(np.asarray(preprocessed_state), axis=-1)
    return state.astype("float32")
