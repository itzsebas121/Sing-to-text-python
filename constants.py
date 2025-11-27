import os
import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_FRAMES = 15

# FEATURE EXTRACTION MODE
USE_NORMALIZED_KEYPOINTS = True  # Set to True for position-invariant recognition

# PATHS
ROOT_PATH = os.getcwd()
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions")
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_JSON_PATH = os.path.join(DATA_PATH, "data.json")
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}.keras")
KEYPOINTS_PATH = os.path.join(DATA_PATH, "keypoints")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words.json")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (5, 30)

words_text = {
    "hola": "HOLA",
    "que_haces": "¿QUE HACES?",
    "como_estas": "¿COMO ESTAS?",
    "buenos_dias":"Buenos dias",
    "adios":"Adios",
    "gusto_conocerte":"Gusto conocerte",
    "nos_vemos":"Nos vemos",
    "mejor_amigo":"Mejor amigo",
    "como_te_llamas":"¿Como te llamas?"
}