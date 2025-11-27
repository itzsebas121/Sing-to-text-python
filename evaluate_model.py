import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from tensorflow.keras.models import load_model
from helpers import *
from constants import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_to_speech import text_to_speech

def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    
    # Handle edge cases
    if current_length == 0:
        return []
    if current_length == 1:
        return [keypoints[0]] * target_length
    if current_length == target_length:
        return keypoints
    
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    
    return interpolated_keypoints

def normalize_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return interpolate_keypoints(keypoints, target_length)
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return [keypoints[i] for i in indices]
    else:
        return keypoints
    
def evaluate_model(src=None, threshold=0.8, margin_frame=1, delay_frames=3, max_frames=50):
    kp_seq, sentence = [], []
    word_ids = get_word_ids(WORDS_JSON_PATH)
    model = load_model(MODEL_PATH)
    count_frame = 0
    fix_frames = 0
    recording = False
    
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(src or 0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret: break

            results = mediapipe_detection(frame, holistic_model)
            
            # Improved: Force prediction when reaching max_frames to prevent infinite capture
            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > margin_frame:
                    # Use normalized keypoints if enabled
                    from constants import USE_NORMALIZED_KEYPOINTS
                    if USE_NORMALIZED_KEYPOINTS:
                        kp_frame = extract_keypoints_normalized(results)
                    else:
                        kp_frame = extract_keypoints(results)
                    kp_seq.append(kp_frame)
                
                # Force prediction if max frames reached
                if count_frame >= max_frames:
                    if len(kp_seq) >= MIN_LENGTH_FRAMES:
                        kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                        res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                        
                        print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                        prob = res[np.argmax(res)]

                        if prob > threshold:
                            word_id = word_ids[np.argmax(res)].split('-')[0]
                            sent = words_text.get(word_id)
                        else:
                            sent = "No reconocido"

                        print("Resultado:", sent, f"({prob*100:.2f}%)")
                        sentence.insert(0, sent)

                        if sent != "No reconocido":
                            text_to_speech(sent)
                    
                    recording = False
                    fix_frames = 0
                    count_frame = 0
                    kp_seq = []
            
            else:
                if count_frame >= MIN_LENGTH_FRAMES + margin_frame:
                    fix_frames += 1
                    if fix_frames < delay_frames:
                        recording = True
                        continue
                    kp_seq = kp_seq[: - (margin_frame + delay_frames)]
                    
                    # Validate sufficient keypoints before prediction
                    if len(kp_seq) >= MIN_LENGTH_FRAMES:
                        kp_normalized = normalize_keypoints(kp_seq, int(MODEL_FRAMES))
                        
                        # Ensure normalized keypoints have correct shape
                        if len(kp_normalized) == int(MODEL_FRAMES):
                            res = model.predict(np.expand_dims(kp_normalized, axis=0))[0]
                            
                            print(np.argmax(res), f"({res[np.argmax(res)] * 100:.2f}%)")
                            prob = res[np.argmax(res)]

                            if prob > threshold:
                                word_id = word_ids[np.argmax(res)].split('-')[0]
                                sent = words_text.get(word_id)
                            else:
                                sent = "No reconocido"

                            print("Resultado:", sent, f"({prob*100:.2f}%)")
                            sentence.insert(0, sent)

                            if sent != "No reconocido":
                                text_to_speech(sent)
                
                recording = False
                fix_frames = 0
                count_frame = 0
                kp_seq = []
            
            if not src:
                cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
                cv2.putText(frame, ' | '.join(sentence), FONT_POS, FONT, FONT_SIZE, (255, 255, 255))
                
                draw_keypoints(frame, results)
                cv2.imshow('Sign Language Translator', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        video.release()
        cv2.destroyAllWindows()
        return sentence
    
if __name__ == "__main__":
    evaluate_model()
