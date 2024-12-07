import cv2
import mediapipe as mp
import pyttsx3
import speech_recognition as sr

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speak(text):
    """Text-to-speech function"""
    print(text)
    engine.say(text)
    engine.runAndWait()

# https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb
def get_user_position():
    """Ask the user to specify the desired face position using speech input."""
    while True:
        speak("Please specify the face position. You can say top left, top right, bottom left, bottom right, or center.")
        with sr.Microphone() as source:
            print("Listening for user position...")
            try:
                audio = recognizer.listen(source)
                recognized_text = recognizer.recognize_google(audio).lower()
                print(f"Recognized speech: {recognized_text}")

                # Map recognized text to valid positions
                position_mapping = {
                    "top left": ["top left", "upper left", "left top", "top-left", "upper-left"],
                    "top right": ["top right", "upper right", "right top", "top-right", "upper-right"],
                    "bottom left": ["bottom left", "lower left", "left bottom", "bottom-left", "lower-left"],
                    "bottom right": ["bottom right", "lower right", "right bottom", "bottom-right", "lower-right"],
                    "center": ["center", "middle", "in the middle", "centre"],
                }

                for valid_position, keywords in position_mapping.items():
                    if any(keyword in recognized_text for keyword in keywords):
                        speak(f"You said {valid_position}. Is that correct? Please say yes or no.")
                        with sr.Microphone() as confirm_source:
                            confirmation_audio = recognizer.listen(confirm_source)
                            confirmation = recognizer.recognize_google(confirmation_audio).lower()
                            if "yes" in confirmation:
                                speak(f"Position confirmed as {valid_position}. Let's position your face.")
                                return valid_position
                            else:
                                speak("Okay, let's try again.")
                                break
                speak("That does not seem like a valid position. Please try again.")
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that. Please try again.")
                

def face_position_in_quadrant(face_box, width, height, tolerance=0.1):

    x_center = face_box.xmin + face_box.width / 2
    y_center = face_box.ymin + face_box.height / 2

    # Quadrant with tolerance: 
    left_bound = 0.5 - tolerance
    right_bound = 0.5 + tolerance
    top_bound = 0.5 - tolerance
    bottom_bound = 0.5 + tolerance

    # Determine position based on center coordinates
    if left_bound < x_center < right_bound and top_bound < y_center < bottom_bound:
        return "center"
    elif x_center < left_bound and y_center < top_bound:
        return "top left"
    elif x_center > right_bound and y_center < top_bound:
        return "top right"
    elif x_center < left_bound and y_center > bottom_bound:
        return "bottom left"
    elif x_center > right_bound and y_center > bottom_bound:
        return "bottom right"
    else:
        return "out of view"

def guide_user(current_position, target_position):
    """Provide guiding instructions to align the face."""
    if current_position == target_position:
        speak(f"Your face is correctly positioned in the {target_position}. Capturing the image.")
        return True  # Face is correctly positioned
    elif current_position == "out of view":
        # If the face is not detected, ask user to move towards the target position
        speak(f"Move your face to the {target_position}.")
    else:
        # Provide specific directional feedback to move face
        if "top" in target_position and "bottom" in current_position:
            speak("Move up.")
        if "bottom" in target_position and "top" in current_position:
            speak("Move down.")
        if "left" in target_position and "right" in current_position:
            speak("Move left.")
        if "right" in target_position and "left" in current_position:
            speak("Move right.")
        if target_position == "center":
            speak("Move to the center.")
    return False

def process_camera():
    """Main function to process webcam input."""
    cap = cv2.VideoCapture(0)
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

    # Get user input for desired position
    user_position = get_user_position()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detector.process(image_rgb)

        if result.detections:
            for detection in result.detections:
                # Get bounding box and determine face position
                box = detection.location_data.relative_bounding_box
                current_position = face_position_in_quadrant(box, width, height)
                print(f"Current Position: {current_position}, Target Position: {user_position}")

                # Provide guidance and capture image if positioned correctly
                if guide_user(current_position, user_position):
                    mp_drawing.draw_detection(frame, detection)
                    output_image_path = "captured_image.jpg"
                    cv2.imwrite(output_image_path, frame)
                    speak(f"Image captured successfully! The image has been saved as {output_image_path}.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        else:
            speak(f"Move your face to the {user_position}.")  # User needs to move to the desired position

        # Display the webcam feed
        cv2.imshow('Webcam', frame)

        # Break loop on 'q' press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the application
process_camera()
