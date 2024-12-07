import cv2
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr


# Text-to-Speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# Speech Recognition
def get_user_input(prompt):
    text_to_speech(prompt)
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            text_to_speech("I couldn't understand. Please repeat.")
        except sr.RequestError as e:
            text_to_speech(f"Speech recognition error: {e}")
    return None


# Capture Image
def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        text_to_speech("Camera not accessible.")
        return None

    text_to_speech("Please position your camera. Capturing the image now.")
    ret, frame = cam.read()
    cam.release()
    if ret:
        return frame
    else:
        text_to_speech("Failed to capture image.")
        return None

# https://huggingface.co/docs/api-inference/tasks/object-detection#object-detection
# Detect Objects
def detect_objects(image):
    model = YOLO('yolov8l.pt')  # Use a larger YOLOv8 model for better detection
    results = model(image)

    detected_objects = []
    for r in results:
        for obj in r.boxes:
            cls = int(obj.cls[0])
            xywh = obj.xywh[0]
            detected_objects.append((model.names[cls], xywh))  # Name and bounding box (x, y, w, h)

    return detected_objects


# Determine Object Position
def get_object_position(x, y, width, height, image_width, image_height):
    cx, cy = x + width / 2, y + height / 2  # Center of the bounding box
    relative_x = cx / image_width
    relative_y = cy / image_height

    if relative_x < 0.33 and relative_y < 0.33:
        return "top left"
    elif relative_x > 0.66 and relative_y < 0.33:
        return "top right"
    elif relative_x < 0.33 and relative_y > 0.66:
        return "bottom left"
    elif relative_x > 0.66 and relative_y > 0.66:
        return "bottom right"
    else:
        return "center"


# Check Threshold for Capture
def is_within_threshold(x, y, w, h, image_width, image_height, desired_position):
    target_positions = {
        "top left": (0.17, 0.17),
        "top right": (0.83, 0.17),
        "bottom left": (0.17, 0.83),
        "bottom right": (0.83, 0.83),
        "center": (0.5, 0.5),
    }
    target_x, target_y = target_positions[desired_position]
    cx, cy = x + w / 2, y + h / 2
    relative_cx = cx / image_width
    relative_cy = cy / image_height

    threshold = 0.4  # Increased the threshold to 40% for easier alignment
    return abs(relative_cx - target_x) <= threshold and abs(relative_cy - target_y) <= threshold


# Provide Guidance to User
def guide_user_to_position(current_position, desired_position):
    directions = {
        ("top left", "center"): "Move down and to the right.",
        ("top right", "center"): "Move down and to the left.",
        ("bottom left", "center"): "Move up and to the right.",
        ("bottom right", "center"): "Move up and to the left.",
        ("top left", "top right"): "Move to the right.",
        ("top right", "top left"): "Move to the left.",
        ("bottom left", "bottom right"): "Move to the right.",
        ("bottom right", "bottom left"): "Move to the left.",
    }
    return directions.get((current_position, desired_position), "Adjust the object to the desired position.")


# Annotate Image with Bounding Boxes
def annotate_image(image, objects):
    for obj in objects:
        name, bbox = obj
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


# Ensure Valid Position
def get_valid_position():
    valid_positions = ["top left", "top right", "bottom left", "bottom right", "center"]
    while True:
        desired_position = get_user_input(
            "Where would you like to place the object? You can say top left, top right, bottom left, bottom right, or center."
        )

        # Normalize input
        desired_position = desired_position.strip().lower()

        if desired_position in valid_positions:
            return desired_position
        else:
            text_to_speech("Invalid position. Please say top left, top right, bottom left, bottom right, or center.")


# Main Program
def main():
    user_response = get_user_input("Do you want to capture the object? Please say yes or no.")
    if user_response != "yes":
        text_to_speech("Exiting the application.")
        return

    while True:
        frame = capture_image()
        if frame is None:
            return

        objects = detect_objects(frame)
        if not objects:
            text_to_speech("No objects detected in the image. Please try again.")
            continue

        image_height, image_width, _ = frame.shape

        detected_names = [obj[0] for obj in objects]
        text_to_speech(f"I have detected the following objects: {', '.join(detected_names)}.")
        user_choice = get_user_input("Which object would you like to position?")
        if user_choice not in detected_names:
            text_to_speech(f"I couldn't find {user_choice} in the image.")
            continue

        chosen_object = next(obj for obj in objects if obj[0] == user_choice)
        x, y, w, h = chosen_object[1]

        desired_position = get_valid_position()

        while True:
            is_aligned = is_within_threshold(x, y, w, h, image_width, image_height, desired_position)

            if is_aligned:
                text_to_speech(f"The object is in the {desired_position}. Capturing the image.")
                cv2.imwrite("final_image.jpg", frame)
                text_to_speech("Image captured Successfully.")
                return
            else:
                current_position = get_object_position(x, y, w, h, image_width, image_height)
                movement_instructions = guide_user_to_position(current_position, desired_position)
                text_to_speech(f"The object is currently at {current_position}. {movement_instructions}")

                frame = capture_image()
                objects = detect_objects(frame)
                chosen_object = next(obj for obj in objects if obj[0] == user_choice)
                x, y, w, h = chosen_object[1]

        cv2.imshow("Detection Results", annotate_image(frame, objects))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()