import cv2
import numpy as np
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    model = load_model("sign_model.h5")

    labels = [chr(i) for i in range(65, 91) if i not in [74, 90]]  # A-Z minus J and Z (25 letters)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        roi = frame[100:400, 100:400]
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)

        result = model.predict(reshaped, verbose=0)
        predicted_index = int(np.argmax(result))

        if predicted_index < len(labels):
            prediction = labels[predicted_index]
        else:
            prediction = "?"
            print(f"[Warning] Invalid predicted index: {predicted_index}, skipping label lookup.")

        cv2.putText(frame, f'Prediction: {prediction}', (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

