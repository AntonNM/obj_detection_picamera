from picamera2 import Picamera2
import cv2
import time

def main():
    # Load label dictionary
    label_dict = {}
    with open('models/ssd_mobilenet/labels.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            label_dict[int(parts[0])] = parts[1].strip()

    # Load the TensorFlow model
    model = cv2.dnn.readNetFromTensorflow(
        'models/ssd_mobilenet/frozen_inference_graph.pb',
        'models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # Initialize the camera with the specified resolution
    camera = Picamera2()
    camera.rotation = 180
    resolution = (640, 480)
    camera.configure(camera.create_preview_configuration(main={"size": resolution, "format": "RGB888"}))
    camera.start()

    time.sleep(2)  # Allow the camera to warm up

    color = (23, 230, 210)  # BGR color for drawing bounding boxes

    try:
        while True:
            # Capture an image in RGB format and convert it to BGR for OpenCV
            image = camera.capture_array()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Resize the image to match the model's input size
            resized_image = cv2.resize(image_bgr, (300, 300))

            # Prepare the image for the model
            blob = cv2.dnn.blobFromImage(resized_image, size=(300, 300), swapRB=True, crop=False)
            model.setInput(blob)
            net_out = model.forward()

            # Process detections
            for detection in net_out[0, 0, :, :]:
                score = float(detection[2])
                if score > 0.5:
                    label = int(detection[1])
                    label_str = label_dict.get(label, 'unknown')
                    print(label_str)

                    # Calculate bounding box coordinates relative to the original image size
                    left = detection[3] * resolution[0]
                    top = detection[4] * resolution[1]
                    right = detection[5] * resolution[0]
                    bottom = detection[6] * resolution[1]

                    # Draw bounding box and label on the original image
                    cv2.rectangle(image_bgr, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)
                    cv2.putText(image_bgr, label_str, (int(left) + 3, int(bottom) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)

            # Display the image
            cv2.imwrite('output.jpg', image)

            # Check for 'q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # Clean up resources
        cv2.destroyAllWindows()
        camera.stop()

if __name__ == '__main__':
    main()