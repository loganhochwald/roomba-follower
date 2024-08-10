import cv2
import sys
import math

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s, cv2.CAP_DSHOW)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 400
mean = [104, 117, 123]
conf_threshold = 0.98

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Draw x and y axis lines
    cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (255, 0, 0), 2)  # X-axis
    cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (255, 0, 0), 2)  # Y-axis

    # Label the origin
    origin_label = "Origin"
    cv2.putText(frame, origin_label, (frame_width // 2 + 10, frame_height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0), thickness=3)
            label = "Human Detected"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Calculate center of the bounding box
            center_x = (x_left_bottom + x_right_top) // 2
            center_y = (y_left_bottom + y_right_top) // 2

            # Draw a small circle at the center point
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), thickness=-1)

            # Display the coordinates next to the center point
            coords_label = f"({center_x}, {center_y})"
            cv2.putText(frame, coords_label, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Calculate the angle relative to the top of the circle (0 degrees)
            delta_x = center_x - (frame_width // 2)
            delta_y = (frame_height // 2) - center_y  # Inverted y-axis to match the coordinate system
            
            angle = math.degrees(math.atan2(delta_x, delta_y))  # atan2(x, y) to get angle in degrees

            # Map angle to -180 to 180 degrees
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

            # Draw the circle with the angle indicated
            circle_center = (frame_width // 2, frame_height // 2)
            circle_radius = 100  # Adjust radius as needed

            # Draw the circle
            cv2.circle(frame, circle_center, circle_radius, (0, 255, 255), 2)

            # Calculate distance from the origin to the detected point
            distance = math.sqrt(delta_x**2 + delta_y**2)

            # Draw the extended line from the origin to the point
            end_x = int(circle_center[0] + distance * math.sin(math.radians(angle)))
            end_y = int(circle_center[1] - distance * math.cos(math.radians(angle)))
            cv2.line(frame, circle_center, (end_x, end_y), (255, 0, 255), 2)

            # Display angle
            angle_label = f"Angle: {angle:.2f}"
            cv2.putText(frame, angle_label, (frame_width // 2 - 50, frame_height // 2 + circle_radius + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    t, _ = net.getPerfProfile()
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
