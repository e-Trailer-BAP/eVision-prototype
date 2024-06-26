import cv2
import time
import threading
import os

# Define frame size and frame rate
frame_width = 1280
frame_height = 960
frame_rate = 30.0

# Define output path
output_path = 'data/videos'

# Ensure the output path exists
os.makedirs(output_path, exist_ok=True)

# Count the number of files in the output directory
file_count = len([name for name in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, name))])

# Determine the prefix based on the number of files
prefix = 't1' + ((file_count // 4) + 1)

# Define video capture objects for four video streams
caps = [cv2.VideoCapture(i) for i in range(1, 5)]

# Set the frame size and frame rate for each capture object
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Define the codec and create VideoWriter objects for each stream
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 files
out_files = [
    cv2.VideoWriter(os.path.join(output_path, f'{prefix}-side{i+1}.mp4'), fourcc, frame_rate, (frame_width, frame_height))
    for i in range(4)
]

# Start time for recording duration
start_time = time.time()
record_duration = 30  # seconds
stop_event = threading.Event()

def record_video(cap, out, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)

# Create and start threads for each video capture
threads = []
for cap, out in zip(caps, out_files):
    thread = threading.Thread(target=record_video, args=(cap, out, stop_event))
    thread.start()
    threads.append(thread)

# Run for the specified duration or until 'q' key is pressed
try:
    while time.time() - start_time < record_duration:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Signal all threads to stop
    stop_event.set()
    for thread in threads:
        thread.join()

    # Release everything if job is finished
    for cap in caps:
        cap.release()
    for out in out_files:
        out.release()
    cv2.destroyAllWindows()