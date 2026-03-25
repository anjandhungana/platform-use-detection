import cv2

def extractFrames(vid):
    cap = cv2.VideoCapture(vid)
    try:
        # Start from first frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = cap.read()

        if success:
            return frame
        return None
    finally:
        cap.release()
    

