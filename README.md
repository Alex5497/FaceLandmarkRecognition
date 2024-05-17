# Face Mesh Detector

This repository contains a Python script for detecting and drawing face meshes using the MediaPipe library and OpenCV. The script captures video input, processes each frame to detect face landmarks, and displays the results with annotated face landmarks and frame rate (FPS).

## Requirements

- Python 3.6 or higher
- OpenCV
- MediaPipe

## Installation

1. Clone the repository to your local machine.
2. Install the required packages:
   ```bash
   pip install opencv-python mediapipe
   ```

## Usage

1. Ensure you have a video file named `Video2.mp4` in a `Videos` directory, or update the video path in the script.
2. Run the script:
   ```bash
   python face_mesh_detector.py
   ```

## Code Explanation

### FaceMeshDetector Class

This class initializes the MediaPipe Face Mesh solution and provides a method to detect and draw face landmarks.

- **Initialization**:
  ```python
  def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
      ...
      self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                               min_detection_confidence=self.minDetectionCon,
                                               min_tracking_confidence=self.minTrackCon)
      self.drawingSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=3)
  ```

- **findFaceMesh Method**:
  This method processes an image to find face landmarks and optionally draws them on the image.
  ```python
  def findFaceMesh(self, image, draw=True):
      self.imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      self.results = self.faceMesh.process(self.imgRGB)
      faces = []
      if self.results.multi_face_landmarks:
          for faceLms in self.results.multi_face_landmarks:
              face = []
              if draw:
                  self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                             landmark_drawing_spec=self.drawingSpec)
              for id, lm in enumerate(faceLms.landmark):
                  ih, iw, ic = image.shape
                  x, y = int(lm.x * iw), int(lm.y * ih)
                  cv2.putText(image, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                  face.append([x, y])
              faces.append(face)
      return image, faces
  ```

### main Function

This function captures video frames, uses the `FaceMeshDetector` to detect face landmarks, and displays the annotated frames along with the FPS.

```python
def main():
    cap = cv2.VideoCapture("Videos/Video2.mp4")
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)
```

## Example Output

The script will display the video with annotated face landmarks and the FPS count.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project uses the MediaPipe Face Mesh solution by Google and OpenCV for video processing.

---

Feel free to modify and extend the script to suit your needs. Contributions and suggestions are welcome!
