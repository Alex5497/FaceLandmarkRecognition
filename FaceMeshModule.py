import time

import cv2
import mediapipe as mp


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawingSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=3)

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


def main():
    cap = cv2.VideoCapture("Videos/Video2.mp4")
    pTime = 0

    detector = FaceMeshDetector(maxFaces=2)

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(len(faces))

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
