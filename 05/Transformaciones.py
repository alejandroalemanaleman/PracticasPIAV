import cv2 as cv
import numpy as np

class Transformaciones:
    def __init__(self, img):
        self.img = img
        self.transformed = self.img.copy()
        self.distorcioned = self.img.copy()
        self.h, self.w = self.img.shape[:2]
        self.size = (self.w, self.h)
        self.initial_values = {
            'tx': self.w,
            'ty': self.h,
            'Angulo': 0,
            'Centro X': self.w // 2,
            'Centro Y': self.h // 2,
            'Uniforme': 0,
            'Escalado X': 100,
            'Escalado Y': 100
        }
        
    def nothing(self, x):
        pass
    
    def create_trackbars_transformacion(self):
        cv.namedWindow("Transformaciones")

        cv.createTrackbar('tx', 'Transformaciones', self.w, 2 * self.w, self.nothing)
        cv.createTrackbar('ty', 'Transformaciones', self.h, 2 * self.h, self.nothing)

        cv.createTrackbar('Angulo', 'Transformaciones', 0, 360, self.nothing)
        cv.createTrackbar('Centro X', 'Transformaciones', self.w // 2, self.w, self.nothing)
        cv.createTrackbar('Centro Y', 'Transformaciones', self.h // 2, self.h, self.nothing)

        cv.createTrackbar('Uniforme', 'Transformaciones', 0, 1, self.nothing)
        cv.createTrackbar('Escalado X', 'Transformaciones', 100, 300, self.nothing)
        cv.setTrackbarMin('Escalado X', 'Transformaciones', 10)
        cv.createTrackbar('Escalado Y', 'Transformaciones', 100, 300, self.nothing)
        cv.setTrackbarMin('Escalado Y', 'Transformaciones', 10)
        
    def reset_trackbars(self):
        for name, val in self.initial_values.items():
            cv.setTrackbarPos(name, 'Transformaciones', val)
        print("✅ Parámetros reiniciados")
    
    def transformacion(self):
        self.create_trackbars_transformacion()
        self.reset_trackbars()
        
        while True:
            tx = cv.getTrackbarPos('tx', 'Transformaciones') - self.w
            ty = cv.getTrackbarPos('ty', 'Transformaciones') - self.h
            ang = cv.getTrackbarPos('Angulo', 'Transformaciones')
            cx = cv.getTrackbarPos('Centro X', 'Transformaciones')
            cy = cv.getTrackbarPos('Centro Y', 'Transformaciones')
            sx = cv.getTrackbarPos('Escalado X', 'Transformaciones') / 100
            sy = cv.getTrackbarPos('Escalado Y', 'Transformaciones') / 100
            uniforme = cv.getTrackbarPos('Uniforme', 'Transformaciones')

            if uniforme == 1:
                sx = sy

            R = cv.getRotationMatrix2D((cx, cy), ang, 1.0)
            rotated = cv.warpAffine(self.img, R, self.size)

            S = np.float32([[sx, 0, 0],
                            [0, sy, 0]])
            scaled = cv.warpAffine(rotated, S, self.size)

            T = np.float32([[1, 0, tx],
                            [0, 1, ty]])
            final = cv.warpAffine(scaled, T, self.size)

            cv.imshow("Transformaciones", final)
            
            self.transformed = final.copy()

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC -> salir del programa
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_trackbars()

        cv.destroyAllWindows()
        cv.waitKey(1)
    
    def apply_distortion(self, image, k1, k2):
        h, w = image.shape[:2]

        distCoeff = np.zeros((4, 1), np.float64)
        distCoeff[0, 0] = k1
        distCoeff[1, 0] = k2

        cam = np.eye(3, dtype=np.float32)
        cam[0, 2] = w / 2.0  
        cam[1, 2] = h / 2.0  
        cam[0, 0] = 10.0  
        cam[1, 1] = 10.0      

        distorted_img = cv.undistort(image, cam, distCoeff)
   
        return distorted_img
    
    def create_trackbars_distorcion(self):
        cv.namedWindow("Distorsiones")
        cv.createTrackbar('K1', 'Distorsiones', 500, 1000, self.nothing)
        cv.createTrackbar('K2', 'Distorsiones', 500, 1000, self.nothing)
        
    def reset_trackbars_distorcion(self):
        cv.setTrackbarPos('K1', 'Distorsiones', 500)
        cv.setTrackbarPos('K2', 'Distorsiones', 500)
        
    def distorcion(self):
        self.create_trackbars_distorcion()
        self.reset_trackbars_distorcion()
        
        while True:
            k1 = (cv.getTrackbarPos('K1', 'Distorsiones') - 500) / 100000
            k2 = (cv.getTrackbarPos('K2', 'Distorsiones') - 500) / 100000

            dist_img = self.apply_distortion(self.img, k1, k2)

            cv.imshow("Distorsiones", dist_img)
            
            self.distorcioned = dist_img.copy()

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC -> salir del programa
                break
            elif key == ord('r') or key == ord('R'):
                self.reset_trackbars_distorcion()

        cv.destroyAllWindows()
        cv.waitKey(1)