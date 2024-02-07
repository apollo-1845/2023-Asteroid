# Get speed of ISS using ORB (Oriented FAST
# and Rotated BRIEF) Computer Vision algorithms

"""
TODO: Write Description; check rules
"""
import numpy as np

"""Imports and instantiation"""
import time
import cv2

# Import the picamera class
from picamera import PiCamera
# Instantiate it and set its resolution
piCam = PiCamera()
piCam.resolution = (4056, 3040)

# Allows image capture to numpy arrays
# from picamera.array import PiRGBArray

count = 1 # TODO: Remove

"""Constants"""
# Computer Vision: ORB
ORB_FEATURE_NUMBER = 500
BF_MATCHER_TYPE = cv2.NORM_HAMMING

# Image data
OPENCV_RESOLUTION = (3040, 4056, 3)
GROUND_SAMPLE_DISTANCE = 12648 / 100000 # Kilometres per pixel = cm per pixel / cm per km

"""Our classes"""
class TimedPhoto:
    """A photo stored in OpenCV format with a timestamp."""
    def __init__(self, camera: PiCamera):
        """Take a new photo and save an array representation of it in memory with its timestamp."""
        camera.capture("capture.png")
        self.array = cv2.imread("capture.png")
        # self.applyProcessing() TODO: Uncomment when works

        self.time = time.time()

    def applyProcessing(self):
        """Apply processing like removing clouds to this image to make it cleaner."""
        # Get blue, green, red channels
        b1, g1, r1 = cv2.split(self.array)
        # Hide clouds
        # TODO: There's currently a bug here!!!!!!!!!!!!!!!!!!!!
        self.mask = np.zeros(self.array.shape[:2], dtype=np.uint8)
        self.mask[(r1 > 150) & (g1 > 150) & (b1 > 150)] = 255 # Make mask visible
        cv2.imwrite("mask.png", self.mask)
        # TODO: Remove - Only for demo
        print("Arr", self.array.dtype, self.array.shape, "Mask", self.mask.dtype, self.mask.shape)
        self.array = cv2.bitwise_and(self.array, self.array, mask=self.mask)

    def __repr__(self):
        """Display this timed photo as text for use in the console."""
        return f"Photo(shape = ({self.array.shape[0]}, {self.array.shape[1]}, {self.array.shape[2]}), time = {self.time})"

class PhotoComparer:
    """2 timed photos which can be used to find ISS speed."""
    def __init__(self, photo1: TimedPhoto, photo2: TimedPhoto):
        """Create a photo comparer, assuming that photo1 was taken before photo2."""
        self.photo1 = photo1
        self.photo2 = photo2
        # self.distance = None

    def getDistance(self):
        """Use ORB and brute-force matching computer vision algorithms to get the distance between the 2 photos in pixels.
        We took inspiration and the main structure of the algorithm from https://projects.raspberrypi.org/en/projects/astropi-iss-speed/."""

        # Instantiate the OpenCV Computer Vision algorithm instances.
        # Use the ORB (oriented fast and rotated brief) OpenCV algorithm to get specific "features" in each of the two photos.
        orb = cv2.ORB.create(nfeatures = ORB_FEATURE_NUMBER)
        # Use the Brute Force matcher to find matches between features in the two photos.
        bruteForce = cv2.BFMatcher.create(BF_MATCHER_TYPE, crossCheck=True)

        # Get information about the features visible

        keypoints1, descriptors1 = orb.detectAndCompute(self.photo1.array, None) # self.photo1.mask) # (r1 < 200) | (g1 < 200) | (b1 < 200)) # Ignore clouds in the mask.
        keypoints2, descriptors2 = orb.detectAndCompute(self.photo2.array, None) # self.photo2.mask)

        # # TODO: Delete
        # global count
        # cv2.imwrite(f"masked{count}.png", self.photo1.array)

        # Get matches across the 2 photos about the features.
        matches = bruteForce.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # # TODO: Delete
        # match_img = cv2.drawMatches(self.photo1.array, keypoints1, self.photo2.array, keypoints2, matches[:100], None)
        # resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)

        # # global count
        # cv2.imwrite(f"match{count}.png", resize)
        # count += 1

        # Return the mean distance between the photos for all of the matches; mean distance = sum of distances / num matches.
        totalDistance = 0
        for match in matches:
            totalDistance += match.distance

        return totalDistance / len(matches)

    def getDistanceKilometres(self):
        """Get the distance between the two photos in KM"""
        return self.getDistance() * GROUND_SAMPLE_DISTANCE

    def getTimeDifference(self):
        """Return the difference in time taken between the two photos in seconds."""
        return self.photo2.time - self.photo1.time

    def getSpeed(self):
        """Get the speed of the ISS between the 2 photos in kilometres per second."""
        return self.getDistanceKilometres() / self.getTimeDifference()

    def __repr__(self):
        """Display this photo comparer as text for use in the console."""
        distance = self.getDistance()
        time = self.getTimeDifference()
        return f"PhotoComparer(time difference = {time}s, distance difference = {distance}px, speed = {distance/time}px/s)"

"""Our functions"""
def writeResult(speedKmps: float):
    """Write the result speed in km/s to the result.txt file, with a precision of 5sf"""

    # Format the speedKmps to have a precision
    # of 5 significant figures
    speedKmpsFormatted = "{:.4f}".format(speedKmps)

    # Write the formatted string to the file
    filePath = "result.txt"
    with open(filePath, 'w') as file:
        file.write(speedKmpsFormatted)

    print(f"Speed {speedKmpsFormatted} written to {filePath}")

"""Main entrypoint"""
# TODO: Timing
lastPhoto = TimedPhoto(piCam)
# Sum of speeds so far in pixels per second
totalSpeed = 0
time.sleep(1)
for i in range(5):
    print("Iteration", i+1)
    thisPhoto = TimedPhoto(piCam)
    comparer = PhotoComparer(lastPhoto, thisPhoto)
    totalSpeed += comparer.getSpeed()
    lastPhoto = TimedPhoto(piCam)
    time.sleep(1)

# Mean is sum of all speeds / num iterations
meanSpeed = totalSpeed / 5

writeResult(meanSpeed)