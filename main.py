# Get speed of ISS using ORB (Oriented FAST
# and Rotated BRIEF) Computer Vision algorithms

"""
Our method includes:
- Taking regular images with the camera
- Identifying clouds in the photos
- Using the ORB OpenCV algorithm to find matches between photos where there are no clouds
- Therefore finding the distance between photos and speed of the ISS.
"""

"""Imports"""
import numpy as np # NumPy
import cv2 # OpenCV, installed as opencv-contrib-python-headless
from datetime import datetime, timedelta # From Python Standard Library
import time # From Python Standard Library, for waiting specific numbers of seconds
import os # From Python Standard Library, only for deleting the tmpCapture.png file that is created

"""Instantiation"""

# Import the picamera class, to access the Astro Pi computer's camera
from picamera import PiCamera
# Instantiate it and set its resolution
piCam = PiCamera()
piCam.resolution = (4056, 3040)

"""Constants"""
# Experiment information
EXPERIMENT_DURATION_CONSERVATIVE = timedelta(minutes=9, seconds=30) # "conservative" means an underestimate, so the program does not run for too long
ITERATION_DURATION_MINIMUM = timedelta(seconds=10) # How long minimum to leave between consecutive photos

# Files
from pathlib import Path
baseFolder = Path(__file__).parent.resolve()
# String paths are supported by all the functions used, but the path datatype isn't.
CAPTURE_FILE_PATH = str(baseFolder / "tmpCapture.png") # Temporary file to save photo captures to
RESULT_FILE_PATH = str(baseFolder / "result.txt")

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
        camera.capture(CAPTURE_FILE_PATH)
        self.array = cv2.imread(CAPTURE_FILE_PATH)
        self.addMask()

        self.time = time.time()

    def addMask(self):
        """Make a mask to hide clouds from the image so only the land/sea is used to find ISS speed."""
        # Get blue, green, red channels
        b1, g1, r1 = cv2.split(self.array)

        # Hide clouds with a mask
        self.mask = np.zeros(self.array.shape[:2], dtype=np.uint8)
        self.mask[(r1 < 130) & (g1 < 130) & (b1 < 130)] = 255

    def __repr__(self):
        """Display this timed photo as text for use in the console."""
        return f"Photo(shape = ({self.array.shape[0]}, {self.array.shape[1]}, {self.array.shape[2]}), time = {self.time})"

class PhotoComparer:
    """2 timed photos which can be used to find ISS speed."""
    def __init__(self, photo1: TimedPhoto, photo2: TimedPhoto):
        """Create a photo comparer, assuming that photo1 was taken before photo2."""
        self.photo1 = photo1 # Taken first
        self.photo2 = photo2 # Taken second

    def getDistance(self):
        """Use ORB and brute-force matching computer vision algorithms to get the distance between the 2 photos in pixels.
        We took inspiration and the main structure of the algorithm from https://projects.raspberrypi.org/en/projects/astropi-iss-speed/."""

        # Instantiate the OpenCV Computer Vision algorithm instances.
        # Use the ORB (oriented fast and rotated brief) OpenCV algorithm to get specific "features" in each of the two photos.
        orb = cv2.ORB.create(nfeatures = ORB_FEATURE_NUMBER)
        # Use the Brute Force matcher to find matches between features in the two photos.
        bruteForce = cv2.BFMatcher.create(BF_MATCHER_TYPE, crossCheck=True)

        # Get information about the features visible
        keypoints1, descriptors1 = orb.detectAndCompute(self.photo1.array, self.photo1.mask) # Ignore clouds in the mask.
        keypoints2, descriptors2 = orb.detectAndCompute(self.photo2.array, self.photo2.mask) # Ignore clouds in the mask.

        # Get matches across the 2 photos between similar features.
        matches = bruteForce.match(descriptors1, descriptors2)

        # Return the mean distance between the photos for all of the matches
        # Create a one-dimensional numpy array to store the distances, with the same length as the number of matches.
        distances = np.zeros((len(matches),))

        for i in range(len(matches)):
            # Get Pythagorean Distance Between
            (x1, y1) = keypoints1[matches[i].queryIdx].pt
            (x2, y2) = keypoints2[matches[i].trainIdx].pt
            distances[i] = ((x2-x1)**2 + (y2-y1)**2)**0.5 # match.distance

        # Return the median average distance between matches = distance between photos
        # If there are 0 matches, a NaN (not-a-number) value will be returned, effectively ignoring this frame.
        if len(distances) == 0:
            return np.nan
        return np.nanmedian(distances)

    def getDistanceKilometres(self):
        """Get the distance between the two photos in KM"""
        distance = self.getDistance()
        return distance * GROUND_SAMPLE_DISTANCE

    def getTimeDifference(self):
        """Return the difference in time taken between the two photos as a timestamp."""
        return self.photo2.time - self.photo1.time

    def getSpeed(self):
        """Get the speed of the ISS between the 2 photos in kilometres per second."""
        distance = self.getDistanceKilometres() # in km
        time = self.getTimeDifference() # in seconds, as calculating with a timestamp
        speed = distance/time
        return speed

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
    with open(RESULT_FILE_PATH, 'w') as file:
        file.write(speedKmpsFormatted)

"""Main entrypoint
Inspired by https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/3."""

# Create a variable to store the start time of the whole program.
startTime = datetime.now()
# Create a variable to store the start time of the loop.
startLoopTime = datetime.now()

# Try to take a photo until a photo is successfully captured or the experiment time runs out.
validPhoto = False
while(not validPhoto and datetime.now() < startLoopTime + EXPERIMENT_DURATION_CONSERVATIVE):
    try:
        # The last photo taken, to be compared with the current photo.
        lastPhoto = TimedPhoto(piCam)
        validPhoto = True
    except Exception as error:
        # Allow keyboard interrupts to end the program properly.
        if type(error) == KeyboardInterrupt:
            raise error

        validPhoto = False

# Run the following if a photo was successfully captured
if(validPhoto):
    # The speeds calculated for each iteration of the loop as a list
    speeds = []

    # Run the main loop for the experiment duration, while the current time is less than the experiment duration after the start time.
    while startLoopTime < startTime + EXPERIMENT_DURATION_CONSERVATIVE:
        try:
            thisPhoto = TimedPhoto(piCam)
            comparer = PhotoComparer(lastPhoto, thisPhoto)
            speeds.append(comparer.getSpeed())
            lastPhoto = TimedPhoto(piCam)

            # Loop iteration must take 10s minimum to run, between photos.
            while datetime.now() < startLoopTime + ITERATION_DURATION_MINIMUM:
                time.sleep(1)
        except Exception as error:
            # Allow keyboard interrupts to end the program properly.
            if type(error) == KeyboardInterrupt:
                raise error
            pass # Don't worry about errors such as no-matches-found error.

        startLoopTime = datetime.now()
    # Out of the loop - stopping

    # Speeds recorded for each frame are saved in a numpy array.
    speeds = np.array(speeds, dtype=float)
    # Get median speed to write to file, ignoring not-a-number frames where no matches were found.
    medianSpeed = np.nanmedian(speeds)

    writeResult(medianSpeed)

"""Close open resources and clean up"""
piCam.close()
# Delete the temporarily captured photo
os.remove(CAPTURE_FILE_PATH)
