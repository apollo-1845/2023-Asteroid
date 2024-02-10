# Get speed of ISS using ORB (Oriented FAST# and Rotated BRIEF) Computer Vision algorithms"""TODO: Write Description; check rules"""import numpy as np"""Imports and instantiation"""from datetime import datetime, timedeltaimport timeimport cv2import os# Import the picamera classfrom picamera import PiCamera# Instantiate it and set its resolutionpiCam = PiCamera()piCam.resolution = (4056, 3040)# Allows image capture to numpy arrays# from picamera.array import PiRGBArraycount = 1 # TODO: Remove"""Constants"""# Experiment informationEXPERIMENT_DURATION_CONSERVATIVE = timedelta(minutes=5) # "conservative" means an overestimateITERATION_DURATION_MINIMUM = timedelta(seconds=10) # How long minimum to leave between consecutive photos# Filesfrom pathlib import PathbaseFolder = Path(__file__).parent.resolve()CAPTURE_FILE_PATH = baseFolder / "tmpCapture.png" # Temporary file to save photo captures toRESULT_FILE_PATH = baseFolder / "result.txt"# Computer Vision: ORBORB_FEATURE_NUMBER = 500BF_MATCHER_TYPE = cv2.NORM_HAMMING# Image dataOPENCV_RESOLUTION = (3040, 4056, 3)GROUND_SAMPLE_DISTANCE = 12648 / 100000 # Kilometres per pixel = cm per pixel / cm per km"""Our classes"""class TimedPhoto:    """A photo stored in OpenCV format with a timestamp."""    def __init__(self, camera: PiCamera):        """Take a new photo and save an array representation of it in memory with its timestamp."""        camera.capture(str(CAPTURE_FILE_PATH))        self.array = cv2.imread(str(CAPTURE_FILE_PATH))        self.applyProcessing()        self.time = time.time()    def applyProcessing(self):        """Apply processing like removing clouds to this image to make it cleaner."""        # Get blue, green, red channels        b1, g1, r1 = cv2.split(self.array)        # Hide clouds with a mask        self.mask = np.zeros(self.array.shape[:2], dtype=np.uint8)        self.mask[(r1 < 130) & (g1 < 130) & (b1 < 130)] = 255        # Apply mask to the image array: TODO: Remove        # self.array = cv2.bitwise_and(self.array, self.array, mask=self.mask)    def __repr__(self):        """Display this timed photo as text for use in the console."""        return f"Photo(shape = ({self.array.shape[0]}, {self.array.shape[1]}, {self.array.shape[2]}), time = {self.time})"class PhotoComparer:    """2 timed photos which can be used to find ISS speed."""    def __init__(self, photo1: TimedPhoto, photo2: TimedPhoto):        """Create a photo comparer, assuming that photo1 was taken before photo2."""        self.photo1 = photo1        self.photo2 = photo2        # self.distance = None    def getDistance(self):        """Use ORB and brute-force matching computer vision algorithms to get the distance between the 2 photos in pixels.        We took inspiration and the main structure of the algorithm from https://projects.raspberrypi.org/en/projects/astropi-iss-speed/."""        # Instantiate the OpenCV Computer Vision algorithm instances.        # Use the ORB (oriented fast and rotated brief) OpenCV algorithm to get specific "features" in each of the two photos.        orb = cv2.ORB.create(nfeatures = ORB_FEATURE_NUMBER)        # Use the Brute Force matcher to find matches between features in the two photos.        bruteForce = cv2.BFMatcher.create(BF_MATCHER_TYPE, crossCheck=True)        # Get information about the features visible        keypoints1, descriptors1 = orb.detectAndCompute(self.photo1.array, self.photo1.mask) # (r1 < 200) | (g1 < 200) | (b1 < 200)) # Ignore clouds in the mask.        keypoints2, descriptors2 = orb.detectAndCompute(self.photo2.array, self.photo2.mask)        # # TODO: Delete        global count        cv2.imwrite(f"masked{count}.png", self.photo1.array)        count += 1        # Get matches across the 2 photos about the features.        matches = bruteForce.match(descriptors1, descriptors2)        matches = sorted(matches, key=lambda x: x.distance)        # matches = matches[:20]        # # Use up to 100 most likely matches, thus removing some incorrect ones        # matches = matches[:100]        # TODO: Delete        match_img = cv2.drawMatches(self.photo1.array, keypoints1, self.photo2.array, keypoints2, matches[:100], None)        resize = cv2.resize(match_img, (1600, 600), interpolation=cv2.INTER_AREA)        # global count        cv2.imwrite(f"match{count}.png", resize)        count += 1        # Return the mean distance between the photos for all of the matches        # Create a 1D numpy array to store the distances, with the same length as the number of matches.        distances = np.zeros((len(matches),))        for i in range(len(matches)):            # Get Pythagorean Distance Between            (x1, y1) = keypoints1[matches[i].queryIdx].pt            (x2, y2) = keypoints2[matches[i].trainIdx].pt            distances[i] = ((x2-x1)**2 + (y2-y1)**2)**0.5 # match.distance        # Mean distance = sum of distances / num matches.        print("distances", distances, "len(matches)", len(matches))        # If there are 0 matches, a ZeroDivisionError will be thrown, effectively ignoring this frame.        if len(distances) == 0:            raise ZeroDivisionError()        return np.nanmedian(distances)    def getDistanceKilometres(self):        """Get the distance between the two photos in KM"""        distance = self.getDistance()        print("Distance (px):", distance)        return distance * GROUND_SAMPLE_DISTANCE    def getTimeDifference(self):        """Return the difference in time taken between the two photos in seconds."""        return self.photo2.time - self.photo1.time    def getSpeed(self):        """Get the speed of the ISS between the 2 photos in kilometres per second."""        # return self.getDistanceKilometres() / self.getTimeDifference()        distance = self.getDistanceKilometres()        print("Distance (km):", distance)        time = self.getTimeDifference()        print("Time Difference (s):", time)        speed = distance/time        print("Speed (km/s):", speed)        return speed    def __repr__(self):        """Display this photo comparer as text for use in the console."""        distance = self.getDistance()        time = self.getTimeDifference()        return f"PhotoComparer(time difference = {time}s, distance difference = {distance}px, speed = {distance/time}px/s)""""Our functions"""def writeResult(speedKmps: float):    """Write the result speed in km/s to the result.txt file, with a precision of 5sf"""    # Format the speedKmps to have a precision    # of 5 significant figures    speedKmpsFormatted = "{:.4f}".format(speedKmps)    # Write the formatted string to the file    filePath = "result.txt"    with open(filePath, 'w') as file:        file.write(speedKmpsFormatted)    print(f"Speed {speedKmpsFormatted} written to {filePath}")"""Main entrypointInspired by https://projects.raspberrypi.org/en/projects/mission-space-lab-creator-guide/3"""print("Starting")# TODO: test + commit the below code; add try/catch# Create a variable to store the start time of the whole program.startTime = datetime.now()# Create a variable to store the start time of the loop.startLoopTime = datetime.now()# Run until a photo is successfully captured or the time runs out.validPhoto = Falsewhile(not validPhoto and datetime.now() < startLoopTime + EXPERIMENT_DURATION_CONSERVATIVE):    try:        # The last photo taken, to be compared with the current photo.        lastPhoto = TimedPhoto(piCam)        validPhoto = True    except Exception as error:        # Allow keyboard interrupts to end the program properly.        if type(error) == KeyboardInterrupt:            raise error        validPhoto = False# Run the following if a photo was successfully capturedif(validPhoto):    # The sum of all of the speeds calculated for each iteration of the loop    totalSpeed = 0    # Number of times the loop has been run    numIterations = 0    # Run the main loop for 1 minute, while the current time is less than 1 minute after the start time.    while startLoopTime < startTime + EXPERIMENT_DURATION_CONSERVATIVE:        print("Iteration", numIterations+1)        try:            thisPhoto = TimedPhoto(piCam)            comparer = PhotoComparer(lastPhoto, thisPhoto)            totalSpeed += comparer.getSpeed()            lastPhoto = TimedPhoto(piCam)            # Loop iteration must take 10s minimum to run, between photos.            while datetime.now() < startLoopTime + ITERATION_DURATION_MINIMUM:                time.sleep(1)            numIterations += 1        except Exception as error:            # Allow keyboard interrupts to end the program properly.            if type(error) == KeyboardInterrupt:                raise error            pass # Don't worry about errors such as no-matches-found error.        startLoopTime = datetime.now()    # Out of the loop — stopping    # Mean is sum of all speeds / num iterations    meanSpeed = totalSpeed / numIterations    writeResult(meanSpeed)"""Close open resources and clean up"""piCam.close()# Delete the temporarily captured photoos.remove(CAPTURE_FILE_PATH)