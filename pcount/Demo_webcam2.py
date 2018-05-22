import numpy as np
import cv2
import libbgs
import os
import configparser
from peopletracker import PeopleTracker

observe_alg = "MOG_closing"
config_path = "settings.ini"
areaTH = 1000

## BGS Library algorithms
#bgs = libbgs.FrameDifference()
#bgs = libbgs.StaticFrameDifference()
#bgs = libbgs.AdaptiveBackgroundLearning()
#bgs = libbgs.AdaptiveSelectiveBackgroundLearning()
#bgs = libbgs.DPAdaptiveMedian()
#bgs = libbgs.DPEigenbackground()
#bgs = libbgs.DPGrimsonGMM()
#bgs = libbgs.DPMean()
#bgs = libbgs.DPPratiMediod()
#bgs = libbgs.DPTexture()
#bgs = libbgs.DPWrenGA()
#bgs = libbgs.DPZivkovicAGMM()
#bgs = libbgs.FuzzyChoquetIntegral()
#bgs = libbgs.FuzzySugenoIntegral()
#bgs = libbgs.GMG() # if opencv 2.x
#bgs = libbgs.IndependentMultimodal()
#bgs = libbgs.KDE()
#bgs = libbgs.KNN() # if opencv 3.x
#bgs = libbgs.LBAdaptiveSOM()
#bgs = libbgs.LBFuzzyAdaptiveSOM()
#bgs = libbgs.LBFuzzyGaussian()
#bgs = libbgs.LBMixtureOfGaussians()
#bgs = libbgs.LBSimpleGaussian()
#bgs = libbgs.LBP_MRF()
#bgs = libbgs.LOBSTER()
#bgs = libbgs.MixtureOfGaussianV1() # if opencv 2.x
#bgs = libbgs.MixtureOfGaussianV2()
#bgs = libbgs.MultiCue()
#bgs = libbgs.MultiLayer()
#bgs = libbgs.PAWCS()
#bgs = libbgs.PixelBasedAdaptiveSegmenter()
#bgs = libbgs.SigmaDelta()
#bgs = libbgs.SuBSENSE()
#bgs = libbgs.T2FGMM_UM()
#bgs = libbgs.T2FGMM_UV()
#bgs = libbgs.T2FMRF_UM()
#bgs = libbgs.T2FMRF_UV()
#bgs = libbgs.VuMeter()
#bgs = libbgs.WeightedMovingMean()
#bgs = libbgs.WeightedMovingVariance()
#bgs = libbgs.TwoPoints()
#bgs = libbgs.ViBe()
#bgs = libbgs.CodeBook()

# load config
config = configparser.ConfigParser()
config.read(config_path)
source = config.get('video_source', 'source')
people_options = dict(config.items('person'))

# hog settings
hog_win_stride = config.getint('hog', 'win_stride')
hog_padding = config.getint('hog', 'padding')
hog_scale = config.getfloat('hog', 'scale')

# setup detectors
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# people tracking
finder = PeopleTracker(people_options=people_options)

def handle_the_people(frame):
	(rects, weight) = hog.detectMultiScale(frame, winStride=(hog_win_stride, hog_win_stride),
												padding=(hog_padding, hog_padding), scale=hog_scale)

	people = finder.people(rects)
	# draw triplines
	# for line in lines:
	# 	for person in people:
	# 		if line.handle_collision(person) == 1:
	# 			new_collision(person)
    #
	# 	frame = line.draw(frame)

	for person in people:
		frame = person.draw(frame)
		person.colliding = False

	return frame

video_file = source

capture = cv2.VideoCapture(video_file)
while not capture.isOpened():
	capture = cv2.VideoCapture(video_file)
	cv2.waitKey(1000)
	print("Wait for the header")

#pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
#pos_frame = capture.get(cv2.CV_CAP_PROP_POS_FRAMES)

for i in range(19):
    print i, capture.get(i)

capture.set(3,640) #set width
capture.set(4,360) #set height

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out_filename = 'observe/%s_%s.avi' % (os.path.splitext( os.path.basename(video_file))[0], observe_alg)
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (640,360), False)

# kernel for opening and closing
kernel = np.ones((5,5), np.uint8)


pos_frame = capture.get(1)
while True:
	flag, frame = capture.read()
	
	if flag:
		cv2.imshow('video', frame)
		#pos_frame = capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
		#pos_frame = capture.get(cv2.CV_CAP_PROP_POS_FRAMES)
		pos_frame = capture.get(1)
		#print str(pos_frame)+" frames"
		
		img_output = bgs.apply(frame)
		img_bgmodel = bgs.getBackgroundModel();

		ret,thresh1 = cv2.threshold(img_output,200,255,cv2.THRESH_BINARY)
		opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
		cv2.imshow('img_output', closing)

		contours0, hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		for cnt in contours0:
			cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
			area = cv2.contourArea(cnt)
			print area
			if area > areaTH:
				#################
				#   TRACKING    #
				#################            
				M = cv2.moments(cnt)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				x,y,w,h = cv2.boundingRect(cnt)
				cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
				img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

		frame = handle_the_people(img_output)

		cv2.imshow('counter', frame)

		cv2.imshow('img_bgmodel', img_bgmodel)
                out.write(img_output)

	else:
		#capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
		#capture.set(cv2.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
		#capture.set(1, pos_frame-1)
		#print "Frame is not ready"
		cv2.waitKey(1000)
		break
	
	if 0xFF & cv2.waitKey(10) == 27:
		break
	
	#if capture.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
	#if capture.get(cv2.CV_CAP_PROP_POS_FRAMES) == capture.get(cv2.CV_CAP_PROP_FRAME_COUNT):
	#if capture.get(1) == capture.get(cv2.CV_CAP_PROP_FRAME_COUNT):
		#break
capture.release()
out.release()
cv2.destroyAllWindows()
