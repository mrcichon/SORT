#!\bin\python2.7

"""
Main module for the real-time tracker class execution. Based on the SORT algorithm
"""

# note to self naprawic RuntimeError: OrderedDict mutated during iteration
# File "/home/mrc/Desktop/SORT-danny_opencv/tracker.py", line 54, in handle_no_detections
# for track_id in self.disappeared.keys():

from __future__ import print_function
import os.path
from tracker import KalmanTracker, ORBTracker, ReIDTracker
from imutils.video import WebcamVideoStream
import numpy as np
import cv2
from tracker_utils import bbox_to_centroid
import random
import colorsys
from human_detection import DetectorAPI
import face_recognition
import os
import glob

#  intializing shite for face_recognition
known_face_names = []
known_face_encodings = []

# reading jpg names from folder & adding them  to list
file = [os.path.basename(x) for x in glob.glob(r'images//' + '*.jpg')]
known_face_names.extend(file)

# adding faces from folder to encodings
for filename in glob.glob('images/*.jpg'):
    filename_image = face_recognition.load_image_file(filename)
    filename_encoding = face_recognition.face_encodings(filename_image)[0]
    known_face_encodings.append(filename_encoding)

#  getting rid of .jpg from list
known_face_names = " ".join(known_face_names).replace(".jpg", " ").split()

# nie no bez jaj chyba widac co sie dzieje
face_locations = []
face_encodings = []
face_names = []

# it might work, it might not
# slightly retarded attempt at associating recognized names with given bbox id
class Tracker:
    def __init__(self, intboxes, listnames):
        self.listtracker = []
        intboxes = intboxes + 1
        self.intboxes = intboxes
        self.listnames = listnames
        self.listboxes = list(range(intboxes))
        self.listtracker.append(self.listnames)
        self.listtracker.append(self.listboxes)
        # self.listtracker = list(zip(self.listboxes, self.listnames))

    def tracking(self):
        check = any(tname in sublist for sublist in self.listtracker for tname in self.listnames)
        # check = any(tname in self.listnames for tname in self.listnames)
        # self.realtracker  = list(self.listboxes)
        # self.realtracker = list(zip(self.listboxes, self.listnames))
        if check is True:
            realtracker = list(zip(self.listboxes, self.listnames))
            return realtracker

# main thing
class SORT:
    def __init__(self, src=None, tracker='Kalman', detector='faster-rcnn', benchmark=False):
        """
         Sets key parameters for SORT
        :param src: path to video file
        :param tracker: (string) 'ORB', 'Kalman' or 'ReID', determines which Tracker class will be used for tracking
        :param benchmark: (bool) determines whether the track will perform a test on the MOT benchmark

        ---- attributes ---
        detections (list) - relevant for 'benchmark' mode, data structure for holding all the detections from file
        frame_count (int) - relevant for 'benchmark' mode, frame counter, used for indexing and looping through frames
        """
        if tracker == 'Kalman': self.tracker = KalmanTracker()
        elif tracker == 'ORB': self.tracker = ORBTracker()
        elif tracker == 'ReID': self.tracker = ReIDTracker()

        self.benchmark = benchmark
        if src is not None:
            # stara wersja jakby mulithreading cos zepsul
            # co sie nieuchronnie stanie
            # self.src = cv2.VideoCapture(src)
            self.src = WebcamVideoStream(src=src).start()
        self.detector = None

        if self.benchmark:
            SORT.check_data_path()
            self.sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof']
            """
            More sequences:
            'ETH-Sunnyday', 'ETH-Pedcross2', 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2'
            """
            self.seq_idx = None
            self.load_next_seq()

        else:
            if detector == 'faster-rcnn':
                model_path = './faster_rcnn_inception_v2/frozen_inference_graph.pb'

                # komentarz coby mi latwiej znalezc bylo bo kinda wazna rzecz
                self.score_threshold = 0.9  # threshold for box score from neural network

            self.detector = DetectorAPI(path_to_ckpt=model_path)
            self.start_tracking()

    def load_next_seq(self):
        """
        When switching sequence - propagate the sequence index and reset the frame count
        Load pre-made detections for .txt file (from MOT benchmark). Starts tracking on next sequence
        """
        if self.seq_idx == len(self.sequences) - 1:
            print('SORT finished going over all the input sequences... closing tracker')
            return

        # Load detection from next sequence and reset the frame count for it
        if self.seq_idx is None:
            self.seq_idx = 0
        else:
            self.seq_idx += 1
        self.frame_count = 1

        # Load detections for new sequence
        file_path = 'data/%s/det.txt' % self.sequences[self.seq_idx]
        self.detections = np.loadtxt(file_path, delimiter=',')

        # reset the tracker and start tracking on new sequence
        self.tracker.reset()
        self.start_tracking()

    def next_frame(self):
        """
        Method for handling the correct way to fetch the next frame according to the 'src' or
         'benchmark' attribute applied
        :return: (np.ndarray) next frame, (np.ndarray) detections for that frame
        """
        if self.benchmark:
            frame = SORT.show_source(self.sequences[self.seq_idx], self.frame_count)
            new_detections = self.detections[self.detections[:, 0] == self.frame_count, 2:7]
            new_detections[:, 2:4] += new_detections[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            self.frame_count += 1
            return frame, new_detections[:, :4]

        else:
            #  nie mam pojceia co robi to _, i sie tego legitnie boje
            # _, frame = self.src.read()

            # ############################################################## #
            # NAJWYZSZEJ WAZNOSCI wyrzuca FATAL: exception not rethrown      #
            # po zakonczeniu pracy. nie mam zielonego pojecia co to robi     #
            # ale wydaje sie dzialac jak powinno - mozliwe ze tylko wizualne #
            # mozliwe ze nie, diabli i Absolut raczy wiedziec                #
            # ############################################################## #
            frame = self.src.read()

            # po ciezki chuj na 1280x720
            # frame = cv2.resize(frame, (1280, 720))
            boxes, scores, classes, num = self.detector.processFrame(frame)
            # supress boxes with scores lower than threshold
            # non maxima suppression good stuff
            boxes_nms = []
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > self.score_threshold:    # Class 1 represents person
                    boxes_nms.append(boxes[i])
            return frame, boxes_nms

    def start_tracking(self):
        """
        Main driver method for the SORT class, starts tracking detections from source.
        Receives list of associated detections for each frame from its tracker (Kalman or ORB),
        Shows the frame with color specific bounding boxes surrounding each unique track.
        """
        while True:
            # Fetch the next frame from video source, if no frames are fetched, stop loop
            frame, detections = self.next_frame()
            if frame is None:
                break

            # Send new detections to set tracker
            if isinstance(self.tracker, KalmanTracker):
                tracks = self.tracker.update(detections)
            elif isinstance(self.tracker, ORBTracker) or isinstance(self.tracker, ReIDTracker):
                tracks = self.tracker.update(frame, detections)
            else:
                raise Exception('[ERROR] Tracker type not specified for SORT')

            # Look through each track and display it on frame (each track is a tuple (ID, [x1,y1,x2,y2])
            try:
                for ID, bbox in tracks:
                    bbox = self.verify_bbox_format(bbox)
                    # Generate pseudo-random colors for bounding boxes for each unique ID
                    random.seed(ID)

                    # Make sure the colors are strong and bright and draw the bounding box around the track
                    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                    color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
                    startX, startY, endX, endY = bbox
                    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)

                    framef = cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), 2)
                    rgb_frame = framef[:, :, ::-1]  # na 95% to jest zbedne bo to juz jest w BGR ale chuj wi, wiec zostawiam
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    face_names = []

                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                        face_names.append(name)
                    kurwa = Tracker(ID, face_names)
                    if kurwa.tracking() is not None:
                        # check1 = any(tname in sublist for sublist in listakurwa for tname in listakurwa)
                        # if check1 is True:
                        listakurwa = set(kurwa.tracking())
                        print(listakurwa)


                    # Calculate centroid from bbox, display it and its unique ID
                    centroid = bbox_to_centroid(bbox)
                    text = "ID {}".format(ID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)


                # Show tracked frame
                cv2.imshow("Video Feed", frame)
            except TypeError:
                pass
            # iqf the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('SORT operation terminated by user... closing tracker')
                return
        if self.benchmark:
            self.load_next_seq()

    def verify_bbox_format(self, bbox):
        """
        Fixes bounding box format according to video type (e.g. benchmark test or video capture)
        :param bbox: (array) list of bounding boxes
        :return: (array) reformatted bounding box
        """
        if self.benchmark:
            return bbox.astype("int")
        else:
            bbox.astype("int")
            return [bbox[1], bbox[0], bbox[3], bbox[2]]


    @staticmethod
    def show_source(seq, frame, phase='train'):
        """ Method for displaying the origin video being tracked """
        return cv2.imread('mot_benchmark/%s/%s/img1/%06d.jpg' % (phase, seq, frame))

    @staticmethod
    def check_data_path():
        """ Validates correct implementation of symbolic link to data for SORT """
        if not os.path.exists('mot_benchmark'):
            print('''
            ERROR: mot_benchmark link not found!\n
            Create a symbolic link to the MOT benchmark\n
            (https://motchallenge.net/data/2D_MOT_2015/#download)
            ''')
            exit()


def main():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    path_to_video = 0
    mot_tracker = SORT(path_to_video)


if __name__ == '__main__':
    main()
