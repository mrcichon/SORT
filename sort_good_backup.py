# File "/home/mrc/Desktop/SORT-danny_opencv/tracker.py", line 13: ważny argument, max frames na zniknięcie.
# Powinno być chyba nastawione albo na 0 klatek, albo na ~10? Jakoś nisko chyba w każdym razie

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
from collections import OrderedDict
import threading


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

# nie no bez jaj chyba widać co się dzieje
face_locations = []
face_encodings = []
face_names = []

# tying
class associate:

    def __init__(self):
        plox = OrderedDict()
        cokolwiek = []
        listaID = set()
        self.plox = plox
        self.ploxiterate = self.plox.copy()
        self.cokolwiek = cokolwiek
        self.listaID = listaID
        self.sendhelp =[]
        self.keystodel = []

    def counter(self, tracks):
        self.tracks = tracks
        self.plox[len(self.tracks)] = None
        return self.plox, self.tracks

    def associating(self, face_names, ID):
        self.ID = ID
        self.face_names = face_names
        self.listaID.add(self.ID)
        for name in self.face_names:
            if name is not None:
                if name not in self.cokolwiek:
                    self.cokolwiek.append(name)
        self.sendhelp = list(zip(self.listaID, self.cokolwiek))
        return self.sendhelp, self.listaID, self.cokolwiek

    def make(self):
        for key in self.plox.keys():
            if key <= len(self.sendhelp) - 1:
                self.plox[key] = self.sendhelp[key]
                self.ploxiterate = self.plox.copy()
            else:
                pass
        return self.plox

    def delnone(self):
        for key, value in self.ploxiterate.items():
            if value is None:
                del self.plox[key]
        return self.plox

    def clean(self):
        self.listaIDCOPY = self.listaID.copy()
        self.IDintracks = [x[0] for x in self.tracks]
        for identificator in self.listaIDCOPY:
            if identificator not in self.IDintracks:
                self.listaID.remove(identificator)
        for element in self.sendhelp:
            listcheck = element[0]
            if listcheck not in self.IDintracks:
                self.sendhelp.remove(element)
                self.cokolwiek.remove(element[1])
        return self.plox, self.sendhelp, self.listaID, self.cokolwiek

    def check(self):
        print(self.plox)


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

                # komentarz coby mi łatwiej znaleźć było, bo kinda ważna rzecz
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
            #  nie mam pojęcia, co robi to _, i się tego legitnie boje
            # _, frame = self.src.read()

            # ############################################################### #
            # Wyrzuca FATAL: exception not rethrown                           #
            # po zakończeniu pracy. Nie mam zielonego pojęcia, co to robi,    #
            # ale wydaje się działać jak powinno — możliwe, że tylko wizualne #
            # możliwe, że nie, diabli i Absolut raczy wiedzieć                #
            # ############################################################### #
            frame = self.src.read()

            # po ciężki chuj na 1280x720
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
        a = associate()
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

            a.counter(tracks)

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
                    rgb_frame = framef[:, :, ::-1]  # na 95% to jest zbędne, bo to już jest w BGR, ale chuj wi, wiec zostawiam
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

                    a.associating(face_names, ID)
                    a.make()
                    # a.delnone()
                    a.clean()
                    a.check()


                        # for location in face_locations:
                        #     x = location[0]
                        #     y = location[1]
                        #     w = location[2]
                        #     h = location[3]
                        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=4)
                        #     if int(startX) < x and int(startY) < y:
                        #         if x + w < int(startX) + int(endX) \
                        #                 and y + h < int(startY) + int(endY):
                        #             print("nie no ciekawie ale co w związku z tym")
                        #         else:
                        #             print("ratunkuuuuuuuuuu")



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
    SORT(path_to_video)


if __name__ == '__main__':
    main()
