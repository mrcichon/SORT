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
import sys
import glob
from collections import OrderedDict
from threading import Thread
import PySimpleGUI as sg
import datetime
import  shutil
import mss

sct =mss.mss()

monitor = {"top": 0, "left": 0, "width": 1366, "height": 768}

#  intializing shite for face_recognition
known_face_names = []
known_face_encodings = []
face_locations = []
face_encodings = []
face_names = []

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


class ScreenFeed:
    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.hope, args=())
        t.daemon = True
        t.start()
        return self

    def hope(self):
        while True:
            self.x = np.array(sct.grab(monitor))

    def read(self):
        return self.x


class Presence:
    @staticmethod
    def select_area(first_frame):
        area = cv2.selectROI("zaznacz drzwi", first_frame)
        save_crds = open("koordynaty" , "w")
        save_crds.write(str(area[0]) + "\n" + str(area[1]) + "\n" + str(area[2]) + "\n" + str(area[3]))
        save_crds.close()
        return int(area[0]), int(area[1]), int(area[2]), int(area[3])

    @staticmethod
    def close():
        cv2.destroyWindow("zaznacz drzwi")

class Distance:
    def __init__(self):
        self.known_distance = 100.0
        self.known_width = 14.0

    def focal_distance(self):
        self.img = cv2.imread("mensura.jpg")
        # self.resize = cv2.resize(self.img, (940, 560))
        self.ref = cv2.selectROI("distance", self.img)
        self.ref_width = int(self.ref[2])
        self.focalLength = (self.ref_width * self.known_distance) / self.known_width
        return self.focalLength

    def skip(self, focalLength):
        self.focalLength = focalLength
        return self.focalLength

    def distance_to_camera(self,  bbox):
        distance = (self.known_width * self.focalLength) / (int(bbox[2]) - int(bbox[0]))
        return distance

    @staticmethod
    def close():
        cv2.destroyWindow("distance")

# tying
class Associate:
    def __init__(self):
        plox = OrderedDict()
        cokolwiek = []
        listaID = set()
        self.plox = plox
        self.present_list = []
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
                    if name != "Unknown":
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
        try:
            for key, value in self.ploxiterate.items():
                if value is None:
                    del self.plox[key]
        except KeyError:
            pass
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

    def check_frq(self, present_id):
        for key, value in  self.plox.items():
            for ID in present_id:
                if ID in value and self.plox[key] not in self.present_list:
                    self.present_list.append(self.plox[key])
        return self.present_list

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

        screen = ScreenFeed()

        self.benchmark = benchmark
        # self.src = screen.start()
        if src is not None:
            # stara wersja jakby multithreading cos zepsuł, co sie nieuchronnie stanie
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
            frame = self.src.read()
            boxes, scores, classes, num = self.detector.processFrame(frame)
            # supress boxes with scores lower than threshold
            boxes_nms = []
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > self.score_threshold:    # Class 1 represents person
                    boxes_nms.append(boxes[i])
            return frame, boxes_nms


    def face_rec(self, frame, startX, startY,endX, endY):
        framef = cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), 2)
        rgb_frame = framef[:, :, ::-1]  # na 95% to jest zbędne, bo to już jest w BGR, ale chuj wi, wiec zostawiam
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        self.face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            self.face_names.append(name)
            return self.face_names

    def start_tracking(self):
        """
        Main driver method for the SORT class, starts tracking detections from source.
        Receives list of associated detections for each frame from its tracker (Kalman or ORB),
        Shows the frame with color specific bounding boxes surrounding each unique track.
        """
        associate = Associate()
        distance = Distance()
        frame, detections = self.next_frame()

        layout = [[sg.Button("skip focal length", key="-SKIP_FOCAL-"), sg.Button("choose doors", key="-DOORS-")],
                  [sg.Button("retake focus length", key="-GET_FOCAL-"), sg.Button("skip doors", key="-SKIP_DOORS-")],
                  [sg.Text("distance"), sg.Input()], # value[0]
                  [sg.Text("focal length") ,sg.Input()], # value[1]
                  [sg.Combo(['podgląd', 'speed'])], # value[2]
                  [sg.Submit(key="-SEND-"), sg.Cancel()]]
        window = sg.Window('menu2', layout)
        while True:
            event, values = window.read()
            if event == "-SKIP_FOCAL-":
                distance.skip(int(values[1]))
            if event == "-GET_FOCAL-":
                print(distance.focal_distance())
                distance.close()
            if event == "-DOORS-":
                coordinates = Presence.select_area(frame)
                Presence.close()
            if event == "-SKIP_DOORS-":
                save_crds = open("koordynaty", "r")
                coordinates = save_crds.readlines()
            if event == sg.WIN_CLOSED or event == 'Exit' or event =="-SEND-":
                break
        window.close()

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

            associate.counter(tracks)

            # Look through each track and display it on frame (each track is a tuple (ID, [x1,y1,x2,y2])
            try:
                for ID, bbox in tracks:
                    bbox = self.verify_bbox_format(bbox)
                    # Generate pseudo-random colors for bounding boxes for each unique ID
                    random.seed(ID)

                    bbox_distance = distance.distance_to_camera(bbox)

                    # Make sure the colors are strong and bright and draw the bounding box around the track
                    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                    color = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
                    startX, startY, endX, endY = bbox
                    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)

                    self.face_rec(frame,startX, startY, endX, endY)
                    associate.associating(self.face_names, ID)
                    associate.make()
                    associate.clean()
                    associate.delnone()
                    associate.check()

                    # Calculate centroid from bbox, display it and its unique ID
                    centroid = bbox_to_centroid(bbox)
                    text = "ID {}".format(ID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, color, -1)

                    present =[]
                    if startX >= int(coordinates[0]) and startY >= int(coordinates[1]) and \
                            endX <= int(coordinates[0]) + int(coordinates[2]) and \
                            endY <= int(coordinates[1]) + int(coordinates[3]) and bbox_distance > int(values[0]):
                        present.append(ID)
                    real_present_popup = [associate.check_frq(present)]
                cv2.imshow("Video Feed", frame)

            except TypeError:
                pass
            # iqf the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('SORT operation terminated by user... closing tracker')
                sg.popup(real_present_popup, no_titlebar=True)
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

def main():
    """ Starts the tracker on source video. Can start multiple instances of SORT in parallel """
    # path_to_video = "http://192.168.1.39:4747/video"
    path_to_video = 0
    SORT(path_to_video)

if __name__ == '__main__':
    main()
