# SORT

This repository is [danbochman's](https://github.com/danbochman) implementation of [SORT](https://github.com/abewley/sort) hacked together with [facial recognition](https://github.com/ageitgey/face_recognition) by [ageitgey](https://github.com/ageitgey/). It's made out of duct tape and bubblegum. It also associates each recognised face with a given ID and checks for presence of a given ID within a marked segment of the field of view. It employs an incredibly simplistic way of measuring distance from camera to each bounding box, and also allows for checking if a given face detected within a selected segment is a set distance away. The repository also has a terrible GUI packed along.

Everything was written with python 3.6 in mind.

The main file is, of course, sort.py


