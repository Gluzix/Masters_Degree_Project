import argparse
from argparse import ArgumentTypeError
from Pix2Pix import pix_2_pix_runner
from FramesFromVideo import frames_from_video_maker
from FaceLandmarkDetection import dlib_detector, media_pipe_detector
from FaceDetection import face_detector
from CreateLandmarksFromImages import landmark_images_creator

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    my_parser.add_argument('action', type=str, help='action argument which allows to choose what action'
                                                    'should be performed')

    my_parser.add_argument('--path_to_dataset', nargs='?', type=str, help='path to dataset', default='')
    my_parser.add_argument('--path_to_model', nargs='?', type=str, help='path to model', default='')

    my_parser.add_argument('--path_to_predictor', nargs='?', type=str, help='path to predictor .dat file', default='')
    my_parser.add_argument('--path_to_haarcascade', nargs='?', type=str, help='path to haarcascade .xml file', default='')

    my_parser.add_argument('--path_to_src', nargs='?', type=str, help='path to landmarks images folder', default='')
    my_parser.add_argument('--path_to_tar', nargs='?', type=str, help='path to real images folder', default='')

    my_parser.add_argument('--path_to_images', nargs='?', type=str, help='path to images folder', default='')
    my_parser.add_argument('--path_to_image', nargs='?', type=str, help='path to image for face prediction', default='')

    my_parser.add_argument('--landmark_algorithm', nargs='?', type=str, help='dlib/mediapipe for testing purposes', default='')

    my_parser.add_argument('--path_to_movie', nargs='?', type=str, help='path to movie for creating separate frames', default='')
    my_parser.add_argument('--count_gap', nargs='?', type=int, help='count gap between read frames', default=1)
    my_parser.add_argument('--max_count', nargs='?', type=int, help='maximum count of taken frames', default=1)

    my_parser.add_argument('--pix2pix_action', nargs='?', type=str, help='action for pix2pix: \n'
                                                                         'webcam \n'
                                                                         'create \n'
                                                                         'train \n'
                                                                         'predict \n'
                                                                         'plot', default='')

    my_parser.add_argument('--filename', nargs='?', type=str, help='filename of the created dataset', default='')

    args = my_parser.parse_args()

    if args.action == 'landmarkscreate':
        if args.path_to_images and args.path_to_predictor and args.path_to_haarcascade:
            landmark_images_creator.LandmarkImagesCreator.create_landmarks(args.path_to_images,
                                                                           args.path_to_predictor,
                                                                           args.path_to_haarcascade)
        else:
            raise ArgumentTypeError("missing 'path_to_images', 'path_to_predictor' or 'path_to_haarcascade' arguments")
    elif args.action == 'testfacedetection':
        if args.path_to_image and args.path_to_haarcascade:
            face_detector.FaceDetector.detect_face(args.path_to_image, args.path_to_haarcascade)
        else:
            raise ArgumentTypeError("missing 'path_to_image', 'path_to_haarcascade'")
    elif args.action == 'testlandmarksdetection':
        if args.landmark_algorithm == 'dlib':
            if args.path_to_predictor:
                dlib_detector.DlibDetector.detect_landmarks(args.path_to_predictor)
            else:
                raise ArgumentTypeError("dlib needs predictor path arg!")
        elif args.landmark_algorithm == 'mediapipe':
            media_pipe_detector.MediaPipeDetector.detect()
        else:
            raise ArgumentTypeError("missing 'landmark_algorithm'")
    elif args.action == 'framesfromvideo':
        if args.path_to_movie and args.count_gap and args.max_count:
            frames_from_video_maker.FramesFromVideoMaker.make_frames(args.path_to_movie,
                                                                     args.count_gap,
                                                                     args.max_count)
        else:
            raise ArgumentTypeError("missing 'path_to_movie', 'count_gap' or 'max_count' arguments")
    elif args.action == 'pix2pix':
        if args.pix2pix_action == 'webcam':
            if args.path_to_haarcascade and args.path_to_model and args.path_to_predictor:
                pix_2_pix_runner.Pix2PixRunner.predict_from_webcam(args.path_to_predictor,
                                                                   args.path_to_model,
                                                                   args.path_to_haarcascade)
            else:
                raise ArgumentTypeError("missing 'path_to_haarcascade', 'path_to_model' or 'path_to_predictor' arguments")
        elif args.pix2pix_action == 'create':
            if args.path_to_src and args.path_to_tar and args.filename:
                pix_2_pix_runner.Pix2PixRunner.create_dataset(args.path_to_src, args.path_to_tar, args.filename)
            else:
                raise ArgumentTypeError("missing 'path_to_src', 'path_to_tar' or 'filename' arguments")
        elif args.pix2pix_action == 'train':
            if args.path_to_dataset:
                pix_2_pix_runner.Pix2PixRunner.train(args.path_to_dataset)
            else:
                raise ArgumentTypeError("missing 'path_to_dataset' argument")
        elif args.pix2pix_action == 'predict':
            if args.path_to_dataset and args.path_to_model:
                pix_2_pix_runner.Pix2PixRunner.try_to_predict(args.path_to_dataset, args.path_to_model)
            else:
                raise ArgumentTypeError("missing 'path_to_dataset' or 'path_to_model' arguments")
        elif args.pix2pix_action == 'plot':
            if args.path_to_dataset:
                pix_2_pix_runner.Pix2PixRunner.plot_images(args.path_to_dataset)
            else:
                raise ArgumentTypeError("missing 'path_to_dataset' argument")
        else:
            raise ArgumentTypeError("missing 'pix2pix_action' argument")
    else:
        raise ArgumentTypeError("Wrong 'action' argument")
