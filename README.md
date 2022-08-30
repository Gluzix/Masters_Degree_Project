# Usage of the application

## Creating frames from video:

### Format:
```
python main.py framesfromvideo --path_to_movie=[path to movie] --count_gap=[gap count] --max_count=[max count]
```

### Arguments:
- **path_to_movie** - argument which takes the path to the movie from which the frames will be read
- **count_gap** - this value stores gap between read frames, it uses modulo operator on the frames which are read
- **max_count** - maximum number of frames to read

### Example of use:
```
python main.py framesfromvideo --path_to_movie="E:/Projekt Magisterski/resources/videos/angela_merkel_speech.mp4" --count_gap=4 --max_count=100
```

## Create source and target folders:

### Format:
```
python main.py landmarkscreate --path_to_images=[path to image] --path_to_predictor=[path to predictor] --path_to_haarcascade=[path to haarcascade]
```
### Arguments:
- **path_to_images** - the path to the images folder which stores all of the frames read from video
- **path_to_predictor** - path to the predictor. Predictor is the file with the .dat extension
- **path_to_haarcascade** - path to the haarcascade frontalface file. It's for the face recognition and it's extension is .xml

### Example of use:
```
python main.py landmarkscreate --path_to_images="E:/Projekt Magisterski/images" --path_to_predictor="E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat" --path_to_haarcascade="E:/Projekt Magisterski/resources/pre_trained_data/haarcascade_frontalface_default.xml" 
```

## Test face detection algorithm:

### Format:
```
python main.py testfacedetection --path_to_image=[path to image] --path_to_haarcascade==[path to haarcascade] 
```

### Arguments:
- **path_to_image** - path to the one image for testing the face detection algorithm. Image should contain face.
- **path_to_haarcascade** - path to the haarcascade frontalface file. It's for the face recognition and it's extension is .xml

### Example of use:
```
python main.py testfacedetection --path_to_image="E:/Projekt Magisterski/images/img13.png" --path_to_haarcascade=="E:/Projekt Magisterski/resources/pre_trained_data/haarcascade_frontalface_default.xml" 
```

## Test face detection algorithm:

### Format:
```
python main.py testlandmarksdetection --landmark_algorithm=[type of algorithm for landmark detection] OPTIONAL (Only for dlib) --path_to_predictor==[path to predictor]
```

### Arguments:
- **landmark_algorithm** - type of algorithm. dlib and mediapipe are supported
- **path_to_predictor** - Path to the predictor. Only for the dlib algorithm

### Example of use:
```
python main.py testlandmarksdetection --landmark_algorithm=mediapipe
python main.py testlandmarksdetection --landmark_algorithm=dlib --path_to_predictor="E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat"
```

## Predicting image from the webcam:
### Format:
```
python main.py pix2pix --pix2pix_action=webcam --path_to_haarcascade=[path to haarcascade] --path_to_model=[path to model] --path_to_predictor=[path to predictor]
```
### Arguments:
- **path_to_haarcascade** - path to the haarcascade frontalface file. It's for the face recognition and it's extension is .xml
- **path_to_model** - path to the trained model
- **path_to_predictor** - path to the predictor. Predictor is the file with the .dat extension

### Example of use:
```
python main.py pix2pix --pix2pix_action=webcam --path_to_haarcascade="E:/Projekt Magisterski/resources/pre_trained_data/haarcascade_frontalface_default.xml" --path_to_model="E:/Projekt Magisterski/resources/datasets_and_ready_models/duda_big_dataset_batch_10_around_4000_samples/model_192000.h5" --path_to_predictor="E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat"
```

## Creating dataset:

### Format:
```
python main.py pix2pix --pix2pix_action=create --path_to_src=[path to source] --path_to_tar=[path to target] --filename=[output filename]
```

### Arguments:
- **path_to_src** - path to the folder which stores source images (landmarks)
- **path_to_tar** - path to the folder which stores real images
- **filename** - name of the output dataset file. The best is to type "name.npz" with .npz extension

### Example of use:
```
python main.py pix2pix --pix2pix_action=create --path_to_src=src --path_to_tar=tar --filename=test_256.npz
```

## Plotting dataset:

### Format:
```
python main.py pix2pix --pix2pix_action=plot --path_to_dataset=[path to dataset]
```

### Arguments:
- **path_to_dataset** - path to the dataset

### Example of use:
```
python main.py pix2pix --pix2pix_action=plot --path_to_dataset=test_256.npz
```
## Predicting random image from dataset:

### Format:
```
python main.py pix2pix --pix2pix_action=predict --path_to_dataset=[path to dataset] --path_to_model=[path to model]
```

### Arguments:
- **path_to_dataset** - path to the dataset
- **path_to_model** - path to the trained model

### Example of use:
```
python main.py pix2pix --pix2pix_action=predict --path_to_dataset=resources/old_datasets/maps_256_kamil_smaller_landmarks_new_validation_samples.npz --path_to_model=resources/datasets_and_ready_models/duda_big_dataset_batch_10_around_4000_samples/model_192000.h5
```