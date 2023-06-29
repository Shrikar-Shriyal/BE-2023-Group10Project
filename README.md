# BE-2023-Group10Project
This is our BE project named "Conversion of sign language to text using mediapipe and lstm"

## Introduction

Sign languages are languages that uses the visual-manual modality to express meaning, instead of just spoken words. It means of communication through bodily movements, especially of the hands and arms, when spoken communication is impossible or not desirable. Sign Language is the methods of communication of deaf and dumb people all over the world. Sign Language to Text Conversion System is a benefit for helping deaf and dumb persons to communicate with others. The paper proposes a method for an application which would help to convert the Sign language into the text format. With the help of neural networks and computer vision we can detect hand gestures and give the respective output. We have come up with a system which uses several methods such as using Media Pipe Holistic
model, collect Key point values for training and testing, pre-processing data Create labels and features, build and train LSTM neural network.

## Features
* Real-time hand tracking using MediaPipe library
* LSTM-based gesture recognition model
* Web interface for visualizing the recognized gestures as text
* Supports Indian sign language gestures
* Easy integration with other projects or systems

## Requirements
to run this project, you need the following:

* Python 3.7 or above
* MediaPipe library (mediapipe)
* TensorFlow (tensorflow 2.12.0)
* OpenCV (opencv-python '4.7.0')
* Django (django 3.2)

## Installation
1. Clone the project repository:
  ```bash
   [git clone https://github.com/your-username/sign-language-conversion.git](https://github.com/Shrikar-Shriyal/BE-2023-Group10Project.git)
  ```

2. Change into the project directory:
  ```bash
   cd mysite
  ```

3. Install the required Python dependencies:
   ```bash
     pip install -r requirements.txt
    ```

## Usage
1. Run the Django development server:
   ```bash
     python manage.py runserver
    ```
2. Open a web browser and navigate to http://localhost:8000 to access the application.
3. Allow access to your webcam when prompted.
4. Click Detect and Perform sign language gestures in front of the webcam, and the recognized gestures will be displayed as text on the web interface.
5. Press q to quit the popup camera.
6. Press Ctrl+C in the terminal to stop the server when you're done.

## Authors
Akash Kamble, Rahul Chalavde, Rahul Dalvi, Shrikar Shriyal

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request to the project repository.

## License
The project is licensed under the MIT License. Feel free to modify and use the code according to the terms of the license.
