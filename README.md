# Basic-Face-Recognition
Created a basic face recognition model, which uses OpenCV to see the faces and if the faces are stored in the database, then the model can use k-Nearest Neighbors algorithm to identify whose face it is and prompts their name.

Initally, one must create a folder called Data, to save all the faces of the people in the database.
Then, one must run the data_collection.py script to capture the faces and save the details of the faces as .npy files in Data folder.

Later, when the person wants to run face recogntion, he/she can simply run the face_recognition.py script to recognize the face(if it is available in the Data folder).
Recognizing the face is done using kNN machine learning algorithm and capturing the facial features and face detection is done using HaarCascade.xml file and OpenCV.
