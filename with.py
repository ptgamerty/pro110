# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
model=tf.keras.models.load_model("keras_model.h5")  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    if status:
        frame=cv2.flip(frame,1)
        resized_frame=cv2.resize(frame,(224,224))
        resized_frame=np.expand_dims(resized_frame,axis=0)    
    
        resized_frame=resized_frame/255.0
        prediction=model.predict(resized_frame)
        rock=int(prediction[0][0]*100)
        scissor=int(prediction[0][2]*100)
        paper=int(prediction[0][1]*100)
        print(f"rock:{rock}%,paper:{paper}%,scissor:{scissor}%")
  
    # Display the resulting frame
        cv2.imshow('frame', frame)

    # Quit window with spacebar
        key = cv2.waitKey(1)
    
        if key == 32:
            break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()