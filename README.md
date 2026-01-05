# AIProcessingTask
This is a github repo for a simple task of loading an image from MNIST data available in keras dataset. This data is then processed and trained to predict the class of that image. We then display the prediction result using OpenCV.  

## Dependencies (For Desktop / Laptop)
1. Install Python 3.7 or above
2. Install required dependencies: Run the following code in terminal of your IDE.
   2.1 pip install tensorflow numpy opencv-python
3. Save the script file.
4. Open the code in IDE and open the terminal.
5. Run by the following command in terminal
   5.1 python test.py

NOTE: The file name is test.py. If you rename the file then running command will be python filename.py

## Run on Google Colab
1. Open Colab in browser
2. Upload `test.py`
3. If opencv not available write the following code
   3.1 !pip install opencv-python
4. Add the following line in import section
   4.1 from google.colab.patches import cv2_imshow
5. Replace the following lines of test.py
  5.1 Lines to be replaced
      cv2.imshow("Sample Image", img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
   5.2 Line which will replace all of the above in google colab
      cv2_imshow(img)
6. Run the cell containing script
