# import cv2
# def camerafeed():
#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()

#         # Display the frame
#         cv2.imshow('Webcam', frame)

#         # Exit the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close the OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()