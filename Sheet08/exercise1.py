import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import random
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

"""
reconstructing_image fuction: 
    to reconstruct the image with the principal components(eigenfaces)
input: the image path, the mean of the training , the principal components (eigenfaces), the width of the training data images, the height of the training data images
output: the reconstruction error

"""

def reconstructing_image(image_path,avg_face,principal_components,training_data_matrix_width,training_data_matrix_height):
    # fetch the image
    an_image = cv2.imread(image_path+".jpg", 0)
    # resize it to be as dimensional as the the images in the trainging set
    an_image_resized = cv2.resize(an_image, (training_data_matrix_width,training_data_matrix_height), interpolation = cv2.INTER_AREA)
    # flatten the image to be a vector
    an_image_flattened = np.reshape(an_image_resized,(training_data_matrix_height*training_data_matrix_width,))
    # normalizing the image (by removing the mean of the training data from it)
    an_image_normalized =  an_image_flattened - avg_face
    # computing the scalar of the eigen vectors(eigenfaces) that will be used to reconstruct the image
    scalars_w = np.dot(principal_components,an_image_normalized)
    # reconstructing the image
    reconstructed_image = avg_face + np.dot(np.transpose(principal_components),scalars_w)
    # computing the reconstruction error
    return np.linalg.norm(an_image_flattened - reconstructed_image)


def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Compute the PCA
    pca_model = PCA(n_components=100, whiten=True , svd_solver='randomized')

    fit_data = pca_model.fit(X_train)
    
    principal_components = fit_data.components_

    # Visualize Eigen Faces
    eigenfaces = np.reshape(principal_components,(100,h,w))
    
    fig = plt.figure(figsize=(10,10))

    for i in range (10):
        fig.add_subplot(3, 4, i+1)
        plt.imshow(eigenfaces[i],cmap="gray")
        plt.title("eigenface " + str(i+1), size=12)
    plt.show()

    # compute the avg face to use it in normalizing the image vector
    avg_face = np.mean(X_train,axis=0)

    # Compute reconstruction error
    reconstruction_error = reconstructing_image("./data/exercise1/detect/face/boris",avg_face,principal_components,w,h)
    print("reconstruction error of boris's photo = ", reconstruction_error)

    # Perform face detection
    main_folders = ["/face","/other"]
    samples_files = ["/boris","/merkel","/obama","/putin","/trump","/cat","/dog","/flag","/flower","/monkey"]
    
    total_number_detection_samples = 10
    correct_detection = 0
    expected_value=0
    
    for i in range(2):
        for j in range (5):
            reconstruction_error = reconstructing_image("./data/exercise1/detect"+main_folders[i]+samples_files[(i*5)+j],avg_face,principal_components,w,h)
            
            if ( reconstruction_error < 860 ):
                expected_value = 0
            else:
                expected_value = 1
                
            if ( expected_value == i):
                correct_detection += 1
    
    print ("accuracy of the detection algorithm = ", correct_detection/total_number_detection_samples*100,"%")
    
    # Perform face recognition
    
    #intializing the classifier that will recognize the test data
    classifier = KNeighborsClassifier(n_neighbors=6, weights='distance')
    # projecting the X train data to the eigenfaces space
    projected_x_train = pca_model.transform(X_train)
    # make the classifier fit the train data
    classifier=classifier.fit(projected_x_train, y_train)
    # projecting the X test data to the eigenfaces space
    projected_x_test = pca_model.transform(X_test)
    # predicating the test data
    expected_values = classifier.predict(projected_x_test)
    # calculating the correct predications(classified data)
    correct_classified = np.sum(expected_values == y_test)
    print("accuracy of classification algorithm= ", correct_classified/y_test.shape[0] * 100, "%")
    
if __name__ == '__main__':
    main()
