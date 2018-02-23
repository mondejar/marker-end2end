import time
from sklearn.svm import SVR
from draw_corners_on_marker import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

classifier_dir = '/home/mondejar/markers_end2end/data_64/'
databaseFilename = '/home/mondejar/markers_end2end/data_64/train_data_list.txt'
dirBase = '/home/mondejar/markers_end2end/'

use_z_score = True
def load_data(filename, marker_size, verbose):

    fileDB = open(filename,'r') 
    content = fileDB.readlines()
    fileDB.close()

    # def X_data, Y_data = load_data()
    X_data = np.zeros((len(content), marker_size * marker_size))
    Y_data = np.zeros((len(content), 8))

    for x in range(0, len(content)):
        line_splitted = content[x].split()

        img_path = line_splitted[0]
        
        # Convert matrix-images to a single array
        img = cv2.imread(dirBase + img_path, 0)
        img_1D = img.flatten() # 128 * 128 = 16384
        X_data[x] = img_1D

        for y in range(1,9):
            Y_data[x, y-1] = float(line_splitted[y])

        # Verbose
        if verbose > 1:
            print("Corners: ")
            print(Y_data[x])
            display_corners_on_marker(img, Y_data[x])

    if verbose > 0: 
        print("Data loaded!")
        print("X data:")
        print(X_data.shape)
        print("Y data:")
        print(Y_data.shape)

    return X_data, Y_data


def main():
    # Load data and ground truth
    marker_size = 64
    X_data, Y_data = load_data(databaseFilename,  marker_size, 0)

    # TODO 
    # Apply some preprocessing
    if use_z_score:
        scaler = StandardScaler()
        scaler.fit(X_data)
        X_data = scaler.transform(X_data)
        # scaled: zero mean unit variance ( z-score )

    # Train SVM regression
    print("Training.... please wait")

    # One regressor per output-value
    for m in range(0,8):
        print("Training regressor: " + str(m+1) + '/8 ...')

        start = time.time()
        # SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
        #    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
        svm_model = SVR(C=0.1, epsilon=0.5)
        svm_model.fit(X_data, Y_data[:,m]) 

        # Export model
        if use_z_score:
            svm_name = classifier_dir + 'svm_model_' + str(m) + '_Z_C_0.1_ep_0.5.joblib.pkl'
        else:
            svm_name = classifier_dir + 'svm_model_' + str(m) + '_C_0.1_ep_0.5.joblib.pkl'

        joblib.dump(svm_model, svm_name)
        
        end = time.time()
        print("Time training Regressor: " + str(format(end - start, '.2f')) + " sec" )

if __name__ == "__main__":
    main()
