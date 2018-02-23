from svr_train import *

classifier_dir = '/home/mondejar/markers_end2end/data_64/'
databaseFilename = '/home/mondejar/markers_end2end/data_64/train_data_list.txt'
dirBase = '/home/mondejar/markers_end2end/'

# Test over the same train data

# One regressor per output-value
use_z_score = True
svm_models = []

for m in range(0,8):
    print("Loading regressor: " + str(m+1) + '/8 ...')

    if use_z_score:
        svm_name = classifier_dir + 'svm_model_' + str(m) + '_Z_C_0.1_ep_1.joblib.pkl'
    else:
        svm_name = classifier_dir + 'svm_model_' + str(m) + '_C_0.1_ep_1.joblib.pkl'

    # Load svm regressors
    svm_models.append(joblib.load(svm_name))

# Load images and shows the outputs and ground-truth 
marker_size = 64
X_data, Y_data = load_data(databaseFilename,  marker_size, 0)

# Apply some preprocessing
if use_z_score:
    scaler = StandardScaler()
    scaler.fit(X_data)
    X_data = scaler.transform(X_data)

num_instances = 10
fileDB = open(databaseFilename,'r') 
content = fileDB.readlines()
fileDB.close()


# Let's test the first 10 images
for x in range(0, num_instances):
    line_splitted = content[x].split()
    img_path = line_splitted[0]
    
    # Convert matrix-images to a single array
    img = cv2.imread(dirBase + img_path, 0)

    predicted_data = np.zeros((8))
    for m in range(0,8):
        predicted_data[m] = svm_models[m].predict([X_data[x]])
    # Represent the corners output over the input image
    
    print("Corners GT: ")
    print(Y_data[x])
    print("Corners predicted: ")
    print(predicted_data)
    display_corners_on_marker(img, predicted_data)
    