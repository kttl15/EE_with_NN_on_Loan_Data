import pandas as pd
import numpy as np
import time
import pydot
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Concatenate, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import plot_model
from keras import regularizers
from keras.optimizers import RMSprop


def data_gen():
	# Read csv file.
	filename = 'train.csv'
	df = pd.read_csv(filename)

	'''
	Columns to drop from dataset. These columns were chosen partly due to:

	(a): not providing any, and in some cases detrimental, contributions to the outcome.
	(b): having upwards of thousands of unique values.
	'''

	col_to_drop = ['UniqueID','Current_pincode_ID','MobileNo_Avl_Flag',
	               'Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag',
	               'Passport_flag','Date.of.Birth',
	               'DisbursalDate','Employee_code_ID','Employment.Type',
	               'CREDIT.HISTORY.LENGTH','AVERAGE.ACCT.AGE']
	df.drop(col_to_drop, axis=1, inplace=True)


	'''
	Columns to embed. These columns contain categorical data as opposed to numeric data.
	Data within these columns will go through an embedding layer.
	'''

	col_to_emb = ['branch_id','supplier_id','manufacturer_id',
	              'State_ID','PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS',
	              'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','SEC.NO.OF.ACCTS',
	              'SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','NEW.ACCTS.IN.LAST.SIX.MONTHS',
	              'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES']

	'''
	Now we create two dataframes that will hold the categorical data and numerical data.
	This is because we will be using the StandardScalar function to normalise the numerical data.

	df_to_emb: dataframe that holds the categorical data
	df_not_emb: dataframe that holds the numerical data
	'''

	df_to_emb = df[col_to_emb]
	df_not_emb = df.drop(col_to_emb, axis=1)
	df_not_emb.drop('loan_default', axis=1)

	# Normalising the numeric data
	df_not_emb = pd.DataFrame(StandardScaler().fit_transform(df_not_emb), columns=df_not_emb.columns).astype(np.float32)

	# Now we will create our labels and featureset
	label = df['loan_default']
	featureset = pd.concat([df_not_emb, df_to_emb], axis=1)

	return featureset, label, df_to_emb


def vectorise(X):
	'''
	Converts input X, with shape (m, n_x), into (n_x, m)
	'''
    
    input_list = []

    col_to_emb = ['branch_id','supplier_id','manufacturer_id',
	              'State_ID','PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS',
	              'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','SEC.NO.OF.ACCTS',
	              'SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','NEW.ACCTS.IN.LAST.SIX.MONTHS',
	              'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES']
    # Appending categorical data to list.
    for col in col_to_emb:
        raw_vals = np.unique(X[col])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list.append(X[col].map(val_map).values)
    
    # Appending numerical data to list.
    for col in X.columns:
        if not col in col_to_emb:
            input_list.append(X[col].values)
    
    return input_list


def build_model(X_train, df_to_emb, l2_reg, neurons):
	'''
	Build NN
	'''
	col_to_emb = ['branch_id','supplier_id','manufacturer_id',
	              'State_ID','PERFORM_CNS.SCORE.DESCRIPTION','PRI.NO.OF.ACCTS',
	              'PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','SEC.NO.OF.ACCTS',
	              'SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','NEW.ACCTS.IN.LAST.SIX.MONTHS',
	              'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS','NO.OF_INQUIRIES']

	l2_reg = l2_reg
	neurons = neurons

	input_models = []
	output_embeddings = []

	for categorical_var in col_to_emb:
	    cat_emb_name = categorical_var.replace(' ', '') + '_Embedding'
	    no_of_unique_cat = df_to_emb[categorical_var].nunique()
	    if no_of_unique_cat <= 4:
	        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50))
	    else:
	        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50))
	    input_name = 'Input_' + categorical_var.replace(' ', '')
	    
	    input_model = Input(shape=(1,))
	    output_model = Embedding(no_of_unique_cat, embedding_size, name=cat_emb_name)(input_model)
	    output_model = Reshape(target_shape=(embedding_size,))(output_model)

	    input_models.append(input_model)
	    output_embeddings.append(output_model)

	for numeric_var in X_train.columns:
	    if not numeric_var in df_to_emb:
	        
	        input_numeric = Input(shape=(1,))
	        embedding_numeric = Dense(1, name=numeric_var)(input_numeric)

	        input_models.append(input_numeric)
	        output_embeddings.append(embedding_numeric)

	output = Concatenate()(output_embeddings)
	output = Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(output)
	output = Dropout(0.5)(output)
	# output = Dense(neurons, activation='relu')(output)
	# output = Dropout(0.2)(output)
	# output = Dense(neurons, activation='relu')(output)
	# output = Dropout(0.2)(output)
	output = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(output)
	output = Dropout(0.2)(output)

	output = Dense(1, activation='sigmoid')(output)

	model = Model(inputs=input_models, outputs=output)

	opt = RMSprop()
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse','mape','accuracy'])

	return model


def prediction(model, X):
	# Get y_pred
	return np.round(model.predict(X))


def get_comp_graph(model, filename):
	# Generates an image of the computation graph using keras.utils.plot_model and pydot.
	plot_model(model, show_shapes=True, show_layer_names=True, to_file=filename)


def plot_cm(y_val, y_pred):
	# Plots confusion matrix.
	cm = confusion_matrix(y_val, y_pred)
	fig, ax = plt.subplots(figsize=(8,8))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.set(xticks = range(cm.shape[1]),
	      yticks = range(cm.shape[0]),
	      ylabel = 'True label',
	      xlabel = 'Predicted label')

	# Printing number of samples for each case in the middle of their corresponding box.
	fmt = '.0f'
	thresh = cm.max()/2.
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        ax.text(j,i, format(cm[i,j], fmt), ha='center', va='center', 
	                color='white' if cm[i,j] > thresh else 'black')

	fig.tight_layout()
	plt.show()


def get_score(y_val, y_pred):
	# Calculates the precision and recall scores.
	precision = precision_score(y_val, y_pred)
	print('Precision is: ', precision)

	recall = recall_score(y_val, y_pred)
	print('Recal is: ', recall)


featureset, label, df_to_emb = data_gen()

X_train, X_test, y_train, y_test = train_test_split(featureset, label, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.01)

X_train_list = vectorise(X_train)
X_test_list = vectorise(X_test)
X_val_list = vectorise(X_val)



l2_reg = 0.01
neurons = 16
batch_size = 128

model = build_model(X_train, df_to_emb, l2_reg, neurons)

name = f'{neurons}x10-HL-{l2_reg}-l2_reg-{batch_size}-BS-TIME-({int(time.time())})'
date = '21-5-19'

# Various tensorflow callbacks
tb = TensorBoard(log_dir=f'logs_{date}/{name}')
ms = ModelCheckpoint(f'best_model-{name}.hdf5', monitor='val_loss', save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', patience=10)
red_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=2)

history = model.fit(X_train_list, y_train, 
                    validation_data=(X_test_list, y_test), 
                    epochs=30, batch_size=batch_size, 
                    verbose=2, callbacks=[red_lr])


# Creates an image of the computation graph and saves it as 'model.png'
# get_comp_graph(model, filename='model.png')

# Get y_pred.
y_pred = prediction(model, X_val_list)

# Plots confusion matrix.
plot_cm(y_val, y_pred)

# Prints precision and recall scores.
get_score(y_val, y_pred)




