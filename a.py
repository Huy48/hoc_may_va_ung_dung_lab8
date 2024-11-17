from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

def train_knn():
	data = load_wine()
	X, y = data.data, data.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(X_train, y_train)

	y_pred = knn.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred, average='macro')
	precision = precision_score(y_test, y_pred, average='macro')

	return accuracy, recall, precision
