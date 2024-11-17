import matplotlib.pyplot as plt
import io
import base64

from a import train_knn
from flask import Flask, render_template


accuracy, recall, precision = train_knn()

app = Flask(__name__)

@app.route('/')
def index():
	metrics = {"Accuracy": accuracy, "Recall": recall, "Precision": precision}
	plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
	plt.title("Model Performance Metrics")
	plt.ylim(0, 1)

	img = io.BytesIO()
	plt.savefig(img, format='png')
	img.seek(0)
	plot_url = base64.b64encode(img.getvalue()).decode()
	plt.close()

	return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
	app.run(debug=True)
