from flask import Flask
from flask import request
import os
import webpredict

app = Flask(__name__)

@app.route('/')
def hello_world():
	return app.send_static_file('demo.html')

@app.route('/predict',methods=['POST'])
def predictFromImg():
	if request.method == 'POST':
		predictImg = request.files['predictImg']

		predictImg.save(os.path.join("/home/chao/tf/MNIST/tmp",predictImg.filename))

		imgurl = '/home/chao/tf/MNIST/tmp/'+predictImg.filename
		result = webpredict.img2class(imgurl)
		print(result)
		return "<ch1>%s</h1>" % result

if __name__ == '__main__':
	app.run(host='127.0.0.1',port=8000)

