import remnist
import reader

def img2class(imgFile):
	img = reader.pre_img(imgFile)
	res=remnist.restore_model_ckpt(img)
	return res
