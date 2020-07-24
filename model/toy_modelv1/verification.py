import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

import argparse
from model import vggvox_model
from wav_reader import get_fft_spectrum
import constants as c


def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets

def get_embeddings_from_file(model, file_path, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.DataFrame({'filename':[file_path]})
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
# 	result['speaker'] = 19
	return result[['filename','embedding']]


def verify(opt):
	input_wav_path = opt.input
	test_wav_path = opt.test
	metric_fn = opt.metric
	threshold = opt.threshold


	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_file(model, input_wav_path, c.MAX_SEC)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
# 	speakers = enroll_result['speaker']

	print("Processing test samples....")
	test_result = get_embeddings_from_file(model, test_wav_path, c.MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=metric_fn), columns=['distance'])

# 	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.DataFrame({'input':[input_wav_path.split('/')[-1]],'test':[test_wav_path.split('/')[-1]]})
	scores['metric'] = metric_fn
	scores = pd.concat([scores,distances],axis =1 )
	scores['threshold'] = threshold
	scores['result'] = scores['distance'] < threshold
# 	scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int
	print(scores)
	print("Writing outputs to [{}]....".format(c.RESULT_FILE))
	result_dir = os.path.dirname(c.RESULT_FILE)
	if not os.path.exists(result_dir):
	    os.makedirs(result_dir)
	with open(c.RESULT_FILE, 'w') as f:
		scores.to_csv(f, index=False)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	parser_scoring = subparsers.add_parser('verify')
	parser_scoring.add_argument('--input')
	parser_scoring.add_argument('--test')
	parser_scoring.add_argument('--metric', default = 'cosine')
	parser_scoring.add_argument('--threshold', default = 0.1)
	parser_scoring.set_defaults(func=verify)
	opt = parser.parse_args()
	opt.func(opt)
