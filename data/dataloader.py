
import torch
from torch.utils.data import Dataset
import pickle
import json
import os
import opts
import numpy as np

opt = opts.parse_opt()
opt = vars(opt)

class NCEDataset(Dataset):

	def __init__(self,opt,mode):
		super(NCEDataset, self).__init__()
		self.video_feats = pickle.load(open(opt["s3d_feat_path"],"rb"))

		self.cap_feats = pickle.load(open(opt["bert_feats_path"],"rb"))

		info = json.load(open(opt["info_json"]))
		self.ix_to_word = info['ix_to_word']
		self.word_to_ix = info['word_to_ix']
		self.splits = info['videos']

		self.splits['train'] = list(filter(lambda x: x in self.video_feats.keys(),self.splits['train']))
		self.splits['val'] = list(filter(lambda x: x in self.video_feats.keys(),self.splits['val']))
		self.splits['test'] = list(filter(lambda x: x in self.video_feats.keys(),self.splits['test']))

		self.train  = self.splits['train']
		self.test = self.splits['test']
		self.val = self.splits['val']
		self.mode = mode



	def __getitem__(self,ix):
		## Get the video featues as well as the caption features
		video_id = self.splits[self.mode][ix]

		vis_feats = self.video_feats[video_id]

		cap_feats = self.cap_feats[video_id]

		data = {}

		data['vis_feats'] = torch.from_numpy(vis_feats).type(torch.FloatTensor)
		data['cap_feats'] = torch.from_numpy(cap_feats).type(torch.FloatTensor)

		return data

	def __len__(self):
		return len(self.splits[self.mode])


def nce_collate(batch_lst):
	global opt


	batch_lens = [_['vis_feats'].shape[0] for _ in batch_lst]
	max_video_len = max(batch_lens)

	vis_feats = np.zeros((len(batch_lens),opt['dim_vid']))
	cap_feats = np.zeros((len(batch_lens),768))

	for batch_id,batch_data in enumerate(batch_lst):
		s3d_feats = torch.mean(batch_data['vis_feats'],0)
		vis_feats[batch_id] = s3d_feats
		cap_feats[batch_id] = batch_data['cap_feats']

	vis_feats = torch.from_numpy(vis_feats).type(torch.FloatTensor)
	pos_lang_feats = torch.from_numpy(cap_feats).type(torch.FloatTensor) ## (batch_size, dim_lang)
	neg_idx = np.random.random_integers(0,len(batch_lens)-1,opt['num_neg_samples'])
	neg_lang_feats = pos_lang_feats[neg_idx] ## For each video in each batch, the same negative samples will be returned
	## neg_lang_feats (num_neg_samples,dim_lang)

	return vis_feats, pos_lang_feats, neg_lang_feats









		





