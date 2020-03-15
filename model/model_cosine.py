import torch
import torch.nn as nn



class Model(nn.Module):

	def __init__(self,dim_model = 768, dim_lang = 768, dim_vid = 1024):
		super().__init__()


		self.vid_enc = nn.Linear(dim_vid,dim_model)

		self.lang_enc = nn.Linear(dim_lang,dim_model)

		self.dim_model = dim_model
		self.dim_lang = dim_lang
		self.dim_vid = dim_vid



	def forward(self,vis_feats, pos_language_feat, neg_language_feat):

		## vis_feats  (batch_size x dim_vid)
		## pos_language_feat (batch_size x dim_lang)
		## neg_language_feat (num_neg_samples x dim_lang)

		pos_vis_feats = self.vid_enc(vis_feats)

		pos_language_feat = self.lang_enc(pos_language_feat)

		neg_language_feat = self.lang_enc(neg_language_feat.view(-1,neg_language_feat.shape[-1]))


		neg_language_feat = neg_language_feat.unsqueeze(0).repeat(vis_feats.shape[0],1,1) ##(batch_size,num_neg_samples,dim_model)

		pos_vis_feats_rep = pos_vis_feats.unsqueeze(1).repeat(1,neg_language_feat.shape[1],1)

		stacked_vis_feats = torch.cat((pos_vis_feats,pos_vis_feats_rep.view(-1,self.dim_model)),dim=0)

		stacked_lang_feats = torch.cat((pos_language_feat,neg_language_feat.view(-1,self.dim_model)),dim=0)

		
		no_pos_samples = pos_vis_feats.shape[0]
		no_neg_samples = neg_language_feat.shape[0]*neg_language_feat.shape[1]

		return stacked_vis_feats,stacked_lang_feats,no_pos_samples,no_neg_samples




