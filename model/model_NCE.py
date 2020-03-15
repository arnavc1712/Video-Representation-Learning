import torch
import torch.nn as nn



class Model(nn.Module):

	def __init__(self,dim_model = 768, dim_lang = 768, dim_vid = 1024):
		super().__init__()


		self.vid_enc = nn.Linear(dim_vid,dim_model)
		# self.lang_enc = nn.Linear(dim_lang,dim_model)

		## Feature Concatenation

		self.cat = nn.Linear(2*dim_model,dim_model)

		## NCE label logits

		self.nce = nn.Linear(3*dim_model,1)


	def feat_concat(self,vis_feats,lang_feats):

		add_feats = vis_feats + lang_feats
		mul_feats = vis_feats * lang_feats

		concat_feats = self.cat(torch.cat((vis_feats,lang_feats),dim=-1))

		## Fuse all features
		fused_feats = torch.cat((add_feats,mul_feats,concat_feats),dim=-1)

		return fused_feats


	def forward(self,vis_feats, pos_language_feat, neg_language_feat):

		## vis_feats  (batch_size x dim_vid)
		## pos_language_feat (batch_size x dim_lang)
		## neg_language_feat (num_neg_samples x dim_lang)

		vis_feats = self.vid_enc(vis_feats)

		# pos_language_feat = self.lang_enc(pos_language_feat)

		# neg_language_feat = self.lang_enc(neg_language_feat.view(-1,neg_language_feat.shape[-1]))

		## For positive language pairs
		pos_feats = self.nce(self.feat_concat(vis_feats,pos_language_feat))

		pos_nce_probs = torch.exp(pos_feats)

		## For negative language pairs

		vis = vis_feats.unsqueeze(1).repeat(1,neg_language_feat.shape[0],1)
		neg_language_feat = neg_language_feat.unsqueeze(0).repeat(vis_feats.shape[0],1,1)



		neg_feats = self.nce(self.feat_concat(vis,neg_language_feat))
		neg_nce_probs = torch.exp(neg_feats)

		pos_nce = (pos_nce_probs / (pos_nce_probs + neg_nce_probs.sum(dim=1)))
		neg_nce = (neg_nce_probs / (pos_nce_probs + neg_nce_probs.sum(dim=1)).unsqueeze(1))

		# print(pos_nce.size())
		# print(neg_nce.size())

		return pos_nce,neg_nce




