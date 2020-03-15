import os
import random
import numpy as np
from utils.utils import *
import torch.optim as optim
# import pytorch_warmup as warmup
import opts
# from utils.opts_C3D_videos import *
from torch.utils.data import DataLoader
from model.model_cosine import Model
from data.dataloader import NCEDataset, nce_collate

def train(loader, model, optimizer, opt):
    model.train()
    step = 0
    for epoch in range(opt['epochs']):

        for (vis_feats, pos_lang_feats, neg_lang_feats) in loader:
            torch.cuda.synchronize()

            vis_feats = vis_feats.cuda()

            # Stack input as pos-neg paris
            pos_language_feat = pos_lang_feats.cuda()
            neg_language_feat = neg_lang_feats.cuda()

            optimizer.zero_grad()

            # Early break current epoch if videos are not properly loaded
            if vis_feats.shape[0] != opt['batch_size']:
                print('loading problem.')
                continue

            # Model Inference, feed in labels for data parallel concatenation
            stacked_vis_feats,stacked_lang_feats,no_pos_samples,no_neg_samples = model(vis_feats, pos_language_feat, neg_language_feat)



            cosine_labels = generate_cosine_labels(no_pos_samples,no_neg_samples)

            # Cosine Loss
            cos_loss = torch.nn.CosineEmbeddingLoss(reduction='sum')

            loss = cos_loss(stacked_vis_feats,stacked_lang_feats,cosine_labels)

            # loss = -torch.log(pos_nce_prob).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)
            optimizer.step()

            

            # update parameters
            print_loss = loss.item()
            torch.cuda.synchronize()

            step += 1
            print('(epoch %d), loss = %.6f, current lr = %.5E'
                  % (epoch, print_loss, optimizer.param_groups[0]['lr']))

        if epoch % opt['save_checkpoint_every'] == 0:
            model_path = os.path.join(opt['checkpoint_path'], 'NCE_model%d.pth' % epoch)
            model_info_path = os.path.join(opt['checkpoint_path'], 'nce_scores.txt')
            torch.save(model.state_dict(), model_path)

            print('model saved to %s' % model_path)
            with open(model_info_path, 'a') as f:
                f.write('model_%d, nce_loss: %.6f' % (epoch, loss))

        if epoch == 100:
            """Sets the learning rate to the initial LR decayed by 10 at 100 epochs"""
            lr = opt['lr'] * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def main(opt):
    # mode = training|validation
    dataset = NCEDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True, collate_fn=nce_collate)

    model = Model(dim_lang = opt['dim_lang'],dim_vid=opt['dim_vid'], dim_model=opt['dim_model'])

    model = nn.DataParallel(model).cuda()
    model.apply(weight_init)

    # if opt['load_checkpoint']:
    #     state_dict = torch.load(opt['load_checkpoint'])
    #     model.load_state_dict(state_dict)
    #     print('Check-point Loaded.')

    opt['lr'] = (1e-5)*5
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=5e-4)
    train(dataloader, model, optimizer, opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)