Note that c3d-pretrained.pth is trained on Kinetics, it correlates to the C3D_model.py,
and Unsupervised_Trained_C3D.pth is trained by clip ordering, and it correlates to c3d.py.

They share almost the same structure however with some tweaks in final FC layers.

c3d-pretrained.pth: Kinetic pretrained c3d.
s3d_nce_pretrained: use NCE train S3D net. output 768d.
Unsupervised_Trained_C3D.pth: save as c3d.
