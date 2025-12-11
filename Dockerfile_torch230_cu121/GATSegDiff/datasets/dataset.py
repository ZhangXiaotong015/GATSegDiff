import torch
import nibabel
# import monai
import numpy as np
from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
import math
import re
import cv2


class Dataset3D_diffusionGAT_sample_LiVS(torch.utils.data.Dataset): # full CT volume as the input
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        num_layers = 3
        num_rows = 32
        num_cols = 32
        nodes_coord = []
        for z in range(num_layers):
            for y in range(0,256,int(256/num_rows)):
                for x in range(0,256,int(256/num_cols)):
                    center_x = x + (1 // 2)
                    center_y = y + (1 // 2)
                    center_z = z + (1 // 2)
                    nodes_coord.append((center_x, center_y, center_z))

        edges = []
        num_nodes = num_rows*num_cols*num_layers
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if math.sqrt((nodes_coord[j][0] - nodes_coord[i][0])**2 + (nodes_coord[j][1] - nodes_coord[i][1])**2 + (nodes_coord[j][2] - nodes_coord[i][2])**2) ==1 \
                or math.sqrt((nodes_coord[j][0] - nodes_coord[i][0])**2 + (nodes_coord[j][1] - nodes_coord[i][1])**2 + (nodes_coord[j][2] - nodes_coord[i][2])**2) ==int(256/num_rows):
                    edges.append([i,j])
        self.edge_index = np.array(edges).transpose(1,0)
        'NOTICE!!! Must define the nodes_coord as float type.'
        nodes_coord = np.array(nodes_coord, dtype=float)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (256-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (256-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (num_layers-1)) -1 # coord_z
        self.nodes_coord = nodes_coord

        '# Create a 3D plot'
        # # import os
        # # from PIL import Image
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_facecolor("none") 
        # # Plot the points
        # ax.scatter(nodes_coord[:, 0], nodes_coord[:, 1], nodes_coord[:, 2], color='red', s=10, alpha=1.0)
        # # Plot the edges
        # for edge in edges:
        #     ax.plot(nodes_coord[edge, 0], nodes_coord[edge, 1], nodes_coord[edge, 2], color='blue', alpha=1.0)
        # # Set plot properties
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('Edges between 3D Points')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        # vessel_tree = sitk.ReadImage(self.list_IDs['vessel'][index])
        '# transpose axis 3 and axis 2 to make it consistent with the data used in training!!!'
        ct = sitk.GetArrayFromImage(ct).transpose(0,1,3,2) # (sliceNum,3,256,256)
        # vessel = sitk.GetArrayFromImage(vessel_tree).transpose(0,1,3,2)

        ct_slice_num_total = ct.shape[0]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        # label = vessel
        # label[label>0] = 1

        ct25d_list = []
        # label_list = []
        for i in range(ct.shape[0]):
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            # label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        # label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        # return tuple((ct25d, label, nodes_coord_, edge_index_, label_slice_idx, self.list_IDs['ct'][index], origin, space, direction, ct_slice_num_total))
        return tuple((ct25d, nodes_coord_, edge_index_, self.list_IDs['ct'][index], origin, space, direction))
