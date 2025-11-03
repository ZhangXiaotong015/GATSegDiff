import torch
import nibabel
import monai
import numpy as np
from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
import math
import re
import cv2

class Dataset3D_diffusion_train(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 300

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['hepatic'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        # filtered_ct = nibabel.load(self.list_IDs['filtered_ct'][index]).get_fdata().transpose(2,0,1)
        hepatic = nibabel.load(self.list_IDs['hepatic'][index]).get_fdata().transpose(2,0,1)
        portal = nibabel.load(self.list_IDs['portal'][index]).get_fdata().transpose(2,0,1)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)
        # filtered_ct = (filtered_ct-filtered_ct.min()) / (filtered_ct.max()-filtered_ct.min()+1e-7)

        label = hepatic + portal
        label[label>0] = 1

        # ct_25d = torch.cat((torch.from_numpy(ct)[None].type(torch.float32), torch.from_numpy(filtered_ct)[None].type(torch.float32)), dim=1)
        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        # ct = ct_25d[:,3,].unsqueeze(1)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,3,].unsqueeze(1)

        return tuple((ct_25d, label))

class Dataset3D_diffusion_sample(torch.utils.data.Dataset): # full CT volume as the input
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 300

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['hepatic'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        # ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata()
        # hepatic = nibabel.load(self.list_IDs['hepatic'][index]).get_fdata()
        # portal = nibabel.load(self.list_IDs['portal'][index]).get_fdata()
        # liver = nibabel.load(self.list_IDs['liver'][index]).get_fdata()

        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        filtered_ct = sitk.ReadImage(self.list_IDs['filtered_ct'][index])
        hepatic_tree = sitk.ReadImage(self.list_IDs['hepatic'][index])
        portal_tree = sitk.ReadImage(self.list_IDs['portal'][index])
        liver_mask = sitk.ReadImage(self.list_IDs['liver'][index])

        ct = sitk.GetArrayFromImage(ct) # (sliceNum,7,256,256)
        filtered_ct = sitk.GetArrayFromImage(filtered_ct)
        hepatic = sitk.GetArrayFromImage(hepatic_tree)
        portal = sitk.GetArrayFromImage(portal_tree)
        liver = sitk.GetArrayFromImage(liver_mask)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)
        filtered_ct = (filtered_ct-filtered_ct.min()) / (filtered_ct.max()-filtered_ct.min()+1e-7)

        label = hepatic + portal
        label[label>0] = 1
        liver[liver>0] = 1

        ct = ct*liver
        filtered_ct = filtered_ct*liver
        label = label*liver

        # ct_list = []
        ct25d_list = []
        filtered_ct25d_list = []
        label_list = []
        liver_list = []
        for i in range(ct.shape[0]):
            # ct_list.append(torch.from_numpy(ct[i,3,:,:])[None,None]) # (1,1,256,256)
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            filtered_ct25d_list.append(torch.from_numpy(filtered_ct[i,:,:,:])[None])
            label_list.append(torch.from_numpy(label[i,3,:,:])[None,None]) # (1,1,256,256)
            liver_list.append(torch.from_numpy(liver[i,3,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        # ct25d = torch.cat((torch.cat(ct25d_list, dim=0).type(torch.float32), torch.cat(filtered_ct25d_list, dim=0).type(torch.float32)), dim=1)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        liver = torch.cat(liver_list, dim=0).type(torch.float32)

        return tuple((ct25d, label, liver, self.list_IDs['ct'][index], origin, space, direction))


class Dataset3D_diffusionGAT_train_liver_example(torch.utils.data.Dataset): #  LUMCData
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        vessel = nibabel.load(self.list_IDs['vessel'][index]).get_fdata().transpose(2,0,1)
        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()
        path = self.list_IDs['ct'][index].split('/')[-2] + self.list_IDs['ct'][index].split('/')[-1]

        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = vessel
        label[label>0] = 1

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        # ct = ct_25d[:,3,].unsqueeze(1)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        return tuple((ct_25d, label, nodes_coord, edge_index, speed, path))

class Dataset3D_diffusionGAT_sample_liver_example(torch.utils.data.Dataset): # full CT volume as the input
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        self.label_name = 'manualgrow_LiverCorrectedemergedPrePortals'
        self.label_name_2 = 'manualgrow_LiverdemergedPrePortals'
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'

        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        liver_mask = sitk.ReadImage(self.list_IDs['ct'][index].replace('orig.nii.gz', 'liver.nii.gz'))

        patient_idx = re.findall(r"\d+",self.list_IDs['ct'][index].split('/')[-1])[-1]
        try:
            manualgrow_label = sitk.ReadImage(self.list_IDs['ct'][index].replace(self.list_IDs['ct'][index].split('/')[-1], self.label_name+patient_idx+'.nii.gz'))
        except:
            manualgrow_label = sitk.ReadImage(self.list_IDs['ct'][index].replace(self.list_IDs['ct'][index].split('/')[-1], self.label_name_2+patient_idx+'.nii.gz'))

        ct = sitk.GetArrayFromImage(ct) # (sliceNum,7,256,256)
        liver = sitk.GetArrayFromImage(liver_mask)
        label = sitk.GetArrayFromImage(manualgrow_label)

        # if 'Arterial' in self.list_IDs['ct'][index]:
        #     clip_min = 0
        #     clip_max = 700
        # elif 'Portal' in self.list_IDs['ct'][index]:
            # clip_min = 0
            # clip_max = 300

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        liver[liver>0] = 1
        label[label>0] = 1

        ct = ct*liver

        ct25d_list = []
        liver_list = []
        label_list = []
        for i in range(ct.shape[0]):
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,3,256,256)
            liver_list.append(torch.from_numpy(liver[i,1,:,:])[None,None]) # (1,1,256,256)
            label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)

        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        liver = torch.cat(liver_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)

        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        return tuple((ct25d, label, nodes_coord_, edge_index_, liver, self.list_IDs['ct'][index], origin, space, direction))


class Dataset3D_diffusionGAT_train(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400
        # for negative blocks near edges
        # num_layers = 3
        # num_rows = 32
        # num_cols = 32
        # nodes_coord = []
        # for z in range(num_layers):
        #     for y in range(0,256,int(256/num_rows)):
        #         for x in range(0,256,int(256/num_cols)):
        #             center_x = x + (1 // 2)
        #             center_y = y + (1 // 2)
        #             center_z = z + (1 // 2)
        #             nodes_coord.append((center_x, center_y, center_z))

        # edges = []
        # num_nodes = num_rows*num_cols*num_layers
        # for i in range(num_nodes):
        #     for j in range(i + 1, num_nodes):
        #         if math.sqrt((nodes_coord[j][0] - nodes_coord[i][0])**2 + (nodes_coord[j][1] - nodes_coord[i][1])**2 + (nodes_coord[j][2] - nodes_coord[i][2])**2) ==1:
        #             edges.append([i,j])
        # self.edge_index = np.array(edges).transpose(1,0)
        # nodes_coord = np.array(nodes_coord)
        # nodes_coord[:,0] = 2*(nodes_coord[:,0] / (256-1)) -1 # coord_x
        # nodes_coord[:,1] = 2*(nodes_coord[:,1] / (256-1)) -1 # coord_y
        # nodes_coord[:,2] = 2*(nodes_coord[:,2] / (num_layers-1)) -1 # coord_z
        # self.nodes_coord = nodes_coord
        # self.num_layers = num_layers
        # self.num_rows = num_rows
        # self.num_cols = num_cols

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        hepatic = nibabel.load(self.list_IDs['hepatic'][index]).get_fdata().transpose(2,0,1)
        portal = nibabel.load(self.list_IDs['portal'][index]).get_fdata().transpose(2,0,1)
        path = self.list_IDs['ct'][index].split('/')[-1]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = hepatic + portal
        label[label>0] = 1

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        # ct = ct_25d[:,3,].unsqueeze(1)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        # if 'Edge' not in path:
        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()
    
        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        # elif 'Edge' in path:
        #     nodes_coord = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        #     nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        #     edge_index = torch.from_numpy(self.edge_index).type(torch.int32)
        #     speed = torch.from_numpy(np.zeros((self.num_layers,self.num_rows,self.num_cols)))[None].type(torch.float32)

        return tuple((ct_25d, label, nodes_coord, edge_index, speed, path))
        
class Dataset3D_diffusionGAT_sample(torch.utils.data.Dataset): # full CT volume as the input
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

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['hepatic'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        # ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata()
        # hepatic = nibabel.load(self.list_IDs['hepatic'][index]).get_fdata()
        # portal = nibabel.load(self.list_IDs['portal'][index]).get_fdata()
        # liver = nibabel.load(self.list_IDs['liver'][index]).get_fdata()

        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        hepatic_tree = sitk.ReadImage(self.list_IDs['hepatic'][index])
        portal_tree = sitk.ReadImage(self.list_IDs['portal'][index])
        liver_mask = sitk.ReadImage(self.list_IDs['liver'][index])
        
        '# transpose axis 3 and axis 2 to make it consistent with the data used in training!!!'
        ct = sitk.GetArrayFromImage(ct).transpose(0,1,3,2) # (sliceNum,7,256,256)
        hepatic = sitk.GetArrayFromImage(hepatic_tree).transpose(0,1,3,2)
        portal = sitk.GetArrayFromImage(portal_tree).transpose(0,1,3,2)
        liver = sitk.GetArrayFromImage(liver_mask).transpose(0,1,3,2)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = hepatic + portal
        label[label>0] = 1
        liver[liver>0] = 1

        # ct = ct*liver
        label = label*liver

        # ct_list = []
        ct25d_list = []
        label_list = []
        liver_list = []
        for i in range(ct.shape[0]):
            # ct_list.append(torch.from_numpy(ct[i,3,:,:])[None,None]) # (1,1,256,256)
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)
            liver_list.append(torch.from_numpy(liver[i,1,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        liver = torch.cat(liver_list, dim=0).type(torch.float32)
        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        return tuple((ct25d, label, nodes_coord_, edge_index_, liver, self.list_IDs['ct'][index], origin, space, direction))
        # return tuple((ct25d, label, liver, self.list_IDs['ct'][index], origin, space, direction))


class Dataset3D_diffusionGAT_train_LiVS(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400 # same as the description in the paper

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['vessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        vessel = nibabel.load(self.list_IDs['vessel'][index]).get_fdata().transpose(2,0,1)
        path = self.list_IDs['ct'][index].split('/')[-1]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        label = vessel
        label[label>0] = 1

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()
    
        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        return tuple((ct_25d, label, nodes_coord, edge_index, speed, path))

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
        return len(self.list_IDs['vessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        vessel_tree = sitk.ReadImage(self.list_IDs['vessel'][index])
        '# transpose axis 3 and axis 2 to make it consistent with the data used in training!!!'
        ct = sitk.GetArrayFromImage(ct).transpose(0,1,3,2) # (sliceNum,3,256,256)
        vessel = sitk.GetArrayFromImage(vessel_tree).transpose(0,1,3,2)

        ct_slice_num_total = ct.shape[0]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        label = vessel
        label[label>0] = 1

        '# only keep slices with real ground truth during the inference'
        # label_slice_idx = np.load(self.list_IDs['label_slice_idx'][index], allow_pickle=True)
        # label_slice_idx = list(label_slice_idx)
        # ct = ct[label_slice_idx,]
        # label = label[label_slice_idx,] 

        # label_slice_idx = list(np.arange(0,ct.shape[0]))

        ct25d_list = []
        label_list = []
        for i in range(ct.shape[0]):
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        # return tuple((ct25d, label, nodes_coord_, edge_index_, label_slice_idx, self.list_IDs['ct'][index], origin, space, direction, ct_slice_num_total))
        return tuple((ct25d, label, nodes_coord_, edge_index_, self.list_IDs['ct'][index], origin, space, direction))

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'cosine':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Dataset3D_diffusionGAT_uncertainty_control_LiVS(torch.utils.data.Dataset): # load training set in the inference phase for the uncertainty estimation.
    def __init__(self, list_IDs, beta_schedule, beta_start, beta_end, num_diffusion_timesteps, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400

        betas = get_beta_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['vessel']) // 50

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        vessel = nibabel.load(self.list_IDs['vessel'][index]).get_fdata().transpose(2,0,1)
        path = self.list_IDs['ct'][index].split('/')[-1]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = vessel
        label[label>0] = 1

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()
    
        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        x = label.float()
        t = torch.randint(low=0, high=self.num_timesteps, size=(1,))[None]
        e = torch.randn_like(x)
        b = self.betas
        a = (1-b).cumprod(dim=0)[t]
        xt = x * a.sqrt() + e * (1.0 - a).sqrt()

        e = torch.flatten(e, start_dim=0, end_dim=-1)[None]

        return tuple((xt, ct_25d, nodes_coord, edge_index, t[0].float(), e))   


class Dataset3D_diffusionGAT_train_LiVS_subset(torch.utils.data.Dataset): # cropped sub volume as output
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 400 # same as the description in the paper

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['vessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        vessel = nibabel.load(self.list_IDs['vessel'][index]).get_fdata().transpose(2,0,1)
        path = self.list_IDs['ct'][index].split('/')[-1]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        label = vessel
        label[label>0] = 1

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()
    
        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        return tuple((ct_25d, label, nodes_coord, edge_index, speed, path))

class Dataset3D_diffusionGAT_sample_LiVS_subset(torch.utils.data.Dataset): # full CT volume as the input
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
        return len(self.list_IDs['vessel'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        vessel_tree = sitk.ReadImage(self.list_IDs['vessel'][index])
        '# transpose axis 3 and axis 2 to make it consistent with the data used in training!!!'
        ct = sitk.GetArrayFromImage(ct).transpose(0,1,3,2) # (sliceNum,3,256,256)
        vessel = sitk.GetArrayFromImage(vessel_tree).transpose(0,1,3,2)

        ct_slice_num_total = ct.shape[0]

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)

        label = vessel
        label[label>0] = 1

        ct25d_list = []
        label_list = []
        for i in range(ct.shape[0]):
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        return tuple((ct25d, label, nodes_coord_, edge_index_, self.list_IDs['ct'][index], origin, space, direction))


class Dataset3D_diffusionGAT_MIXtrain(torch.utils.data.Dataset): # Mix 3DIRCADB to LUMCData
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 300

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        vessel = nibabel.load(self.list_IDs['vessel'][index]).get_fdata().transpose(2,0,1)
        graph = np.load(self.list_IDs['graph'][index], allow_pickle=True).item()

        if 'LUMCData' not in self.list_IDs['ct'][index]:
            path = self.list_IDs['ct'][index].split('/')[-1]
        else:
            path = self.list_IDs['ct'][index].split('/')[-2] + self.list_IDs['ct'][index].split('/')[-1]

        nodes_coord = np.array(graph['nodes']) # (num_nodes,3)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (ct.shape[-1]-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (ct.shape[-2]-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (ct.shape[-3]-1)) -1 # coord_z
        
        edge_index = np.array(graph['edges']).transpose(1,0) # (2,num_edges)
        speed = graph['speed'].transpose(2,1,0) # (3,32,32)->(D,H,W)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = vessel
        label[label>0] = 1

        nodes_coord = torch.from_numpy(nodes_coord)[None].type(torch.float32)
        nodes_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index = torch.from_numpy(edge_index).type(torch.int32)
        speed = torch.from_numpy(speed)[None].type(torch.float32)

        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        # ct = ct_25d[:,3,].unsqueeze(1)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,1,].unsqueeze(1)

        return tuple((ct_25d, label, nodes_coord, edge_index, speed, path))

class Dataset3D_diffusionGAT_MIXsample(torch.utils.data.Dataset): 
    def __init__(self, list_IDs, name=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 300
        self.label_name = 'manualgrow_LiverCorrectedemergedPrePortals'
        self.label_name_2 = 'manualgrow_LiverdemergedPrePortals'
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
                if math.sqrt((nodes_coord[j][0] - nodes_coord[i][0])**2 + (nodes_coord[j][1] - nodes_coord[i][1])**2 + (nodes_coord[j][2] - nodes_coord[i][2])**2) ==1:
                    edges.append([i,j])
        self.edge_index = np.array(edges).transpose(1,0)
        nodes_coord = np.array(nodes_coord)
        nodes_coord[:,0] = 2*(nodes_coord[:,0] / (256-1)) -1 # coord_x
        nodes_coord[:,1] = 2*(nodes_coord[:,1] / (256-1)) -1 # coord_y
        nodes_coord[:,2] = 2*(nodes_coord[:,2] / (num_layers-1)) -1 # coord_z
        self.nodes_coord = nodes_coord

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs['ct'])

    def __getitem__(self, index):
        'load vessel tree volume and pre-processing'
        ct = sitk.ReadImage(self.list_IDs['ct'][index])
        origin = ct.GetOrigin()
        space = ct.GetSpacing()
        direction = ct.GetDirection()

        if 'LUMCData' not in self.list_IDs['ct'][index]:
            vessel_tree = sitk.ReadImage(self.list_IDs['vessel'][index])
            liver_mask = sitk.ReadImage(self.list_IDs['liver'][index])
        else:
            liver_mask = sitk.ReadImage(self.list_IDs['ct'][index].replace('orig.nii.gz', 'liver.nii.gz'))
            patient_idx = re.findall(r"\d+",self.list_IDs['ct'][index].split('/')[-1])[-1]
            try:
                vessel_tree = sitk.ReadImage(self.list_IDs['ct'][index].replace(self.list_IDs['ct'][index].split('/')[-1], self.label_name+patient_idx+'.nii.gz'))
            except:
                vessel_tree = sitk.ReadImage(self.list_IDs['ct'][index].replace(self.list_IDs['ct'][index].split('/')[-1], self.label_name_2+patient_idx+'.nii.gz'))

        ct = sitk.GetArrayFromImage(ct) # (sliceNum,7,256,256)
        vessel = sitk.GetArrayFromImage(vessel_tree)
        liver = sitk.GetArrayFromImage(liver_mask)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)

        label = vessel
        label[label>0] = 1
        liver[liver>0] = 1

        # ct = ct*liver
        label = label*liver

        # ct_list = []
        ct25d_list = []
        label_list = []
        liver_list = []
        for i in range(ct.shape[0]):
            # ct_list.append(torch.from_numpy(ct[i,3,:,:])[None,None]) # (1,1,256,256)
            ct25d_list.append(torch.from_numpy(ct[i,:,:,:])[None]) # (1,7,256,256)
            label_list.append(torch.from_numpy(label[i,1,:,:])[None,None]) # (1,1,256,256)
            liver_list.append(torch.from_numpy(liver[i,1,:,:])[None,None]) # (1,1,256,256)

        # ct = torch.cat(ct_list, dim=0).type(torch.float32)
        ct25d = torch.cat(ct25d_list, dim=0).type(torch.float32)
        label = torch.cat(label_list, dim=0).type(torch.LongTensor)
        liver = torch.cat(liver_list, dim=0).type(torch.float32)
        nodes_coord_ = torch.from_numpy(self.nodes_coord)[None].type(torch.float32)
        nodes_coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        edge_index_ = torch.from_numpy(self.edge_index).type(torch.int32)

        return tuple((ct25d, label, nodes_coord_, edge_index_, liver, self.list_IDs['ct'][index], origin, space, direction))
        # return tuple((ct25d, label, liver, self.list_IDs['ct'][index], origin, space, direction))