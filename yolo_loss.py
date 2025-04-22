import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        #x, y relative to the grid cellw, h of the box
        #x1, y1 top left corner of the box
        # extract the x,y,w,h
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        # convert to x1,y1,x2,y2
        x1 = (x/self.S) - (0.5 * w)
        y1 = (y/self.S) - (0.5 * h)
        x2 = (x/self.S) + (0.5 * w)
        y2 = (y/self.S) + (0.5 * h)

        # create a new tensor to hold the converted boxes
        boxes_new = torch.zeros_like(boxes)
        boxes_new[:, 0] = x1
        boxes_new[:, 1] = y1
        boxes_new[:, 2] = x2
        boxes_new[:, 3] = y2
    
        # return the new boxes
        # the new boxes are in the format [x1, y1, x2, y2]
        return boxes_new                
        

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        # convert the target boxes to the format [x1, y1, x2, y2]
        box_target_new = self.xywh2xyxy(box_target)
        #initialize the best iou and best boxes
        best_iou = None
        best_boxes = None
        # iterate over the pred boxes
        for i in range(len(pred_box_list)):
            pred_box_coord = pred_box_list[i][:, :4]
            pred_box_coord_new = self.xywh2xyxy(pred_box_coord)
            iou = compute_iou(pred_box_coord_new, box_target_new)
            # find the best iou
            iou_max,_ = torch.max(iou, dim=0)
            iou_max = iou_max.unsqueeze(1)

            if best_ious is None:
                best_ious = iou_max
                best_boxes = pred_box_list[i]
            else:
                mask = iou_max > best_ious
                best_ious[mask] = torch.where(mask, iou_max[mask], best_ious)
                mask_2 = mask.expand_as(pred_box_list[i])
        # return the best iou and best boxes
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        has_object_map = has_object_map.unsqueeze(-1)
        # compute the class loss for cells which contain an object
        pred_cls = classes_pred * has_object_map
        target_cls = classes_target * has_object_map
        # compute the class loss
        class_loss = F.mse_loss(pred_cls, target_cls, reduction='sum')
        return class_loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE
        # your code here
        no_object_map = 1 - has_object_map
        # initialize the loss
        no_obj_loss = 0.0
        # iterate over the pred boxes
        for i in range(len(pred_boxes_list)):
            # extract conf scores (5th element)
            box_pred_conf = pred_boxes_list[i][:, :, :, 4]
            no_obj_loss = no_obj_loss + F.mse_loss(box_pred_conf * no_object_map, torch.zeros_like(box_pred_conf), reduction='sum')
             
        return no_obj_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        contain_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')
        return contain_loss
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        #extract the x,y,w,h for the pred and target boxes
        box_pred_x = box_pred_response[:, 0]
        box_pred_y = box_pred_response[:, 1]
        box_pred_w = box_pred_response[:, 2]
        box_pred_h = box_pred_response[:, 3]
        box_target_x = box_target_response[:, 0]
        box_target_y = box_target_response[:, 1]
        box_target_w = box_target_response[:, 2]
        box_target_h = box_target_response[:, 3]
        # compute the regression loss
        loss_xy = F.mse_loss(box_pred_x, box_target_x, reduction='sum') + F.mse_loss(box_pred_y, box_target_y, reduction='sum')
        loss_wh = F.mse_loss(box_pred_w, box_target_w, reduction='sum') + F.mse_loss(box_pred_h, box_target_h, reduction='sum')
        # compute the regression loss
        reg_loss = self.l_coord * (loss_xy + loss_wh)
        ### CODE
        # your code here
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)

        pred_boxes_list = []
        for i in range(self.B):
            pred_boxes_list.append(pred_tensor[:, :, :, i * 5:(i + 1) * 5])
        pred_cls = pred_tensor[:, :, :, self.B * 5:]

        # compcute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)
        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        has_object_map_expanded = has_object_map.unsqueeze(-1).expand_as(target_boxes)
        target_boxes_flat = target_boxes[has_object_map_expanded].view(-1, 4)
        
        # Find best IoU and corresponding boxes
        pred_boxes_flat_list = []
        for i in range(self.B):
            # Expand has_object_map to match pred_boxes dimensions
            has_object_map_expanded_for_pred = has_object_map.unsqueeze(-1).expand_as(pred_boxes_list[i])
            
            # Extract only cells that have objects
            pred_boxes_flat = pred_boxes_list[i][has_object_map_expanded_for_pred].view(-1, 5)
            pred_boxes_flat_list.append(pred_boxes_flat)
        
        # Find the best boxes among the B predicted boxes
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_flat_list, target_boxes_flat)
        
        # Extract confidence and box coordinates from best boxes
        best_boxes_conf = best_boxes[:, 4].unsqueeze(1)
        best_boxes_coords = best_boxes[:, :4]
        
        # Ground truth confidence should be the IoU between best predicted box and target
        target_conf = best_ious
        
        # Compute contain object loss (confidence loss for cells with objects)
        containing_obj_loss = self.get_contain_conf_loss(best_boxes_conf, target_conf)
        
        # Compute regression loss for the best boxes
        reg_loss = self.get_regression_loss(best_boxes_coords, target_boxes_flat)
        
        # Compute the final loss by combining all components
        # Apply the loss weights for coordinates (lambda_coord) and no objects (lambda_noobj)
        total_loss = self.l_coord * reg_loss + containing_obj_loss + self.l_noobj * no_obj_loss + cls_loss
        
        # Construct return loss_dict
        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'containing_obj_loss': containing_obj_loss,
            'no_obj_loss': no_obj_loss,
            'cls_loss': cls_loss,
        }
        
        return loss_dict

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects

        # compute contain_object_loss

        # compute final loss

        # construct return loss_dict
        loss_dict = dict(
            total_loss=...,
            reg_loss=...,
            containing_obj_loss=...,
            no_obj_loss=...,
            cls_loss=...,
        )
        return loss_dict
