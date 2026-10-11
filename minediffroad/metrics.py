import numpy as np
from scipy import ndimage


def postprocess(probability,threshold=.5,min_area=50):
    mask = probability >= threshold
    if min_area > 0:
        labels,count = ndimage.label(mask,structure=np.ones((3,3),dtype=bool))
        areas = np.bincount(labels.ravel(),minlength=count+1)
        keep = areas >= min_area
        keep[0] = False
        mask = keep[labels]
    return mask


def confusion(prediction,target,valid):
    p,y = prediction[valid].astype(bool),target[valid].astype(bool)
    return np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum(),(~p&~y).sum()],dtype=np.int64)


def from_confusion(counts):
    tp,fp,fn,tn = map(int,counts)
    ratio = lambda n,d,empty=1.: n/d if d else empty
    road_iou = ratio(tp,tp+fp+fn)
    return {"road_iou":road_iou,"miou":(road_iou+ratio(tn,tn+fp+fn))/2,
            "dice":ratio(2*tp,2*tp+fp+fn),"precision":ratio(tp,tp+fp,1. if fn==0 else 0.),
            "recall":ratio(tp,tp+fn),"accuracy":ratio(tp+tn,tp+fp+fn+tn),
            "tp":tp,"fp":fp,"fn":fn,"tn":tn}


def boundary_f1(prediction,target,valid,tolerance=2):
    valid = ndimage.binary_erosion(valid,structure=np.ones((3,3)))
    def boundary(mask):
        return (mask & ~ndimage.binary_erosion(mask,structure=np.ones((3,3)))) & valid
    p,y = boundary(prediction),boundary(target)
    if not p.any() and not y.any():
        return 1.
    if not p.any() or not y.any():
        return 0.
    precision = (ndimage.distance_transform_edt(~y)[p] <= tolerance).mean()
    recall = (ndimage.distance_transform_edt(~p)[y] <= tolerance).mean()
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.
