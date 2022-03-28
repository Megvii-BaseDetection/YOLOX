from json import dumps

from cv2 import data

def json(boxes, scores, cls_ids, conf=0.5, class_names=None):
  data = {'predictions':[]}
  for i in range(len(boxes)):
    box = boxes[i]
    cls_id = int(cls_ids[i])
    score = scores[i]
    if score < conf:
        continue
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    class_name = class_names[cls_id]
    bbobj = {"x": (x0 + x1)/2, 
      "y": (y0 + y1)/2, 
      "width": x1 - x0, 
      "height": y1 - y0,
      "class": class_name,
      "confidence": float(score)}
    data["predictions"].append(bbobj)
  jstr = dumps(data, ensure_ascii=False, indent=4)
  return jstr