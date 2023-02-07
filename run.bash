# python3 segment/train.py --data custom.yaml --batch 16 --weights '' --cfg yolov7-seg.yaml --epochs 300 --name yolov7-seg --img 1280 --hyp hyp.scratch-high.yaml

python3 segment/train.py --data custom.yaml --batch 8 --weights '' --cfg yolov7-seg-x.yaml --epochs 300 --name yolov7-seg-x --img 1280 --hyp hyp.scratch-high.yaml