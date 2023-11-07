import _init_paths
from model.utils import hell
import os
def main():
    os.system(
        'python test.py tracking --exp_id mot17_half --dataset mot --input_h 544 --input_w 960 --num_classes 1 --dataset_version 17halfval --gpus 0 --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model /home/beeno/pycharm/py_code/CenterTrack/exp/tracking/mot17_half/model_last.pth --refine_thresh 0.45 --patch_thresh 0 --num_patch_val 1 --eva --nms_thre 0.9 --nms_thre_nd 0.55 --real_num_queries 300 --hm_prob_init 0.1 --dataty_debug val --dec_layers 6')


if __name__ == '__main__':
    main()