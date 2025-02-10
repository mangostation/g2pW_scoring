import sys
sys.path.insert(0, './g2pw/')

import os
import argparse
from api import G2PWConverter
import onnxruntime

# ort_session = onnxruntime.InferenceSession("/mnt/disk1/m11115119/g2pW_scoring/saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/G2PWModel/g2pw.onnx",providers=['CUDAExecutionProvider'])

def main(config, checkpoint, sent_path, output_path=None):
    conv = G2PWConverter(style='pinyin', model_dir=checkpoint, model_source='saved_models/bert-base-chinese')
    print(conv('然而，他红了20年以后，他竟退出了大家的视线。'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=False, help='config path', default='../saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/G2PWModel/config.py')
    parser.add_argument('--checkpoint', required=False, help='checkpoint', default='/mnt/disk1/m11115119/g2pW_scoring/saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/G2PWModel')
    parser.add_argument('--sent_path', required=False, help='path of *.sent file', default='../data/zhwiki-20210501.sent')
    parser.add_argument('--output_path', required=False, help='path of prediction results', default='../data/zhwiki-20210501.sent')
    opt = parser.parse_args()

    main(opt.config, opt.checkpoint, opt.sent_path, output_path=opt.output_path)