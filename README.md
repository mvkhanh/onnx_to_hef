nohup /home/ubuntu/miniconda3/envs/hailo310/bin/python -u /data/khanh/convert_onnx_to_hef/convert.py emotion --onnx-path /data/khanh/convert_onnx_to_hef/model/icml_emotion.b1.onnx --calib-set /data/khanh/convert_onnx_to_hef/calib_set/calibration_0_255_emotion.npy \
  > /data/khanh/convert_onnx_to_hef/emotion_convert.log 2>&1 & echo $! > convert.pid

kill $(cat convert.pid)