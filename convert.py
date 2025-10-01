from hailo_sdk_client import ClientRunner, InferenceContext
import os
import numpy as np
import argparse
import subprocess

def main(args):
    chosen_hw_arch = "hailo8l"

    os.makedirs(args.name, exist_ok=True)
    
    hailo_model_har_path = os.path.join(args.name, f"{args.name}_hailo_model.har")
    if not os.path.exists(hailo_model_har_path):
        # Parsing
        runner = ClientRunner(hw_arch=chosen_hw_arch)
        hn, npz = runner.translate_onnx_model(
            args.onnx_path,
            args.name,
            # end_node_names=['/gender_out/Conv', '/age_out/Conv']
        )
        runner.save_har(hailo_model_har_path)
        svg_path = os.path.join(args.name, f"{args.name}.svg")
        subprocess.run(
        ["hailo", "visualizer", str(hailo_model_har_path), "--no-browser", "--out-path", str(svg_path)],
        check=True,
    )
    else:
        print('The model has been parse, continue to optimize')
    
    # 2. Optimizing        
    quantized_model_har_path = os.path.join(args.name, f"{args.name}_quantized_model.har")
    if not os.path.exists(quantized_model_har_path):
        # First, we will prepare the calibration set. Resize the images to the correct size and crop them.
        calib_set = np.load(args.calib_set)
        assert os.path.isfile(hailo_model_har_path), "Please provide valid path for HAR file"
        runner = ClientRunner(har=hailo_model_har_path)
        # By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.

        # Now we will create a model script, that tells the compiler to add a normalization on the beginning
        # of the model (that is why we didn't normalize the calibration set;
        # Otherwise we would have to normalize it before using it)
        # # Load the model script to ClientRunner so it will be considered on optimization
        model_script_path = args.model_script
        if os.path.exists(model_script_path):
            runner.load_model_script(model_script_path)

        # Call Optimize to perform the optimization process
        runner.optimize(calib_set)

        # Save the result state to a Quantized HAR file
        
        runner.save_har(quantized_model_har_path)
    else:
        print('The model has been optimized, continue to compiling!')
    
    out_path = os.path.join(args.name, f"{args.name}.hef")
    if not os.path.exists(out_path):
        runner = ClientRunner(har=quantized_model_har_path)
        compile_script_path = args.compile_script
        if os.path.exists(compile_script_path):
            runner.load_model_script(compile_script_path)

        hef = runner.compile()
        with open(out_path, "wb") as f:
            f.write(hef)

        har_path = os.path.join(args.name, f"{args.name}_compiled_model.har")
        runner.save_har(har_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ONNX to HEF')
    parser.add_argument('name', type=str, help='The name of the model.')
    parser.add_argument('--onnx-path', help='ONNX file path.')
    parser.add_argument('--calib-set', help='Calibration set folder path.')
    parser.add_argument('--model-script', default='model_script.alls', help='Model script path for optimize.')
    parser.add_argument('--compile-script', default='compile_script.alls', help='Compile script path for compile.')
    
    args = parser.parse_args()
    
    main(args)