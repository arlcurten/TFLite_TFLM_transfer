# TensorFLow Lite & TFLM (Microprocessor) Model Transfering

**Purpose**: transfering TensorFlow Keras model into TFLite & TFLM versions

**Settings**: 
Python 3.10.16  
Tensorflow 2.14.0
Tensorflow-intel 2.14.0  (since I use windows)
Numpy 1.26.4  (need to be <2.0)
Keras 2.14.0
Tf-keras 2.14.1


# Project Structure
```bash
project-root/
├── README.md                         # Project overview
├── src/                
│   ├── transformer_block_tracer.py   # Main function
│   ├── llama_loader.py               # Model import
│   ├── scheduler.py                  # Scheduling tasks (in processing)
│   └── perfetto_writer.py            # Write JSON for Perfetto 
│       └── profiler.py               # Profiles for writer
└── output/
    ├── llama_model_structure.txt     # Impoted Model dump
    └── transformer_trace.json        # JSON file to Perfetto
```

2. TFLM directory
    -> model_tflite.cc (TFLM model)
    -> main_function.cc (checked no syntax error but haven't resolved library dependency issue for entire compilation (probably need a pure Linux environment & platform info) )
3. assignment.ipynb (Jupyter notebook script to transfer & inference model)
4. tf_lite_model.tflite (tflite model)

<br/>

**To-do Items**:
1. Well adjust scheduling with other algorithms
2. Profile(duration) update should be accommodated to real execution environment 
   (maybe perform a forward pass with torch.profile() on original model → update parameters in profiler.py/directly edit default values of logging)

<br/>

**References**:
