//
// Description: This code is a C++ implementation of a TensorFlow Lite Micro (TFLM) model inference on a microcontroller.
// setup() : Initialization (load the model, initializationetc.)
// loop(): Read sensor data + call interpreter for inference 
//
// References:
// https://ai.google.dev/edge/litert/microcontrollers/get_started#run_inference
// https://developer.arm.com/documentation/109267/0101/ML-software-development-for-Arm-Cortex-M-processors/Example-software-development-flow-using-TFLM
//

#include "model_tflite.cc"  // compiled TFLite model
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <iostream>
#include <chrono>
#include <thread>

using namespace std;

// Constants
constexpr int kTensorArenaSize = 60 * 1024;  // Increase if model allocation fails
uint8_t tensor_arena[kTensorArenaSize];

// Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// === Setup function ===
void setup() {
    // Load model
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        cerr << "Model schema version mismatch!" << endl;
        exit(1);
    }

    // Initialize op resolver
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddSoftmax(); 

    // Create interpreter
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate tensor buffers
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        cerr << "Tensor allocation failed!" << endl;
        exit(1);
    }

    // Get input and output tensors
    input = interpreter.input(0);
    output = interpreter.output(0);

    // Debug info
    cout << "Setup completed. Input size: " << input->bytes << ", Output size: " << output->bytes << endl;
}

// === Loop function ===
void loop() {
    // Simulate sensor data input (replace with actual sensor reading logic)
    if (input->dims->size != 3 || input->dims->data[1] != 55 || input->dims->data[2] != 12) {
        cerr << "Unexpected input shape!" << endl;
        return;
    }
    // Fill input with dummy data for demo
    for (int i = 0; i < 55 * 12; ++i) {
        input->data.f[i] = static_cast<float>(i % 10) / 10.0f;
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        cerr << "Model inference failed!" << endl;
        return;
    }

    // Read and print output
    for (int i = 0; i < 55; ++i) {
        float class0 = output->data.f[i * 2 + 0];
        float class1 = output->data.f[i * 2 + 1];
        cout << "Frame " << i << ": Class0=" << class0 << ", Class1=" << class1 << endl;
    }

    // Simulate real-time delay
    this_thread::sleep_for(chrono::milliseconds(1000));
}

// === Main entry point ===
int main(int argc, char const *argv[]) {
    setup();
    while (true) {
        loop();
    }
    return 0;
}
