// CPU-only ONNX reference client. Input: N, then N rows of five SI parameters.
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: simml_predict model.onnx input.txt output.txt");
        std::ifstream input(argv[2]);
        int64_t batch;
        if (!(input >> batch) || batch < 1 || batch > 256) throw std::runtime_error("batch must be in [1,256]");
        const std::array<float,5> low{2,.03f,1,.2f,.5f}, high{15,.3f,20,3,3};
        std::vector<float> parameters(static_cast<size_t>(batch)*5);
        for (size_t i=0;i<parameters.size();++i) {
            if (!(input >> parameters[i]) || !std::isfinite(parameters[i]) ||
                parameters[i]<low[i%5] || parameters[i]>high[i%5])
                throw std::runtime_error("invalid or missing physical parameter");
        }
        std::string trailing;
        if (input >> trailing) throw std::runtime_error("unexpected trailing input");
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"simml");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        Ort::Session session(env,argv[1],options);
        Ort::AllocatorWithDefaultOptions alloc;
        if (session.GetInputCount()!=1 || session.GetOutputCount()!=1 ||
            std::string(session.GetInputNameAllocated(0,alloc).get())!="parameters" ||
            std::string(session.GetOutputNameAllocated(0,alloc).get())!="metrics")
            throw std::runtime_error("wrong model input/output names");
        auto input_type=session.GetInputTypeInfo(0);
        auto output_type=session.GetOutputTypeInfo(0);
        auto it=input_type.GetTensorTypeAndShapeInfo();
        auto ot=output_type.GetTensorTypeAndShapeInfo();
        auto shape_in=it.GetShape(), shape_out=ot.GetShape();
        if (shape_in.size()!=2 || shape_in[1]!=5 || shape_out.size()!=2 || shape_out[1]!=3 ||
            it.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || ot.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("wrong model tensor shape/type");
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        const std::array<int64_t,2> shape{batch,5};
        auto tensor=Ort::Value::CreateTensor<float>(memory,parameters.data(),parameters.size(),shape.data(),2);
        const char* inputs[]{"parameters"};
        const char* outputs[]{"metrics"};
        auto result=session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
        if (result[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{batch,3})
            throw std::runtime_error("unexpected output shape");
        const float* values=result[0].GetTensorData<float>();
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot open output");
        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (int64_t row=0;row<batch;++row) {
            for (int col=0;col<3;++col) {
                float value=values[row*3+col];
                if (!std::isfinite(value) || value<0) throw std::runtime_error("invalid output value");
                output << value << (col==2?'\n':' ');
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
