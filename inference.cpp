#include <daal.h>

#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace daal_data = daal::data_management;
namespace daal_algo = daal::algorithms;

constexpr std::string_view native_model_path = "native_binary.txt";
constexpr std::string_view test_dataset = "test_dataset.csv";
constexpr std::size_t n_features = 18;
constexpr std::size_t n_classes = 2;

auto read_native_binary(const std::string_view path = native_model_path) {
    std::cout << "Read the daal4py model from " << path << "...\n";
    std::ifstream input{path.data(), std::ios::binary};
    input.seekg(0, std::ios::end);
    const std::size_t length = input.tellg();
    input.seekg(0, std::ios::beg);
    auto buffer = new daal::byte[length];
    input.read(reinterpret_cast<char*>(buffer), length);
    return std::pair{buffer, length};
}

void load_data_from_csv(daal_data::NumericTablePtr& data,
                        daal_data::NumericTablePtr& dependent_var,
                        const std::string_view file = test_dataset) {
    std::cout << "Read test data from " << file << "...\n";
    daal_data::FileDataSource<daal_data::CSVFeatureManager> data_source{
        file.data(), daal_data::DataSource::notAllocateNumericTable,
        daal_data::DataSource::doDictionaryFromContext};

    data.reset(new daal_data::HomogenNumericTable<>(
        n_features, 0, daal_data::NumericTable::notAllocate));
    dependent_var.reset(new daal_data::HomogenNumericTable<>(
        1, 0, daal_data::NumericTable::notAllocate));
    daal_data::NumericTablePtr merged_data(
        new daal_data::MergedNumericTable(data, dependent_var));

    data_source.loadDataBlock(merged_data.get());
}

int main() {
    auto [buffer, length] = read_native_binary();
    daal_data::OutputDataArchive out_data_arch{buffer, length};
    delete[] buffer;

    std::cout << "Unpack model...\n";
    auto deserialized_model =
        daal_algo::gbt::classification::Model::create(n_features);
    deserialized_model->deserialize(out_data_arch);

    daal_data::NumericTablePtr X_test;
    daal_data::NumericTablePtr y_test;

    load_data_from_csv(X_test, y_test);

    std::cout << "Load model...\n";
    daal_algo::gbt::classification::prediction::Batch<> algo{n_classes};
    algo.input.set(daal_algo::classifier::prediction::data, X_test);
    algo.input.set(daal_algo::classifier::prediction::model,
                   deserialized_model);
    std::cout << "Start inferring (oneDAL with C++)...\n";
    auto begin = std::chrono::system_clock::now();
    algo.compute();
    auto end = std::chrono::system_clock::now();
    std::cout << "Inference time (oneDAL with C++): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "ms\n";

    auto pred_res = algo.getResult();
    auto num_ptr = pred_res->get(daal_algo::classifier::prediction::prediction);

    // Verify that the inference results are same with python.
    auto num_of_rows = num_ptr->getNumberOfRows();
    daal_data::BlockDescriptor<float> block;
    num_ptr->getBlockOfRows(0, num_of_rows, daal_data::readOnly, block);
    float* array = block.getBlockSharedPtr().get();
    std::cout << "First ten results:\n";
    for (std::size_t i = 0; i < 10; i++)
        std::cout << "\tResult[" << i << "] = " << array[i] << "\n";
    num_ptr->releaseBlockOfRows(block);
}
