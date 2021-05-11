#include <flashlight/fl/flashlight.h>
#include <flashlight/app/asr/criterion/SequenceCriterion.h>
#include <flashlight/app/asr/data/Utils.h>
#include <flashlight/lib/audio/feature/FeatureParams.h>
#include <flashlight/lib/sequence/criterion/Defines.h>
#include <flashlight/lib/text/dictionary/Dictionary.h>

class Engine {
public:
    Engine();
    ~Engine() {}

    void test(int, int);
    w2l_emission *forward(float *samples, size_t sample_count);
    af::array process(const af::array &features);

    bool loadW2lModel(std::string modelPath, std::string tokensPath);
    bool loadB2lModel(std::string path);
    bool exportW2lModel(std::string path);
    bool exportB2lModel(std::string path);

    std::vector<float> transitions() const;
private:
    std::string exportTokens();
    std::string layerArch(fl::Module *module);
    void loadFlags(std::map<std::string, std::string> &flags);

private:
    bool loaded;
    std::map<std::string, std::string> savedFlags;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<fl::Module> network;
    std::shared_ptr<fl::app::asr::SequenceCriterion> criterion;
    std::string criterionType;
    fl::lib::text::Dictionary tokenDict;

    int featCount;
    fl::app::asr::FeatureType featType;
    fl::lib::audio::FeatureParams featParams;
    fl::Dataset::DataTransformFunction inputFeatures;
};
