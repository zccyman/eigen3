#include <map>
#include "register_custom_ops.h"
#include "custom_op_library.h"
#include "custom_splice.h"
#include "custom_deformconv.h"

#include "onnxruntime_cxx_api.h"

// static const char *c_OpDomain = "test.customop";
static const char *c_OpDomain = "timesintelli.com";

MMCVDeformConvOp c_MMCVDeformConvOp;
CustomOpTwo c_CustomOpTwo;
CustomOpOne c_CustomOpOne;
TIMESINTELLISpliceOp c_TIMESINTELLISplice;
std::vector<OrtCustomOp *> ops = {&c_MMCVDeformConvOp, &c_CustomOpTwo, &c_CustomOpOne, &c_TIMESINTELLISplice};
std::map<std::string, std::vector<OrtCustomOp *>> ops_list = {
    std::make_pair("test.customop", ops),
    std::make_pair("timesintelli.com", ops),
};

struct OrtCustomOpDomainDeleter
{
    explicit OrtCustomOpDomainDeleter(const OrtApi *ort_api)
    {
        ort_api_ = ort_api;
    }
    void operator()(OrtCustomOpDomain *domain) const
    {
        ort_api_->ReleaseCustomOpDomain(domain);
    }

    const OrtApi *ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain *domain, const OrtApi *ort_api)
{
    std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
    auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
    ort_custom_op_domain_container.push_back(std::move(ptr));
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api)
{
    OrtStatus * status;
    for (auto iter=ops_list.begin(); iter!=ops_list.end(); iter++){
        std::string s_OpDomain = iter->first;
        std::vector<OrtCustomOp *> ops = iter->second;
        OrtCustomOpDomain *domain = nullptr;
        const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

        if (auto status = ortApi->CreateCustomOpDomain(s_OpDomain.c_str(), &domain))
        {
            return status;
        }

        AddOrtCustomOpDomainToContainer(domain, ortApi);
        for(auto &op : ops){
            if (auto status = ortApi->CustomOpDomain_Add(domain, op))
            {
                return status;
            }
        }
        status = ortApi->AddCustomOpDomain(options, domain);
    }
   
    return status;
    
}