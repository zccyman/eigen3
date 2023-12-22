
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
namespace py = pybind11;
#define FLT_MAX 1e15
#define DEBUG 0

std::vector<uint32_t> generate_histogram(const std::vector<float>& input, int num_bins, float histogram_interval)
{
    std::vector<uint32_t> histogram(num_bins, 0);
    const int size = input.size();
#pragma omp parallel for 
    for (int i = 0; i < size; i++)
    {
        if (input[i] == 0)
            continue;
        const int index = std::min(static_cast<int>(std::abs(input[i]) / histogram_interval), num_bins-1);
        histogram[index]++;
    }
    return histogram;
}

float compute_kl_divergence(const std::vector<float>& a, const std::vector<float>& b)
{
    const size_t length = a.size();
    assert(b.size() == length);
    float result = 0;

    for (size_t i = 0; i < length; i++)
    {
        if (a[i] != 0)
        {
            if (b[i] == 0)
            {
                result += 1;
            }
            else
            {
                result += a[i] * log(a[i] / b[i]);
            }
        }
    }
    return result;
}

float calculate_kld_kernel(std::vector<float> &histogram, int threshold, int target_bin)
{
    const float kl_eps = 0.0001f;
    int num_histogram_bins = histogram.size(); 

    float sum = 0;
    for (int i = 0; i < num_histogram_bins; i++)
    {
        sum += histogram[i];
    }
    bool is_normalized = sum - 1 < 1e-3 && sum - 1 > -1e-3;
    if (!is_normalized)
    {
        for (int i = 0; i < num_histogram_bins; i++)
        {
            histogram[i] = histogram[i] / sum;
        }
    }
    
    std::vector<float> clip_distribution(threshold, kl_eps);
    {
        for (int j = 0; j < threshold; j++)
        {
            clip_distribution[j] += histogram[j];
        }
        for (int j = threshold; j < num_histogram_bins; j++)
        {
            clip_distribution[threshold - 1] += histogram[j];
        }
    }

    const float num_per_bin = (float)threshold / target_bin;

    std::vector<float> quantize_distribution(target_bin, 0.f);
    {
        {
            const float end = num_per_bin;

            const int right_lower = (int)floor(end);
            const float right_scale = end - right_lower;

            if (right_scale > 0)
            {
                quantize_distribution[0] += right_scale * histogram[right_lower];
            }

            for (int k = 0; k < right_lower; k++)
            {
                quantize_distribution[0] += histogram[k];
            }

            quantize_distribution[0] /= right_lower + right_scale;
        }
        for (int j = 1; j < target_bin - 1; j++)
        {
            const float start = j * num_per_bin;
            const float end = (j + 1) * num_per_bin;

            const int left_upper = (int)ceil(start);
            const float left_scale = left_upper - start;

            const int right_lower = (int)floor(end);
            const float right_scale = end - right_lower;

            if (left_scale > 0)
            {
                quantize_distribution[j] += left_scale * histogram[left_upper - 1];
            }

            if (right_scale > 0)
            {
                quantize_distribution[j] += right_scale * histogram[right_lower];
            }

            for (int k = left_upper; k < right_lower; k++)
            {
                quantize_distribution[j] += histogram[k];
            }

            quantize_distribution[j] /= right_lower - left_upper + left_scale + right_scale;
        }
        {
            const float start = threshold - num_per_bin;

            const int left_upper = (int)ceil(start);
            const float left_scale = left_upper - start;

            if (left_scale > 0)
            {
                quantize_distribution[target_bin - 1] += left_scale * histogram[left_upper - 1];
            }

            for (int k = left_upper; k <= threshold; k++) //Modified by QinNan, for fix bug of last bin has most energy but not calculated
            {
                quantize_distribution[target_bin - 1] += histogram[k];
            }

            quantize_distribution[target_bin - 1] /= threshold - left_upper + left_scale + 1;
        }
    }

    std::vector<float> expand_distribution(threshold, kl_eps);
    {
        {
            const float end = num_per_bin;

            const int right_lower = (int)floor(end);
            const float right_scale = end - right_lower;

            if (right_scale > 0)
            {
                expand_distribution[right_lower] += right_scale * quantize_distribution[0];
            }

            for (int k = 0; k < right_lower; k++)
            {
                expand_distribution[k] += quantize_distribution[0];
            }
        }
        for (int j = 1; j < target_bin - 1; j++)
        {
            const float start = j * num_per_bin;
            const float end = (j + 1) * num_per_bin;

            const int left_upper = (int)ceil(start);
            const float left_scale = left_upper - start;

            const int right_lower = (int)floor(end);
            const float right_scale = end - right_lower;

            if (left_scale > 0)
            {
                expand_distribution[left_upper - 1] += left_scale * quantize_distribution[j];
            }

            if (right_scale > 0)
            {
                expand_distribution[right_lower] += right_scale * quantize_distribution[j];
            }

            for (int k = left_upper; k < right_lower; k++)
            {
                expand_distribution[k] += quantize_distribution[j];
            }
        }
        {
            const float start = threshold - num_per_bin;

            const int left_upper = (int)ceil(start);
            const float left_scale = left_upper - start;

            if (left_scale > 0)
            {
                expand_distribution[left_upper - 1] += left_scale * quantize_distribution[target_bin - 1];
            }

            for (int k = left_upper; k < threshold; k++)
            {
                expand_distribution[k] += quantize_distribution[target_bin - 1];
            }
        }
    }

    // kl
    float kl_divergence = compute_kl_divergence(clip_distribution, expand_distribution);
    return kl_divergence;
}

std::vector<float> calculate_kld_vector(const std::vector<float> &hist, float histogram_interval, float si, float sk)
{
    const int target_bin = 128;
    int num_histogram_bins = hist.size();
    std::vector<float>histogram(num_histogram_bins);
    std::vector<float>kld_vec(num_histogram_bins, FLT_MAX);
    //Normalization
    {
        float sum = 0;
        for (int j = 0; j < num_histogram_bins; j++)
        {
            sum += hist[j];
        }

        for (int j = 0; j < num_histogram_bins; j++)
        {
            histogram[j] = (float)(hist[j] / sum);
        }
    }
    
    int target_threshold = target_bin;
    float min_kl_divergence = FLT_MAX;

    int threshold_prev = 0;
    for (int idx = target_bin; idx < num_histogram_bins; idx++)
    {
        //std::cout<<"check bin: "<< idx <<std::endl;
        
        int threshold = idx;

        if (si != -1 && sk != -1)
        {
            float so = idx * histogram_interval / 127;
            float x = si * sk / so;
            int out_shift = 0;
            float out_scale = 0;
            for (out_shift = -32; out_shift < 32; out_shift++)
            {
                out_scale = x * pow(2, -out_shift);
                if (out_scale <= 1 && out_scale > 0.5)
                    break;
            }  
            
            so = so * out_scale;   
            threshold = int(so * 127 / histogram_interval + 0.5); 
            if (threshold <= 128)
            {
                continue;
            }
            if (threshold_prev==0)
            {
                threshold_prev = threshold;
#if DEBUG
                std::cout<<"First time -- check bin: "<< idx <<" out_shift = "<<out_shift<<" out_scale = "<<out_scale<<" threshold = "<< threshold <<" kdl_min =" << min_kl_divergence << std::endl;  
#endif
                threshold_prev = threshold;                
            }  
            if (threshold != threshold_prev) 
            {
#if DEBUG
                std::cout<<"check bin: "<< idx <<" out_shift = "<<out_shift<<" out_scale = "<<out_scale<<" threshold = "<< threshold <<" kdl_min =" << min_kl_divergence << std::endl;  
#endif
                threshold_prev = threshold;
            }
        }
        
        // calc kld
        float kl_divergence = calculate_kld_kernel(histogram, threshold, target_bin);
        // save to list
        kld_vec[idx] = kl_divergence;
        // the best num of bin
        if (kl_divergence < min_kl_divergence)
        {
            min_kl_divergence = kl_divergence;
            //target_threshold = threshold;
            target_threshold = idx; // modified by qinnan for intscale
#if DEBUG
            std::cout<<"KL min = "<<min_kl_divergence<<" bin = "<<target_threshold<<std::endl;
#endif
        }
    }
#if DEBUG
    std::cout<<"########## Finish KL min = "<<min_kl_divergence<<" bin = "<<target_threshold<<std::endl;
#endif    
    //return (target_threshold+1) * histogram_interval / 127;
    return kld_vec;
}


float calculate_threshold(const std::vector<float> &hist, float histogram_interval, float si, float sk)
{
    const int target_bin = 128;
    int num_histogram_bins = hist.size();
    std::vector<float>histogram(num_histogram_bins);
    //Normalization
    {
        float sum = 0;
        for (int j = 0; j < num_histogram_bins; j++)
        {
            sum += hist[j];
        }

        for (int j = 0; j < num_histogram_bins; j++)
        {
            histogram[j] = (float)(hist[j] / sum);
        }
    }
    
    int target_threshold = target_bin;
    float min_kl_divergence = FLT_MAX;

    int threshold_prev = 0;
    for (int idx = target_bin; idx < num_histogram_bins; idx++)
    {
        //std::cout<<"check bin: "<< idx <<std::endl;
        const float kl_eps = 0.0001f;
        int threshold = idx;

        if (si != -1 && sk != -1)
        {
            float so = idx * histogram_interval / 127;
            float x = si * sk / so;
            int out_shift = 0;
            float out_scale = 0;
            for (out_shift = -32; out_shift < 32; out_shift++)
            {
                out_scale = x * pow(2, -out_shift);
                if (out_scale <= 1 && out_scale > 0.5)
                    break;
            }  
            
            so = so * out_scale;   
            threshold = int(so * 127 / histogram_interval + 0.5); 
            if (threshold <= 128)
            {
                continue;
            }
            if (threshold_prev==0)
            {
                threshold_prev = threshold;
#if DEBUG
                std::cout<<"First time -- check bin: "<< idx <<" out_shift = "<<out_shift<<" out_scale = "<<out_scale<<" threshold = "<< threshold <<" kdl_min =" << min_kl_divergence << std::endl;  
#endif
                threshold_prev = threshold;                
            }  
            if (threshold != threshold_prev) 
            {
#if DEBUG
                std::cout<<"check bin: "<< idx <<" out_shift = "<<out_shift<<" out_scale = "<<out_scale<<" threshold = "<< threshold <<" kdl_min =" << min_kl_divergence << std::endl;  
#endif
                threshold_prev = threshold;
            }
        }
        
        std::vector<float> clip_distribution(threshold, kl_eps);
        {
            for (int j = 0; j < threshold; j++)
            {
                clip_distribution[j] += histogram[j];
            }
            for (int j = threshold; j < num_histogram_bins; j++)
            {
                clip_distribution[threshold - 1] += histogram[j];
            }
        }

        const float num_per_bin = (float)threshold / target_bin;

        std::vector<float> quantize_distribution(target_bin, 0.f);
        {
            {
                const float end = num_per_bin;

                const int right_lower = (int)floor(end);
                const float right_scale = end - right_lower;

                if (right_scale > 0)
                {
                    quantize_distribution[0] += right_scale * histogram[right_lower];
                }

                for (int k = 0; k < right_lower; k++)
                {
                    quantize_distribution[0] += histogram[k];
                }

                quantize_distribution[0] /= right_lower + right_scale;
            }
            for (int j = 1; j < target_bin - 1; j++)
            {
                const float start = j * num_per_bin;
                const float end = (j + 1) * num_per_bin;

                const int left_upper = (int)ceil(start);
                const float left_scale = left_upper - start;

                const int right_lower = (int)floor(end);
                const float right_scale = end - right_lower;

                if (left_scale > 0)
                {
                    quantize_distribution[j] += left_scale * histogram[left_upper - 1];
                }

                if (right_scale > 0)
                {
                    quantize_distribution[j] += right_scale * histogram[right_lower];
                }

                for (int k = left_upper; k < right_lower; k++)
                {
                    quantize_distribution[j] += histogram[k];
                }

                quantize_distribution[j] /= right_lower - left_upper + left_scale + right_scale;
            }
            {
                const float start = threshold - num_per_bin;

                const int left_upper = (int)ceil(start);
                const float left_scale = left_upper - start;

                if (left_scale > 0)
                {
                    quantize_distribution[target_bin - 1] += left_scale * histogram[left_upper - 1];
                }

                //for (int k = left_upper; k < threshold; k++)
                for (int k = left_upper; k <= threshold; k++) //Modified by QinNan, for fix bug of last bin has most energy but not calculated
                {
                    quantize_distribution[target_bin - 1] += histogram[k];
                }

                quantize_distribution[target_bin - 1] /= threshold - left_upper + left_scale + 1;
            }
        }

        std::vector<float> expand_distribution(threshold, kl_eps);
        {
            {
                const float end = num_per_bin;

                const int right_lower = (int)floor(end);
                const float right_scale = end - right_lower;

                if (right_scale > 0)
                {
                    expand_distribution[right_lower] += right_scale * quantize_distribution[0];
                }

                for (int k = 0; k < right_lower; k++)
                {
                    expand_distribution[k] += quantize_distribution[0];
                }
            }
            for (int j = 1; j < target_bin - 1; j++)
            {
                const float start = j * num_per_bin;
                const float end = (j + 1) * num_per_bin;

                const int left_upper = (int)ceil(start);
                const float left_scale = left_upper - start;

                const int right_lower = (int)floor(end);
                const float right_scale = end - right_lower;

                if (left_scale > 0)
                {
                    expand_distribution[left_upper - 1] += left_scale * quantize_distribution[j];
                }

                if (right_scale > 0)
                {
                    expand_distribution[right_lower] += right_scale * quantize_distribution[j];
                }

                for (int k = left_upper; k < right_lower; k++)
                {
                    expand_distribution[k] += quantize_distribution[j];
                }
            }
            {
                const float start = threshold - num_per_bin;

                const int left_upper = (int)ceil(start);
                const float left_scale = left_upper - start;

                if (left_scale > 0)
                {
                    expand_distribution[left_upper - 1] += left_scale * quantize_distribution[target_bin - 1];
                }

                for (int k = left_upper; k < threshold; k++)
                {
                    expand_distribution[k] += quantize_distribution[target_bin - 1];
                }
            }
        }

        // kl
        const float kl_divergence = compute_kl_divergence(clip_distribution, expand_distribution);

        // the best num of bin
        if (kl_divergence < min_kl_divergence)
        {
            min_kl_divergence = kl_divergence;
            //target_threshold = threshold;
            target_threshold = idx; // modified by qinnan for intscale
#if DEBUG
            std::cout<<"KL min = "<<min_kl_divergence<<" bin = "<<target_threshold<<std::endl;
#endif
        }
    }
#if DEBUG
    std::cout<<"########## Finish KL min = "<<min_kl_divergence<<" bin = "<<target_threshold<<std::endl;
#endif    
    return (target_threshold+1) * histogram_interval / 127;
}

PYBIND11_MODULE(kld, m) {
	m.doc() = "";
	m.def("calculate_threshold", &calculate_threshold, "");
	m.def("calculate_kld_vector", &calculate_kld_vector, "");
    m.def("calculate_kld_kernel", &calculate_kld_kernel, "");
}