#include "soma/runtime/timestep_spike_attention.hpp"
#include <stdexcept>
namespace soma {
TimestepSpikeAttention::TimestepSpikeAttention(std::uint32_t h,std::uint32_t r,std::uint32_t d,std::uint64_t b,std::uint64_t c,float s):h_(h),r_(r),d_(d),begin_(b),count_(c),scale_(s){if(!h||!r||!d||b+c>static_cast<std::uint64_t>(h)*r*d)throw std::runtime_error("timestep_spike_attention shape 非法");}
TimestepAttentionResult TimestepSpikeAttention::update(const std::vector<std::int8_t>&q,const std::vector<std::int8_t>&k,const std::vector<std::int8_t>&v)const{
 const auto n=static_cast<std::size_t>(h_)*r_*d_;if(q.size()!=n||k.size()!=n||v.size()!=n)throw std::runtime_error("timestep_spike_attention operand shape 不匹配");
 TimestepAttentionResult z;z.output.resize(count_);std::vector<std::int32_t> kv(static_cast<std::size_t>(h_)*d_*d_);
 for(std::uint32_t h=0;h<h_;++h)for(std::uint32_t a=0;a<d_;++a)for(std::uint32_t b=0;b<d_;++b)for(std::uint32_t r=0;r<r_;++r){kv[(h*d_+a)*d_+b]+=static_cast<int>(k[(h*r_+r)*d_+a])*static_cast<int>(v[(h*r_+r)*d_+b]);++z.kv_updates;}
 for(std::uint64_t i=0;i<count_;++i){auto g=begin_+i;auto c=g%d_;auto h=(g/d_)%h_;auto r=g/(static_cast<std::uint64_t>(h_)*d_);std::int32_t x=0;for(std::uint32_t a=0;a<d_;++a){x+=static_cast<int>(q[(h*r_+r)*d_+a])*kv[(h*d_+a)*d_+c];++z.q_updates;}z.output[i]=x*scale_;}return z;
}
}
