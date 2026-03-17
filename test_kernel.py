import os
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
import torch, torch.utils.cpp_extension
def _gfx950_only(cflags=None): return ['--offload-arch=gfx950']
torch.utils.cpp_extension._get_rocm_arch_flags = _gfx950_only
from functools import lru_cache
from torch.utils.cpp_extension import load_inline

NUM_HEADS=16; KV_LORA_RANK=512; QK_ROPE_HEAD_DIM=64
QK_HEAD_DIM=KV_LORA_RANK+QK_ROPE_HEAD_DIM; V_HEAD_DIM=KV_LORA_RANK

# =========================================================================
# HIP C++ kernel: Fused Q quantize + stage1 with pipelined V accum (v60)
# =========================================================================
HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>
namespace mla {
static constexpr int NH=16,DQK=576,DV=512,THR=256,WS=64,KVT=32,HPW=4,VE=DV/WS;
static constexpr float SMSC=1.0f/24.0f,LOG2E=1.4426950408889634f;
static constexpr int MFMAK=16,NMFMA=DQK/MFMAK,QLDS=NH*DQK,KVLDS=KVT*DQK,SMEM=QLDS+2*KVLDS;
using bf16=hip_bfloat16;
using fp8x8_t=__attribute__((vector_size(8))) unsigned char;
using f32x16_t=__attribute__((vector_size(64))) float;
__device__ __forceinline__ float tof(bf16 x){return static_cast<float>(x);}
__device__ __forceinline__ bf16 tob(float x){return bf16(x);}
__device__ __forceinline__ uint8_t t8(float f){f=fminf(fmaxf(f,-240.f),240.f);uint32_t r=0;r=__builtin_amdgcn_cvt_pk_fp8_f32(f,0.f,r,false);return(uint8_t)(r&0xFF);}
__device__ __forceinline__ int qoff(int qi,int h,int d){return(qi*NH+h)*DQK+d;}
__device__ __forceinline__ int ooff(int qi,int h,int d){return(qi*NH+h)*DV+d;}
__device__ __forceinline__ int poff(int b,int s,int h,int ns){return(b*ns+s)*NH+h;}
__device__ __forceinline__ int voff(int b,int s,int h,int d,int ns){return((b*ns+s)*NH+h)*DV+d;}
__device__ __forceinline__ void smg(float mb,float lb,float ob,float&ma,float&la,float&oa){float mn=fmaxf(ma,mb);float fa=expf(ma-mn),fb=expf(mb-mn);oa=fa*oa+fb*ob;la=fa*la+fb*lb;ma=mn;}

// Fused Q quantize: read bf16 Q from global, convert to fp8, write to LDS
__device__ inline void fused_q_to_lds(const bf16*__restrict__ q, uint8_t*ql, int qi, int tid){
    // Each thread converts multiple elements. Q is [qi, NH, DQK] in bf16
    // LDS layout: ql[h*DQK + d] for h in 0..15, d in 0..575
    // Total: NH*DQK = 9216 bytes to write
    for(int i=tid;i<NH*DQK;i+=THR){
        int h=i/DQK,d=i%DQK;
        ql[i]=t8(tof(q[qoff(qi,h,d)]));
    }
}

__global__ __launch_bounds__(THR,1)
void mla_s2(const float*__restrict__ pm,const float*__restrict__ pl,const float*__restrict__ pv,bf16*__restrict__ out,const int32_t*__restrict__ qoi,int bs,int ns){
    int b=blockIdx.x;if(b>=bs)return;int tid=threadIdx.x,w=tid>>6,ln=tid&63,hb=w*HPW;
    int qs_=qoi[b],qe_=qoi[b+1];if(qe_-qs_!=1)return;int qi=qs_;
    float mm[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY},ll[4]={0,0,0,0};
    float r0[VE]={0},r1[VE]={0},r2[VE]={0},r3[VE]={0};
    for(int s=0;s<ns;++s){float m0=pm[poff(b,s,hb,ns)],m1=pm[poff(b,s,hb+1,ns)],m2=pm[poff(b,s,hb+2,ns)],m3=pm[poff(b,s,hb+3,ns)];
        float l0=pl[poff(b,s,hb,ns)],l1=pl[poff(b,s,hb+1,ns)],l2=pl[poff(b,s,hb+2,ns)],l3=pl[poff(b,s,hb+3,ns)];
        #pragma unroll
        for(int j=0;j<VE;++j){int vd=ln+j*WS;smg(m0,l0,pv[voff(b,s,hb,vd,ns)],mm[0],ll[0],r0[j]);smg(m1,l1,pv[voff(b,s,hb+1,vd,ns)],mm[1],ll[1],r1[j]);
            smg(m2,l2,pv[voff(b,s,hb+2,vd,ns)],mm[2],ll[2],r2[j]);smg(m3,l3,pv[voff(b,s,hb+3,vd,ns)],mm[3],ll[3],r3[j]);}}
    for(int j=0;j<VE;++j){int vd=ln+j*WS;out[ooff(qi,hb,vd)]=tob(r0[j]/ll[0]);out[ooff(qi,hb+1,vd)]=tob(r1[j]/ll[1]);out[ooff(qi,hb+2,vd)]=tob(r2[j]/ll[2]);out[ooff(qi,hb+3,vd)]=tob(r3[j]/ll[3]);}}

__device__ inline void skv(const uint8_t*kv,uint8_t*kl,int base,int rows,int tid){int t16=(rows*DQK)/16;for(int i=tid;i<t16;i+=THR){int e=i*16,r=e/DQK,d=e%DQK;*reinterpret_cast<uint4*>(&kl[r*DQK+d])=*reinterpret_cast<const uint4*>(&kv[(base+r)*DQK+d]);}int rem=t16*16;for(int i=tid;i<rows*DQK-rem;i+=THR){int e=rem+i,r=e/DQK,d=e%DQK;kl[r*DQK+d]=kv[(base+r)*DQK+d];}}
__device__ __forceinline__ f32x16_t asm_mfma_qk(const uint8_t*ql,const uint8_t*cur,int col32,int grp,int rows){f32x16_t sa;for(int i=0;i<16;++i)sa[i]=0.f;uint64_t af_c,bf_c,af_n,bf_n;if(col32<NH&&grp*8+7<DQK)af_c=*reinterpret_cast<const uint64_t*>(&ql[col32*DQK+grp*8]);else af_c=0;if(col32<rows&&grp*8+7<DQK)bf_c=*reinterpret_cast<const uint64_t*>(&cur[col32*DQK+grp*8]);else bf_c=0;for(int mk=0;mk<NMFMA;++mk){if(mk+1<NMFMA){int kb1=(mk+1)*MFMAK;if(col32<NH&&kb1+grp*8+7<DQK)af_n=*reinterpret_cast<const uint64_t*>(&ql[col32*DQK+kb1+grp*8]);else af_n=0;if(col32<rows&&kb1+grp*8+7<DQK)bf_n=*reinterpret_cast<const uint64_t*>(&cur[col32*DQK+kb1+grp*8]);else bf_n=0;}asm volatile("s_waitcnt lgkmcnt(0)\n s_setprio 2":::"memory");sa=__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)af_c,(long)bf_c,sa,0,0,0);asm volatile("s_setprio 0":::);asm volatile("":::"memory");af_c=af_n;bf_c=bf_n;}return sa;}

// Pipelined V accumulation: prefetch next token's reads while computing current
__device__ __forceinline__ void asm_vaccum(
    uint32_t ad0,uint32_t ad1,uint32_t ad2,uint32_t ad3,uint32_t ad4,uint32_t ad5,uint32_t ad6,uint32_t ad7,
    float sc0,float sc1,float sc2,float sc3,
    float&m0,float&l0,float&m1,float&l1,float&m2,float&l2,float&m3,float&l3,
    float&a00,float&a01,float&a02,float&a03,float&a04,float&a05,float&a06,float&a07,
    float&a10,float&a11,float&a12,float&a13,float&a14,float&a15,float&a16,float&a17,
    float&a20,float&a21,float&a22,float&a23,float&a24,float&a25,float&a26,float&a27,
    float&a30,float&a31,float&a32,float&a33,float&a34,float&a35,float&a36,float&a37,
    float log2e,float kvsc)
{
    uint32_t rv0,rv1,rv2,rv3,rv4,rv5,rv6,rv7;
    float al0,bt0,al1,bt1,al2,bt2,al3,bt3,mn,d0,d1,vv;

    // Issue current token's LDS reads, then softmax (hides latency),
    // then wait and do FMA, then issue NEXT token's reads at the end
    asm volatile(
        // === Issue 8 LDS reads for current token ===
        "ds_read_u8 %[rv0],%[ad0]\n"
        "ds_read_u8 %[rv1],%[ad1]\n"
        "ds_read_u8 %[rv2],%[ad2]\n"
        "ds_read_u8 %[rv3],%[ad3]\n"
        "ds_read_u8 %[rv4],%[ad4]\n"
        "ds_read_u8 %[rv5],%[ad5]\n"
        "ds_read_u8 %[rv6],%[ad6]\n"
        "ds_read_u8 %[rv7],%[ad7]\n"
        // === Softmax for 4 heads (40 VALU ops, hides LDS latency) ===
        "v_max_f32 %[mn],%[m0],%[sc0]\n"
        "v_sub_f32 %[d0],%[m0],%[mn]\n"
        "v_sub_f32 %[d1],%[sc0],%[mn]\n"
        "v_mul_f32 %[d0],%[d0],%[log2e]\n"
        "v_mul_f32 %[d1],%[d1],%[log2e]\n"
        "v_exp_f32 %[al0],%[d0]\n"
        "v_exp_f32 %[bt0],%[d1]\n"
        "v_mul_f32 %[l0],%[al0],%[l0]\n"
        "v_add_f32 %[l0],%[l0],%[bt0]\n"
        "v_mov_b32 %[m0],%[mn]\n"
        "v_max_f32 %[mn],%[m1],%[sc1]\n"
        "v_sub_f32 %[d0],%[m1],%[mn]\n"
        "v_sub_f32 %[d1],%[sc1],%[mn]\n"
        "v_mul_f32 %[d0],%[d0],%[log2e]\n"
        "v_mul_f32 %[d1],%[d1],%[log2e]\n"
        "v_exp_f32 %[al1],%[d0]\n"
        "v_exp_f32 %[bt1],%[d1]\n"
        "v_mul_f32 %[l1],%[al1],%[l1]\n"
        "v_add_f32 %[l1],%[l1],%[bt1]\n"
        "v_mov_b32 %[m1],%[mn]\n"
        "v_max_f32 %[mn],%[m2],%[sc2]\n"
        "v_sub_f32 %[d0],%[m2],%[mn]\n"
        "v_sub_f32 %[d1],%[sc2],%[mn]\n"
        "v_mul_f32 %[d0],%[d0],%[log2e]\n"
        "v_mul_f32 %[d1],%[d1],%[log2e]\n"
        "v_exp_f32 %[al2],%[d0]\n"
        "v_exp_f32 %[bt2],%[d1]\n"
        "v_mul_f32 %[l2],%[al2],%[l2]\n"
        "v_add_f32 %[l2],%[l2],%[bt2]\n"
        "v_mov_b32 %[m2],%[mn]\n"
        "v_max_f32 %[mn],%[m3],%[sc3]\n"
        "v_sub_f32 %[d0],%[m3],%[mn]\n"
        "v_sub_f32 %[d1],%[sc3],%[mn]\n"
        "v_mul_f32 %[d0],%[d0],%[log2e]\n"
        "v_mul_f32 %[d1],%[d1],%[log2e]\n"
        "v_exp_f32 %[al3],%[d0]\n"
        "v_exp_f32 %[bt3],%[d1]\n"
        "v_mul_f32 %[l3],%[al3],%[l3]\n"
        "v_add_f32 %[l3],%[l3],%[bt3]\n"
        "v_mov_b32 %[m3],%[mn]\n"
        // === Wait for LDS reads, convert fp8→f32, rescale+FMA ===
        "s_waitcnt lgkmcnt(0)\n"
        "v_cvt_f32_fp8 %[vv],%[rv0]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a00],%[al0],%[a00]\nv_fmac_f32 %[a00],%[bt0],%[vv]\n"
        "v_mul_f32 %[a10],%[al1],%[a10]\nv_fmac_f32 %[a10],%[bt1],%[vv]\n"
        "v_mul_f32 %[a20],%[al2],%[a20]\nv_fmac_f32 %[a20],%[bt2],%[vv]\n"
        "v_mul_f32 %[a30],%[al3],%[a30]\nv_fmac_f32 %[a30],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv1]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a01],%[al0],%[a01]\nv_fmac_f32 %[a01],%[bt0],%[vv]\n"
        "v_mul_f32 %[a11],%[al1],%[a11]\nv_fmac_f32 %[a11],%[bt1],%[vv]\n"
        "v_mul_f32 %[a21],%[al2],%[a21]\nv_fmac_f32 %[a21],%[bt2],%[vv]\n"
        "v_mul_f32 %[a31],%[al3],%[a31]\nv_fmac_f32 %[a31],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv2]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a02],%[al0],%[a02]\nv_fmac_f32 %[a02],%[bt0],%[vv]\n"
        "v_mul_f32 %[a12],%[al1],%[a12]\nv_fmac_f32 %[a12],%[bt1],%[vv]\n"
        "v_mul_f32 %[a22],%[al2],%[a22]\nv_fmac_f32 %[a22],%[bt2],%[vv]\n"
        "v_mul_f32 %[a32],%[al3],%[a32]\nv_fmac_f32 %[a32],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv3]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a03],%[al0],%[a03]\nv_fmac_f32 %[a03],%[bt0],%[vv]\n"
        "v_mul_f32 %[a13],%[al1],%[a13]\nv_fmac_f32 %[a13],%[bt1],%[vv]\n"
        "v_mul_f32 %[a23],%[al2],%[a23]\nv_fmac_f32 %[a23],%[bt2],%[vv]\n"
        "v_mul_f32 %[a33],%[al3],%[a33]\nv_fmac_f32 %[a33],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv4]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a04],%[al0],%[a04]\nv_fmac_f32 %[a04],%[bt0],%[vv]\n"
        "v_mul_f32 %[a14],%[al1],%[a14]\nv_fmac_f32 %[a14],%[bt1],%[vv]\n"
        "v_mul_f32 %[a24],%[al2],%[a24]\nv_fmac_f32 %[a24],%[bt2],%[vv]\n"
        "v_mul_f32 %[a34],%[al3],%[a34]\nv_fmac_f32 %[a34],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv5]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a05],%[al0],%[a05]\nv_fmac_f32 %[a05],%[bt0],%[vv]\n"
        "v_mul_f32 %[a15],%[al1],%[a15]\nv_fmac_f32 %[a15],%[bt1],%[vv]\n"
        "v_mul_f32 %[a25],%[al2],%[a25]\nv_fmac_f32 %[a25],%[bt2],%[vv]\n"
        "v_mul_f32 %[a35],%[al3],%[a35]\nv_fmac_f32 %[a35],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv6]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a06],%[al0],%[a06]\nv_fmac_f32 %[a06],%[bt0],%[vv]\n"
        "v_mul_f32 %[a16],%[al1],%[a16]\nv_fmac_f32 %[a16],%[bt1],%[vv]\n"
        "v_mul_f32 %[a26],%[al2],%[a26]\nv_fmac_f32 %[a26],%[bt2],%[vv]\n"
        "v_mul_f32 %[a36],%[al3],%[a36]\nv_fmac_f32 %[a36],%[bt3],%[vv]\n"
        "v_cvt_f32_fp8 %[vv],%[rv7]\n"
        "v_mul_f32 %[vv],%[vv],%[kvsc]\n"
        "v_mul_f32 %[a07],%[al0],%[a07]\nv_fmac_f32 %[a07],%[bt0],%[vv]\n"
        "v_mul_f32 %[a17],%[al1],%[a17]\nv_fmac_f32 %[a17],%[bt1],%[vv]\n"
        "v_mul_f32 %[a27],%[al2],%[a27]\nv_fmac_f32 %[a27],%[bt2],%[vv]\n"
        "v_mul_f32 %[a37],%[al3],%[a37]\nv_fmac_f32 %[a37],%[bt3],%[vv]\n"
        :[rv0]"=&v"(rv0),[rv1]"=&v"(rv1),[rv2]"=&v"(rv2),[rv3]"=&v"(rv3),[rv4]"=&v"(rv4),[rv5]"=&v"(rv5),[rv6]"=&v"(rv6),[rv7]"=&v"(rv7),[al0]"=&v"(al0),[bt0]"=&v"(bt0),[al1]"=&v"(al1),[bt1]"=&v"(bt1),[al2]"=&v"(al2),[bt2]"=&v"(bt2),[al3]"=&v"(al3),[bt3]"=&v"(bt3),[mn]"=&v"(mn),[d0]"=&v"(d0),[d1]"=&v"(d1),[vv]"=&v"(vv),[m0]"+v"(m0),[l0]"+v"(l0),[m1]"+v"(m1),[l1]"+v"(l1),[m2]"+v"(m2),[l2]"+v"(l2),[m3]"+v"(m3),[l3]"+v"(l3),[a00]"+v"(a00),[a01]"+v"(a01),[a02]"+v"(a02),[a03]"+v"(a03),[a04]"+v"(a04),[a05]"+v"(a05),[a06]"+v"(a06),[a07]"+v"(a07),[a10]"+v"(a10),[a11]"+v"(a11),[a12]"+v"(a12),[a13]"+v"(a13),[a14]"+v"(a14),[a15]"+v"(a15),[a16]"+v"(a16),[a17]"+v"(a17),[a20]"+v"(a20),[a21]"+v"(a21),[a22]"+v"(a22),[a23]"+v"(a23),[a24]"+v"(a24),[a25]"+v"(a25),[a26]"+v"(a26),[a27]"+v"(a27),[a30]"+v"(a30),[a31]"+v"(a31),[a32]"+v"(a32),[a33]"+v"(a33),[a34]"+v"(a34),[a35]"+v"(a35),[a36]"+v"(a36),[a37]"+v"(a37):[ad0]"v"(ad0),[ad1]"v"(ad1),[ad2]"v"(ad2),[ad3]"v"(ad3),[ad4]"v"(ad4),[ad5]"v"(ad5),[ad6]"v"(ad6),[ad7]"v"(ad7),[sc0]"v"(sc0),[sc1]"v"(sc1),[sc2]"v"(sc2),[sc3]"v"(sc3),[log2e]"v"(log2e),[kvsc]"v"(kvsc):"memory");
}

__global__ __launch_bounds__(THR,1) __attribute__((amdgpu_flat_work_group_size(THR,THR)))
void mla_s1(const bf16*__restrict__ q,const uint8_t*__restrict__ kv,float kvsc,const int32_t*__restrict__ qoi,const int32_t*__restrict__ kvi,float*__restrict__ pm,float*__restrict__ pl,float*__restrict__ pv,bf16*__restrict__ out,int bs,int ns){
    int bid=blockIdx.x,b=bid/ns,sid=bid%ns;if(b>=bs)return;int tid=threadIdx.x,w=tid>>6,ln=tid&63,hb=w*HPW,col32=ln%32,grp=ln/32;
    int h_grp[4],h_sa[4];for(int i=0;i<4;++i){int h=hb+i;h_grp[i]=(h>>2)&1;h_sa[i]=(h&3)+((h>>3)<<2);}
    int qs_=qoi[b],qe_=qoi[b+1];if(qe_-qs_!=1)return;int qi=qs_,kvs=kvi[b],kve=kvi[b+1],kvl=kve-kvs;
    int ss=kvs+(kvl*sid)/ns,se=kvs+(kvl*(sid+1))/ns;
    extern __shared__ char smem[];uint8_t*ql=(uint8_t*)smem,*kl0=ql+QLDS,*kl1=kl0+KVLDS;

    // FUSED Q QUANTIZE: read bf16 Q from global, convert to fp8, write to LDS
    fused_q_to_lds(q, ql, qi, tid);
    __syncthreads();

    if(se<=ss){if(ns==1&&out){for(int j=0;j<VE;++j){int vd=ln+j*WS;out[ooff(qi,hb,vd)]=tob(0.f);out[ooff(qi,hb+1,vd)]=tob(0.f);out[ooff(qi,hb+2,vd)]=tob(0.f);out[ooff(qi,hb+3,vd)]=tob(0.f);}}else{if(ln==0)for(int i=0;i<4;++i){pm[poff(b,sid,hb+i,ns)]=-INFINITY;pl[poff(b,sid,hb+i,ns)]=0.f;}for(int j=0;j<VE;++j){int vd=ln+j*WS;pv[voff(b,sid,hb,vd,ns)]=0.f;pv[voff(b,sid,hb+1,vd,ns)]=0.f;pv[voff(b,sid,hb+2,vd,ns)]=0.f;pv[voff(b,sid,hb+3,vd,ns)]=0.f;}}return;}
    float sm0=-INFINITY,sl0=0.f,sm1=-INFINITY,sl1=0.f,sm2=-INFINITY,sl2=0.f,sm3=-INFINITY,sl3=0.f;
    float a00=0,a01=0,a02=0,a03=0,a04=0,a05=0,a06=0,a07=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0,a16=0,a17=0;
    float a20=0,a21=0,a22=0,a23=0,a24=0,a25=0,a26=0,a27=0,a30=0,a31=0,a32=0,a33=0,a34=0,a35=0,a36=0,a37=0;
    float log2e_c=LOG2E,kvsc_c=kvsc;int nt=(se-ss+KVT-1)/KVT;if(nt>0)skv(kv,kl0,ss,min(KVT,se-ss),tid);__syncthreads();
    int ping=0;for(int ti=0;ti<nt;++ti){int tb=ss+ti*KVT,rows=min(KVT,se-tb);uint8_t*cur=(ping==0)?kl0:kl1,*nxt=(ping==0)?kl1:kl0;bool hn=(ti+1<nt);if(hn)skv(kv,nxt,tb+KVT,min(KVT,se-tb-KVT),tid);
        f32x16_t sa=asm_mfma_qk(ql,cur,col32,grp,rows);__syncthreads();float cs=kvsc*SMSC;uint32_t cur_lds=(uint32_t)(cur-(uint8_t*)smem),base_ln=cur_lds+(uint32_t)ln;
        for(int t=0;t<rows;++t){float sc0=__shfl(sa[h_sa[0]],h_grp[0]*32+t)*cs,sc1=__shfl(sa[h_sa[1]],h_grp[1]*32+t)*cs,sc2=__shfl(sa[h_sa[2]],h_grp[2]*32+t)*cs,sc3=__shfl(sa[h_sa[3]],h_grp[3]*32+t)*cs;uint32_t toff=base_ln+(uint32_t)(t*DQK);asm_vaccum(toff,toff+64,toff+128,toff+192,toff+256,toff+320,toff+384,toff+448,sc0,sc1,sc2,sc3,sm0,sl0,sm1,sl1,sm2,sl2,sm3,sl3,a00,a01,a02,a03,a04,a05,a06,a07,a10,a11,a12,a13,a14,a15,a16,a17,a20,a21,a22,a23,a24,a25,a26,a27,a30,a31,a32,a33,a34,a35,a36,a37,log2e_c,kvsc_c);}
        if(hn){asm volatile("s_waitcnt vmcnt(0)":::"memory");__syncthreads();}ping^=1;}
    if(ns==1&&out){float il0=1.f/sl0,il1=1.f/sl1,il2=1.f/sl2,il3=1.f/sl3;
        #define WO(H,J,A,IL) out[ooff(qi,hb+H,ln+J*WS)]=tob(A*IL)
        WO(0,0,a00,il0);WO(0,1,a01,il0);WO(0,2,a02,il0);WO(0,3,a03,il0);WO(0,4,a04,il0);WO(0,5,a05,il0);WO(0,6,a06,il0);WO(0,7,a07,il0);WO(1,0,a10,il1);WO(1,1,a11,il1);WO(1,2,a12,il1);WO(1,3,a13,il1);WO(1,4,a14,il1);WO(1,5,a15,il1);WO(1,6,a16,il1);WO(1,7,a17,il1);WO(2,0,a20,il2);WO(2,1,a21,il2);WO(2,2,a22,il2);WO(2,3,a23,il2);WO(2,4,a24,il2);WO(2,5,a25,il2);WO(2,6,a26,il2);WO(2,7,a27,il2);WO(3,0,a30,il3);WO(3,1,a31,il3);WO(3,2,a32,il3);WO(3,3,a33,il3);WO(3,4,a34,il3);WO(3,5,a35,il3);WO(3,6,a36,il3);WO(3,7,a37,il3);
        #undef WO
    }else{if(ln==0){pm[poff(b,sid,hb,ns)]=sm0;pm[poff(b,sid,hb+1,ns)]=sm1;pm[poff(b,sid,hb+2,ns)]=sm2;pm[poff(b,sid,hb+3,ns)]=sm3;pl[poff(b,sid,hb,ns)]=sl0;pl[poff(b,sid,hb+1,ns)]=sl1;pl[poff(b,sid,hb+2,ns)]=sl2;pl[poff(b,sid,hb+3,ns)]=sl3;}
        #define WP(H,J,A) pv[voff(b,sid,hb+H,ln+J*WS,ns)]=A
        WP(0,0,a00);WP(0,1,a01);WP(0,2,a02);WP(0,3,a03);WP(0,4,a04);WP(0,5,a05);WP(0,6,a06);WP(0,7,a07);WP(1,0,a10);WP(1,1,a11);WP(1,2,a12);WP(1,3,a13);WP(1,4,a14);WP(1,5,a15);WP(1,6,a16);WP(1,7,a17);WP(2,0,a20);WP(2,1,a21);WP(2,2,a22);WP(2,3,a23);WP(2,4,a24);WP(2,5,a25);WP(2,6,a26);WP(2,7,a27);WP(3,0,a30);WP(3,1,a31);WP(3,2,a32);WP(3,3,a33);WP(3,4,a34);WP(3,5,a35);WP(3,6,a36);WP(3,7,a37);
        #undef WP
}}

void launch_all(torch::Tensor q,torch::Tensor kv,float sc,torch::Tensor qoi,torch::Tensor kvi,torch::Tensor pm,torch::Tensor pl,torch::Tensor pv,torch::Tensor out,int bs,int ns){
    // NO separate quantize_q_kernel! Q quantization is fused into mla_s1.
    if(ns==1){hipLaunchKernelGGL(mla_s1,dim3(bs),dim3(THR),SMEM,0,(const bf16*)q.data_ptr<at::BFloat16>(),(const uint8_t*)kv.data_ptr(),sc,qoi.data_ptr<int32_t>(),kvi.data_ptr<int32_t>(),pm.data_ptr<float>(),pl.data_ptr<float>(),pv.data_ptr<float>(),(bf16*)out.data_ptr<at::BFloat16>(),bs,1);
    }else{hipLaunchKernelGGL(mla_s1,dim3(bs*ns),dim3(THR),SMEM,0,(const bf16*)q.data_ptr<at::BFloat16>(),(const uint8_t*)kv.data_ptr(),sc,qoi.data_ptr<int32_t>(),kvi.data_ptr<int32_t>(),pm.data_ptr<float>(),pl.data_ptr<float>(),pv.data_ptr<float>(),(bf16*)nullptr,bs,ns);
        hipLaunchKernelGGL(mla_s2,dim3(bs),dim3(THR),0,0,pm.data_ptr<float>(),pl.data_ptr<float>(),pv.data_ptr<float>(),(bf16*)out.data_ptr<at::BFloat16>(),qoi.data_ptr<int32_t>(),bs,ns);}}
} // namespace mla
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("launch_all",&mla::launch_all);}
"""

@lru_cache(maxsize=1)
def _ext():
    return load_inline(name="mla_v60",cpp_sources="",cuda_sources=HIP_SRC,
        functions=None,extra_cuda_cflags=["-O3","--offload-arch=gfx950"],with_cuda=True,verbose=False)

NUM_CUS=304
def _ns(bs,kvl):
    nc=max(1,304//bs);nc=min(nc,kvl//32)if kvl>=32 else 1
    nk=max(1,kvl//512);n=max(nc,nk);p=1
    while p*2<=n:p*=2
    return max(1,min(p,128))

_alloc_cache={}
def _get_bufs(bs,ns,device):
    k=(bs,ns,str(device))
    if k not in _alloc_cache:
        _alloc_cache[k]=(
            torch.empty((bs,ns,NUM_HEADS),dtype=torch.float32,device=device),
            torch.empty((bs,ns,NUM_HEADS),dtype=torch.float32,device=device),
            torch.empty((bs,ns,NUM_HEADS,V_HEAD_DIM),dtype=torch.float32,device=device),
            torch.empty((bs,NUM_HEADS,V_HEAD_DIM),dtype=torch.bfloat16,device=device),
        )
    return _alloc_cache[k]

def custom_kernel(data):
    q,kv_data,qo_indptr,kv_indptr,config=data
    q=q.contiguous();qo_indptr=qo_indptr.contiguous();kv_indptr=kv_indptr.contiguous()
    bs=int(config["batch_size"]);kvl=int(config["kv_seq_len"])
    ns=_ns(bs,kvl);ext=_ext()
    fp8_data=kv_data["fp8"];kv_fp8=fp8_data[0].contiguous()
    if kv_fp8.dim()==3:kv_fp8=kv_fp8.squeeze(1).contiguous()
    sc=float(fp8_data[1].item())if fp8_data[1].numel()==1 else float(fp8_data[1].flatten()[0].item())
    pm,pl,pv,out=_get_bufs(bs,ns,q.device)
    ext.launch_all(q,kv_fp8,sc,qo_indptr,kv_indptr,pm,pl,pv,out,bs,ns)
    return out
