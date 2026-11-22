// Native-depth first-layer observations. All coordinates and thresholds are metric.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

__global__ void count_views(int n, const double *points, const int *sources,
    const int *views, int nviews, const int *group, const double *pose,
    const unsigned short *depth, const float *edge, int h, int w,
    double fx,double fy,double cx,double cy,const double *terms, int groups,
    int *counts, double *mean, int *witness) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const double *p=points+3*i;
    short independent[256];
    int used=0,s=0,c=0,occluded=0,observed=0,representative=-1;
    double sum=0,best_support=1e30;
    for(int j=0;j<nviews;) {
        int g=group[views[j]], best=-1,category=0;
        double best_edge=-1,best_error=1e30;
        do {
            int v=views[j++];
            if(g==sources[i]) continue;
            const double *a=pose+16*v;
            double x=p[0]-a[3],y=p[1]-a[7],z=p[2]-a[11];
            double Z=x*a[2]+y*a[6]+z*a[10];
            if(Z<0.3 || Z>6.) continue;
            int u=__double2int_rn(fx*(x*a[0]+y*a[4]+z*a[8])/Z+cx);
            int t=__double2int_rn(fy*(x*a[1]+y*a[5]+z*a[9])/Z+cy);
            if(u<0 || u>=w || t<0 || t>=h) continue;
            size_t k=((size_t)v*h+t)*w+u;
            int mm=depth[k];
            if(mm<300 || mm>6000 || edge[k]<=2.f) continue;
            double measured=(double)((float)mm/1000.f);
            int b=mm<1000?0:(mm<2000?1:(mm<4000?2:3));
            double tau=fmin(.20,fmax(terms[b],fmax(.05,.03*measured)));
            double delta=Z-measured, err=fabs(delta);
            if(edge[k]>best_edge || (edge[k]==best_edge && err<best_error)) {
                best=v; best_edge=edge[k]; best_error=err;
                category=err<=tau ? 1 : (delta<0 ? 2 : 3);
            }
        } while(j<nviews && group[views[j]]==g);
        if(best<0) continue;
        observed++;
        if(category==3){occluded++;continue;} // An occluder abstains and never suppresses a support vote.
        const double *a=pose+16*best;
        double x=a[3]-p[0],y=a[7]-p[1],z=a[11]-p[2],length=x*x+y*y+z*z;
        bool duplicate=false;
        for(int k=0;k<used;k++) {
            const double *b=pose+16*independent[k];
            double dx=a[3]-b[3],dy=a[7]-b[7],dz=a[11]-b[11];
            double bx=b[3]-p[0],by=b[7]-p[1],bz=b[11]-p[2];
            double dot=x*bx+y*by+z*bz;
            if(dx*dx+dy*dy+dz*dz<.01 ||
               (dot>0 && dot*dot>0.9972609476841366*length*(bx*bx+by*by+bz*bz))) {
                duplicate=true;break;
            }
        }
        if(duplicate) continue;
        independent[used++]=best;
        if(category==2){c++;continue;}
        s++;sum+=best_error;
        if(best_error<best_support){best_support=best_error;representative=best;}
    }
    counts[4*i]=s;counts[4*i+1]=c;counts[4*i+2]=occluded;
    counts[4*i+3]=groups-(sources[i]>=0)-observed;
    mean[i]=s?sum/s:INFINITY;witness[i]=representative;
}

extern "C" int votes(int n,const double* p,const int* src,const int* views,int nv,const int* groups,
    const double* poses,const unsigned short* depth,const float* edge,int h,int w,
    double fx,double fy,double cx,double cy,const double* terms,int ng,int* count,double* error,int* witness){
    if(ng>256)return -1;
    count_views<<<(n+127)/128,128>>>(n,p,src,views,nv,groups,poses,depth,edge,h,w,fx,fy,cx,cy,terms,ng,count,error,witness);
    return (int)cudaDeviceSynchronize();
}

// Faces arrive in deterministic order: cell, decreasing support, residual, owner, original face.
extern "C" void choose_owners(int n,const int64_t* cell,const double* centers,const double* normals,
                               const int* group,unsigned char* keep){
    std::vector<int> layers;
    for(int i=0;i<n;i++){
        if(!i || cell[i]!=cell[i-1]) layers.clear();
        int match=-1;
        for(int j:layers){
            const double *a=normals+3*i,*b=normals+3*j;
            if(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]<0.8660254037844386)continue;
            const double *u=centers+3*i,*v=centers+3*j;
            double dx=u[0]-v[0],dy=u[1]-v[1],dz=u[2]-v[2];
            if(fabs(dx*b[0]+dy*b[1]+dz*b[2])>.005 || fabs(dx*a[0]+dy*a[1]+dz*a[2])>.005)continue;
            match=j;break;
        }
        if(match<0){layers.push_back(i);keep[i]=1;}
        else keep[i]=group[i]==group[match];
    }
}
