// One thread per measured training pixel. Keep the CPU group and tie order.
#include <cuda_runtime.h>
#include <cmath>

__device__ bool same_view(const double *poses, int a, int b, const double *p) {
    const double *ca = poses + 16*a, *cb = poses + 16*b;
    double separation = 0., dot = 0., aa = 0., bb = 0.;
    for (int axis = 0; axis < 3; ++axis) {
        double x = ca[4*axis+3], y = cb[4*axis+3];
        separation += (x-y)*(x-y);
        x -= p[axis];
        y -= p[axis];
        dot += x*y;
        aa += x*x;
        bb += y*y;
    }
    return sqrt(separation) < .1 || dot / sqrt(aa*bb) > 0.9986295347545738;
}

__global__ void weights(int n, const double *points, int source, int views,
    const int *group, const double *poses, const unsigned short *depth,
    const float *edges, int h, int w, double fx, double fy, double cx, double cy,
    const double *terms, int score_first, int *result, double *residual) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double *p = points + 3*i;
    short accepted[256];
    int used = 0, support = 0, conflict = 0, held = 0;
    double error_sum = 0.;
    for (int pass=0; pass <= score_first; ++pass) for (int j = 0; j < views;) {
        int g = group[j], best = -1, category = 0;
        double best_edge = -1., best_error = INFINITY;
        do {
            int v = j++;
            if (g == group[source]) continue;
            if (score_first && (g%5 == group[source]%5) != (pass==1)) continue;
            const double *a = poses + 16*v;
            double x=p[0]-a[3], y=p[1]-a[7], z=p[2]-a[11];
            double Z=x*a[2]+y*a[6]+z*a[10];
            if (Z < .3 || Z > 6.) continue;
            int u=__double2int_rn(fx*(x*a[0]+y*a[4]+z*a[8])/Z+cx);
            int t=__double2int_rn(fy*(x*a[1]+y*a[5]+z*a[9])/Z+cy);
            if (u<0 || u>=w || t<0 || t>=h) continue;
            size_t idx=((size_t)v*h+t)*w+u;
            int mm=depth[idx];
            float distance=edges[idx];
            if (distance<=2.f || mm<300 || mm>6000) continue;
            double measured=(float)mm/1000.f;
            int b=(mm>=1000)+(mm>=2000)+(mm>=4000);
            double tol=fmin(.20,fmax(terms[b],(double)fmaxf(.05f,.03f*(float)measured)));
            double err=fabs(Z-measured);
            if (distance>best_edge || (distance==best_edge && err<best_error)) {
                best=v; best_edge=distance; best_error=err;
                category=err<=tol ? 1 : (Z<measured ? 2 : 3);
            }
        } while (j<views && group[j]==g);
        if (best<0 || same_view(poses, best, source, p)) continue;
        bool duplicate=false;
        for (int k=0; k<used; ++k) {
            if (same_view(poses, best, accepted[k], p)) { duplicate=true; break; }
        }
        if (duplicate) continue;
        accepted[used++]=best;
        if (g%5 == group[source]%5) {
            if (category!=3) { ++held; error_sum+=best_error; }
        } else {
            support+=(category==1); conflict+=(category==2);
        }
    }
    result[3*i]=support; result[3*i+1]=conflict; result[3*i+2]=held;
    residual[i]=error_sum;
}

extern "C" int weight_votes(int n, const double *p, int source, int nv,
    const int *groups, const double *poses, const unsigned short *depth, const float *edge,
    int h, int w, double fx, double fy, double cx, double cy, const double *terms,
    int score_first, int *out, double *error) {
    weights<<<(n+127)/128,128>>>(n,p,source,nv,groups,poses,depth,edge,h,w,fx,fy,cx,cy,terms,score_first,out,error);
    return (int)cudaDeviceSynchronize();
}
