// Compute and accumulate voxel flux contributions for a particle track.
//
// Parameters:
//  - po: starting particle position in normalized domain coordinates [0,1]^3
//  - pd: ending particle position in normalized domain coordinates [0,1]^3
//  - d_tally: device pointer to flattened 3D voxel flux tally array
//  - voxelSize: physical size of each voxel (in domain units)
//
// Preconditions:
//  - Both po and pd are inside the unit cube [0,1]^3.
//  - `d_tally` has enough entries for the full voxel grid.
//  - Direction from po to pd is nonzero.
//
// Behavior:
//  - Trace the segment from po to pd through the Cartesian voxel grid.
//  - Add path-length contribution to each traversed voxel cell in d_tally.

#include "helper_math.h"

inline __device__ int getsign(const float f) {
    return (int)copysignf(1.0f, f);
}

inline __device__ int3 signOfDir(const float3 dir) {
    return make_int3(getsign(dir.x), getsign(dir.y), getsign(dir.z));
}

inline __device__ void flux (float3 r0, float3 rf, double* grid, uint gridSize, double voxelSize) {
    
    float3 dir = normalize(rf - r0);
    float3 delta = voxelSize / fabs(dir);
    uint3 n = make_uint3(
        (uint)floorf(r0.x / voxelSize),
        (uint)floorf(r0.y / voxelSize),
        (uint)floorf(r0.z / voxelSize)
    );

    int3 sign = signOfDir(dir);

    int3 step = make_int3(max(sign.x, 0), max(sign.y, 0), max(sign.z, 0));
    float3 t = (make_float3(n + make_uint3(step)) * voxelSize - r0) / dir;

    float tMax = length(rf - r0);
    float tCur = 0.0f;

    while (tCur < tMax) {
        float tNext = fminf(t.x, fminf(t.y, t.z));
        float segment = fminf(tNext, tMax) - tCur;
        int idx = n.x + n.y * gridSize + n.z * gridSize * gridSize;
        atomicAdd(&grid[idx], (double)segment);

        tCur = tNext;
        int stepX = (tNext == t.x);
        int stepY = (tNext == t.y);
        int stepZ = (tNext == t.z);

        n.x += stepX * sign.x;
        t.x += stepX * delta.x;

        n.y += stepY * sign.y;
        t.y += stepY * delta.y;

        n.z += stepZ * sign.z;
        t.z += stepZ * delta.z;

    }

}
