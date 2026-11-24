// Charlotte forward PBR shader (glTF metallic-roughness) with optional inline ray tracing.
// Compile with DXC: vs_6_0 / ps_6_0 (CHARLOTTE_RT=0) or ps_6_5 (CHARLOTTE_RT=1, DXR 1.1 RayQuery).

#ifndef CHARLOTTE_RT
#define CHARLOTTE_RT 0
#endif

#define MAX_LIGHTS 16
#define LIGHT_DIRECTIONAL 0
#define LIGHT_POINT 1
#define LIGHT_SPOT 2

#define FRAME_RT_SHADOWS 1
#define FRAME_RT_AO 2
#define DRAW_ALPHA_MASK 1

struct Light {
    float3 position;  float range;
    float3 direction; uint type;
    float3 color;     float intensity;
    float innerCos;   float outerCos; float2 _pad;
};

cbuffer Frame : register(b0) {
    float4x4 viewProj;
    float3 camPos;    float exposure;
    float3 skyColor;  float aoStrength;
    float3 groundColor; float aoRadius;
    float3 camForward; float fovTan;
    float3 camRight; float aspect;
    float3 camUp; float iblStrength;
    uint lightCount; uint frameFlags; uint aoSamples; uint shadowSamples;
    float envMipCount; float shadowSoftness; uint2 _pad0;
    Light lights[MAX_LIGHTS];
    float4 envSH[9];
};

cbuffer Draw : register(b1) {
    float4x4 model;
    float4 baseColorFactor;
    float3 emissiveFactor; float metallicFactor;
    float roughnessFactor; float normalScale; float occlusionStrength; float alphaCutoff;
    uint drawFlags; uint3 _pad1;
};

#if CHARLOTTE_RT
RaytracingAccelerationStructure gTLAS : register(t0);
#endif
Texture2D<float4> texBase      : register(t1);
Texture2D<float4> texMetalRough: register(t2);
Texture2D<float4> texNormal    : register(t3);
Texture2D<float4> texEmissive  : register(t4);
Texture2D<float4> texOcclusion : register(t5);
Texture2D<float4> envMap       : register(t6);
SamplerState samp : register(s0);
SamplerState envSampler : register(s1);

struct VSIn {
    float3 pos : POSITION;
    float3 nrm : NORMAL;
    float4 tan : TANGENT;
    float2 uv  : TEXCOORD0;
};

struct PSIn {
    float4 sv   : SV_Position;
    float3 wpos : TEXCOORD1;
    float3 nrm  : TEXCOORD2;
    float4 tan  : TEXCOORD3;
    float2 uv   : TEXCOORD0;
};

float3x3 normalMatrix(float3x3 model3x3) {
    // HLSL has no built-in inverse() intrinsic. The cofactor matrix is
    // proportional to inverse-transpose; normalizing later removes |det|.
    float3 c0 = cross(model3x3[1], model3x3[2]);
    float3 c1 = cross(model3x3[2], model3x3[0]);
    float3 c2 = cross(model3x3[0], model3x3[1]);
    float determinant = dot(model3x3[0], c0);
    float orientation = determinant < 0.0 ? -1.0 : 1.0;
    return float3x3(c0 * orientation, c1 * orientation, c2 * orientation);
}

PSIn VSMain(VSIn v) {
    PSIn o;
    float4 wp = mul(model, float4(v.pos, 1.0));
    o.wpos = wp.xyz;
    o.sv = mul(viewProj, wp);
    o.nrm = mul(normalMatrix((float3x3)model), v.nrm);
    o.tan = float4(mul((float3x3)model, v.tan.xyz), v.tan.w);
    o.uv = v.uv;
    return o;
}

static const float PI = 3.14159265359;

float D_GGX(float NdotH, float a) {
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float V_SmithGGX(float NdotV, float NdotL, float a) {
    float a2 = a * a;
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

float3 F_Schlick(float3 f0, float VdotH) {
    return f0 + (1.0 - f0) * pow(1.0 - VdotH, 5.0);
}

float3 tonemapACES(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

float2 directionToLatLong(float3 d) {
    return float2(atan2(d.z, d.x) * (0.5 / PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

float3 sampleEnvironment(float3 d, float lod) {
    float2 uv = directionToLatLong(normalize(d));
    uv.y = saturate(uv.y);
    return envMap.SampleLevel(envSampler, uv, lod).rgb;
}

float3 diffuseIrradianceSH(float3 n) {
    float x = n.x, y = n.y, z = n.z;
    return envSH[0].rgb * 0.886227
        + envSH[1].rgb * (1.023327 * y)
        + envSH[2].rgb * (1.023327 * z)
        + envSH[3].rgb * (1.023327 * x)
        + envSH[4].rgb * (0.858086 * x * y)
        + envSH[5].rgb * (0.858086 * y * z)
        + envSH[6].rgb * (0.247708 * (3.0 * y * y - 1.0))
        + envSH[7].rgb * (0.858086 * x * z)
        + envSH[8].rgb * (0.429043 * (x * x - z * z));
}

struct SkyVertexOut { float4 sv : SV_Position; float2 ndc : TEXCOORD0; };

SkyVertexOut SkyVS(uint vertexId : SV_VertexID) {
    SkyVertexOut o;
    float2 p = float2((vertexId << 1) & 2, vertexId & 2);
    o.ndc = p * 2.0 - 1.0;
    o.sv = float4(o.ndc, 0.0, 1.0);
    return o;
}

float4 SkyPS(SkyVertexOut i) : SV_Target {
    float3 ray = normalize(camForward + camRight * (i.ndc.x * aspect * fovTan) + camUp * (i.ndc.y * fovTan));
    return float4(tonemapACES(sampleEnvironment(ray, 0.0) * exposure), 1.0);
}

#if CHARLOTTE_RT
float traceVisibility(float3 origin, float3 dir, float tmax) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    RayDesc r;
    r.Origin = origin;
    r.Direction = dir;
    r.TMin = 0.001;
    r.TMax = tmax;
    q.TraceRayInline(gTLAS, 0, 0xFF, r);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0 : 1.0;
}

float radicalInverse(uint bits) {
    return float(reversebits(bits)) * (1.0 / 4294967296.0);
}

float interleavedGradientNoise(float2 pixel) {
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

float traceAOSample(float3 origin, float3 direction, float radius) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    RayDesc r;
    r.Origin = origin;
    r.Direction = direction;
    r.TMin = 0.001;
    r.TMax = radius;
    q.TraceRayInline(gTLAS, 0, 0xFF, r);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) return 1.0;
    // Distance weighting avoids turning every nearby triangle hit into a black binary dot.
    return smoothstep(0.0, radius, q.CommittedRayT());
}

float rtSoftShadow(float3 P, float3 N, float3 L, float tmax, uint2 pixel, uint lightIndex) {
    uint rays = clamp(shadowSamples, 1u, 16u);
    if (rays == 1u || shadowSoftness <= 0.00001) {
        float side = dot(N, L) >= 0.0 ? 1.0 : -1.0;
        return traceVisibility(P + N * side * 0.0025, L, tmax);
    }
    float3 helper = abs(L.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(helper, L));
    float3 B = cross(L, T);
    float vis = 0.0;
    float maxAngle = min(shadowSoftness, 0.25);
    float rotation = interleavedGradientNoise(float2(pixel) + float2(lightIndex * 17u, lightIndex * 29u));
    for (uint i = 0; i < rays; i++) {
        float u = (float(i) + 0.5) / float(rays);
        float phi = 2.0 * PI * frac(radicalInverse(i) + rotation);
        float cosTheta = lerp(1.0, cos(maxAngle), u);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float3 d = normalize(L * cosTheta + (T * cos(phi) + B * sin(phi)) * sinTheta);
        float side = dot(N, d) >= 0.0 ? 1.0 : -1.0;
        vis += traceVisibility(P + N * side * 0.0025, d, tmax);
    }
    return vis / rays;
}

float rtAmbientOcclusion(float3 P, float3 N, uint2 pixel) {
    float3 up = abs(N.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    float vis = 0.0;
    uint rays = clamp(aoSamples, 1u, 32u);
    float radius = max(aoRadius, 0.01);
    float rotation = interleavedGradientNoise(float2(pixel));
    for (uint i = 0; i < rays; i++) {
        float u1 = (float(i) + 0.5) / float(rays);
        float u2 = frac(radicalInverse(i) + rotation);
        float r = sqrt(u1);
        float phi = 2.0 * PI * u2;
        float3 d = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        float3 dir = normalize(T * d.x + B * d.y + N * d.z);
        vis += traceAOSample(P + N * 0.0025, dir, radius);
    }
    return pow(max(vis / rays, 0.0001), max(aoStrength, 0.0));
}
#endif

float4 PSMain(PSIn i, bool isFront : SV_IsFrontFace) : SV_Target {
    float4 baseSample = texBase.Sample(samp, i.uv);
    float4 baseColor = baseSample * baseColorFactor;
    if ((drawFlags & DRAW_ALPHA_MASK) != 0) {
        clip(baseColor.a - alphaCutoff);
    }

    float4 mr = texMetalRough.Sample(samp, i.uv);
    float roughness = clamp(mr.g * roughnessFactor, 0.045, 1.0);
    float metallic = saturate(mr.b * metallicFactor);
    float ao = lerp(1.0, texOcclusion.Sample(samp, i.uv).r, occlusionStrength);

    float3 N = normalize(i.nrm);
    if (!isFront) N = -N;
    float3 Ngeo = N;
    if (i.tan.w != 0.0) {
        float3 tn = texNormal.Sample(samp, i.uv).xyz * 2.0 - 1.0;
        tn.xy *= normalScale;
        float3 T = normalize(i.tan.xyz - N * dot(N, i.tan.xyz));
        float3 B = cross(N, T) * i.tan.w;
        N = normalize(tn.x * T + tn.y * B + tn.z * N);
    }

    float3 V = normalize(camPos - i.wpos);
    float NdotV = max(dot(N, V), 1e-4);
    float a = roughness * roughness;
    float3 diffuseColor = baseColor.rgb * (1.0 - metallic);
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);

    float3 color = 0;
    for (uint li = 0; li < lightCount; li++) {
        Light l = lights[li];
        float3 L;
        float atten = 1.0;
        float dist = 1e6;
        if (l.type == LIGHT_DIRECTIONAL) {
            L = -l.direction;
        } else {
            float3 toLight = l.position - i.wpos;
            dist = length(toLight);
            L = toLight / max(dist, 1e-4);
            atten = 1.0 / max(dist * dist, 1e-4);
            if (l.range > 0.0) {
                float r = dist / l.range;
                atten *= saturate(1.0 - r * r * r * r);
            }
            if (l.type == LIGHT_SPOT) {
                float cd = dot(l.direction, -L);
                float s = saturate((cd - l.outerCos) / max(l.innerCos - l.outerCos, 1e-4));
                atten *= s * s;
            }
        }
        float NdotL = saturate(dot(N, L));
        if (NdotL <= 0.0 || atten <= 0.0) continue;

        float visibility = 1.0;
#if CHARLOTTE_RT
        if ((frameFlags & FRAME_RT_SHADOWS) != 0) {
            float tmax = (l.type == LIGHT_DIRECTIONAL) ? 1e5 : max(dist - 0.02, 0.0);
            visibility = rtSoftShadow(i.wpos, Ngeo, L, tmax, (uint2)i.sv.xy, li);
        }
#endif
        float3 H = normalize(V + L);
        float NdotH = saturate(dot(N, H));
        float VdotH = saturate(dot(V, H));
        float3 F = F_Schlick(f0, VdotH);
        float3 spec = F * (D_GGX(NdotH, a) * V_SmithGGX(NdotV, NdotL, a));
        float3 diff = (1.0 - F) * diffuseColor / PI;
        color += (diff + spec) * l.color * l.intensity * atten * NdotL * visibility;
    }

    // Fallback hemisphere ambient for clients without an environment map.
    float3 hemi = lerp(groundColor, skyColor, N.y * 0.5 + 0.5);
    float3 R = reflect(-V, N);
    float3 hemiR = lerp(groundColor, skyColor, R.y * 0.5 + 0.5);
    float3 envF = f0 + (max(float3(1 - roughness, 1 - roughness, 1 - roughness), f0) - f0) * pow(1.0 - NdotV, 5.0);
    float aoRT = 1.0;
#if CHARLOTTE_RT
    if ((frameFlags & FRAME_RT_AO) != 0) {
        aoRT = rtAmbientOcclusion(i.wpos, Ngeo, (uint2)i.sv.xy);
    }
#endif
    if ((frameFlags & 4) != 0) {
        float3 diffuseIrradiance = diffuseIrradianceSH(N) / PI;
        float3 prefilteredReflection = sampleEnvironment(R, roughness * envMipCount);
        float specularAo = saturate(pow(saturate(NdotV + aoRT), exp2(-16.0 * roughness - 1.0)) - 1.0 + aoRT);
        color += (diffuseIrradiance * diffuseColor * ao * aoRT
            + prefilteredReflection * envF * specularAo) * iblStrength;
    } else {
        float specularAo = saturate(pow(saturate(NdotV + aoRT), exp2(-16.0 * roughness - 1.0)) - 1.0 + aoRT);
        color += hemi * diffuseColor * ao * aoRT
            + hemiR * envF * (1.0 - roughness * 0.7) * specularAo;
    }

    float3 emissive = emissiveFactor * texEmissive.Sample(samp, i.uv).rgb;
    color += emissive;

    color = tonemapACES(color * exposure);
    return float4(color, 1.0); // RTV is *_SRGB, hardware applies the sRGB curve.
}
