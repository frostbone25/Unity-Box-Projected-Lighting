Shader "Unlit/BoxProjectedLightingV5"
{
    Properties
    {
        [HideInInspector] _Seed("_Seed", Float) = 0

        [Header(Mask)]
        [Toggle(_USE_CUBEMASK)] _UseCubeMask("Use Cube Mask", Float) = 0
        _CubeMask("Cube Mask", Cube) = "white" {}

        [Header(Direct Lighting)]
        [Toggle(_DIRECT_LIGHT)] _DirectLighting("Direct Lighting", Float) = 1
        _DirectLightingShadowSoftness("Direct Lighting Shadow Softness", Float) = 0

        [Header(Bounce Lighting)]
        [Toggle(_BOUNCE_LIGHT)] _BounceLighting("Bounce Lighting", Float) = 1
        [IntRange] _BounceSamples("Bounce Samples", Range(1, 512)) = 32

        [Header(Environment Lighting)]
        [Toggle(_ENVIORMENT_LIGHT)] _EnviormentLighting("Enviorment Lighting", Float) = 1
        [IntRange] _EnviormentSamples("Enviorment Samples", Range(1, 512)) = 32

        [Header(Reflections)]
        [Toggle(_REFLECTIONS)] _REFLECTIONS("Reflections", Float) = 0
        [IntRange] _ReflectionSamples("Reflection Samples", Range(1, 512)) = 32

        [Header(Material)]
        _Smoothness("Smoothness", Range(0, 1)) = 0
        _Metallic("Metallic", Range(0, 1)) = 0
        _Reflectance("Reflectance", Range(0, 1)) = 0.42

        [Header(Rendering)]
        _MipOffsetMultiplier("Mip Offset Multiplier", Float) = 0
        _MipOffsetBias("Mip Offset Bias", Float) = 0
        _JitterTexture("Jitter Texture", 2D) = "white" {}
        [Toggle(_ANIMATE_NOISE)] _AnimateNoise("Animate Noise", Float) = 0
    }
    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
        }

        Pass
        {
            Name "BoxProjectedLighting_ForwardBase"

            Tags
            {
                "LightMode" = "ForwardBase"
            }

            CGPROGRAM

            #pragma vertex vertex_forward_base
            #pragma fragment fragment_forward_base

            //||||||||||||||||||||||||||||| CUSTOM KEYWORDS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| CUSTOM KEYWORDS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| CUSTOM KEYWORDS |||||||||||||||||||||||||||||

            #pragma shader_feature_local _USE_CUBEMASK
            #pragma shader_feature_local _DIRECT_LIGHT
            #pragma shader_feature_local _BOUNCE_LIGHT
            #pragma shader_feature_local _ENVIORMENT_LIGHT
            #pragma shader_feature_local _REFLECTIONS
            #pragma shader_feature_local _ANIMATE_NOISE

            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||

            //BUILT IN RENDER PIPELINE
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"
            #include "UnityLightingCommon.cginc"
            #include "UnityStandardBRDF.cginc"

            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||

            samplerCUBE _CubeMask;
            sampler2D _JitterTexture;
            float4 _JitterTexture_TexelSize;

            float _DirectLightingShadowSoftness;
            float _MipOffsetMultiplier;
            float _MipOffsetBias;
            float _Smoothness;
            float _Metallic;
            float _Reflectance;
            float _BounceSamples;
            float _EnviormentSamples;
            float _ReflectionSamples;
            float _Seed;

            //||||||||||||||||||||||||||||| RANDOM FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| RANDOM FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| RANDOM FUNCTIONS |||||||||||||||||||||||||||||

            // A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
            uint JenkinsHash(uint x)
            {
                x += (x << 10u);
                x ^= (x >> 6u);
                x += (x << 3u);
                x ^= (x >> 11u);
                x += (x << 15u);
                return x;
            }

            // Compound versions of the hashing algorithm.
            uint JenkinsHash(uint2 v)
            {
                return JenkinsHash(v.x ^ JenkinsHash(v.y));
            }

            uint JenkinsHash(uint3 v)
            {
                return JenkinsHash(v.x ^ JenkinsHash(v.yz));
            }

            uint JenkinsHash(uint4 v)
            {
                return JenkinsHash(v.x ^ JenkinsHash(v.yzw));
            }

            // Construct a float with half-open range [0, 1) using low 23 bits.
            // All zeros yields 0, all ones yields the next smallest representable value below 1.
            float ConstructFloat(int m) {
                const int ieeeMantissa = 0x007FFFFF; // Binary FP32 mantissa bitmask
                const int ieeeOne = 0x3F800000; // 1.0 in FP32 IEEE

                m &= ieeeMantissa;                   // Keep only mantissa bits (fractional part)
                m |= ieeeOne;                        // Add fractional part to 1.0

                float  f = asfloat(m);               // Range [1, 2)
                return f - 1;                        // Range [0, 1)
            }

            float ConstructFloat(uint m)
            {
                return ConstructFloat(asint(m));
            }

            // Pseudo-random value in half-open range [0, 1). The distribution is reasonably uniform.
            // Ref: https://stackoverflow.com/a/17479300
            float GenerateHashedRandomFloat(uint x)
            {
                return ConstructFloat(JenkinsHash(x));
            }

            float GenerateHashedRandomFloat(uint2 v)
            {
                return ConstructFloat(JenkinsHash(v));
            }

            float GenerateHashedRandomFloat(uint3 v)
            {
                return ConstructFloat(JenkinsHash(v));
            }

            float GenerateHashedRandomFloat(uint4 v)
            {
                return ConstructFloat(JenkinsHash(v));
            }

            float GenerateRandomFloat(float2 screenUV)
            {
                #if defined (_ANIMATE_NOISE)
                    float time = unity_DeltaTime.y * _Time.y + _Seed;
                    _Seed += 1.0;
                    return GenerateHashedRandomFloat(uint3(screenUV * _ScreenParams.xy, time));
                #else
                    _Seed += 1.0;
                    return GenerateHashedRandomFloat(uint3(screenUV * _ScreenParams.xy, _Seed));
                #endif
            }

            //||||||||||||||||||||||||||||| HAMMERSLEY FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| HAMMERSLEY FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| HAMMERSLEY FUNCTIONS |||||||||||||||||||||||||||||

            // Ref: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
            uint ReverseBits32(uint bits)
            {
                #if (SHADER_TARGET >= 45)
                    return reversebits(bits);
                #else
                    bits = (bits << 16) | (bits >> 16);
                    bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
                    bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
                    bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
                    bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);
                    return bits;
                #endif
            }

            float VanDerCorputBase2(uint i)
            {
                return ReverseBits32(i) * rcp(4294967296.0); // 2^-32
            }

            float2 Hammersley2dSeq(uint i, uint sequenceLength)
            {
                return float2(float(i) / float(sequenceLength), VanDerCorputBase2(i));
            }

            //||||||||||||||||||||||||||||| OLD RANDOM FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| OLD RANDOM FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| OLD RANDOM FUNCTIONS |||||||||||||||||||||||||||||

            float hash(float2 p)  // replace this by something better
            {
                p = 50.0 * frac(p * 0.3183099 + float2(0.71, 0.113));
                return -1.0 + 2.0 * frac(p.x * p.y * (p.x + p.y));
            }

            float rand(float co) 
            { 
                return frac(sin(co * (91.3458)) * 47453.5453); 
            }

            float rand(float2 co) 
            { 
                return frac(sin(dot(co.xy, float2(12.9898, 78.233))) * 43758.5453); 
            }

            float rand(float3 co) 
            { 
                return rand(co.xy + rand(co.z)); 
            }

            //||||||||||||||||||||||||||||| PRECOMPUTED NOISE FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| PRECOMPUTED NOISE FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| PRECOMPUTED NOISE FUNCTIONS |||||||||||||||||||||||||||||

            float noise(float2 uv)
            {
                return tex2Dlod(_JitterTexture, float4(uv * _ScreenParams.xy * _JitterTexture_TexelSize.xy, 0, 0));
            }

            float SampleJitter(float2 screenUV, float3 vector_viewDirection)
            {
                float jitter = noise(screenUV + length(vector_viewDirection));
                jitter = jitter * 2.0f - 1.0f;

                return jitter;
            }

            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||

            // Only makes sense for Monte-Carlo integration.
            // Normalize by dividing by the total weight (or the number of samples) in the end.
            // Integrate[6*(u^2+v^2+1)^(-3/2), {u,-1,1},{v,-1,1}] = 4 * Pi
            // Ref: "Stupid Spherical Harmonics Tricks", p. 9.
            float ComputeCubemapTexelSolidAngle(float2 uv)
            {
                float u = uv.x, v = uv.y;
                return pow(1 + u * u + v * v, -1.5);
            }

            float3 randomSpherePoint(float3 rand)
            {
                float ang1 = (rand.x + 1.0) * UNITY_PI; // [-1..1) -> [0..2*PI)
                float u = rand.y; // [-1..1), cos and acos(2v-1) cancel each other out, so we arrive at [-1..1)
                float u2 = u * u;
                float sqrt1MinusU2 = sqrt(1.0 - u2);
                float x = sqrt1MinusU2 * cos(ang1);
                float y = sqrt1MinusU2 * sin(ang1);
                float z = u;
                return float3(x, y, z);
            }

            float3 randomHemispherePoint(float3 rand, float3 n)
            {
                float3 v = randomSpherePoint(rand);
                return v * sign(dot(v, n));
            }

            float3 randomCosineWeightedHemispherePoint(float3 rand, float3 n)
            {
                float r = rand.x * 0.5 + 0.5; // [-1..1) -> [0..1)
                float angle = (rand.y + 1.0) * UNITY_PI; // [-1..1] -> [0..2*PI)
                float sr = sqrt(r);
                float2 p = float2(sr * cos(angle), sr * sin(angle));
                /*
                 * Unproject disk point up onto hemisphere:
                 * 1.0 == sqrt(x*x + y*y + z*z) -> z = sqrt(1.0 - x*x - y*y)
                 */
                float3 ph = float3(p.xy, sqrt(1.0 - dot(p, p)));
                /*
                 * Compute some arbitrary tangent space for orienting
                 * our hemisphere 'ph' around the normal. We use the camera's up vector
                 * to have some fix reference vector over the whole screen.
                 */
                float3 tangent = normalize(rand);
                float3 bitangent = cross(tangent, n);
                tangent = cross(bitangent, n);

                /* Make our hemisphere orient around the normal. */
                return tangent * ph.x + bitangent * ph.y + n * ph.z;
            }

            // Rotation with angle (in radians) and axis
            float3x3 angleAxis3x3(float angle, float3 axis)
            {
                float c, s;
                sincos(angle, s, c);

                float t = 1 - c;
                float x = axis.x;
                float y = axis.y;
                float z = axis.z;

                return float3x3(
                    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
                    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
                    t * x * z - s * y, t * y * z + s * x, t * z * z + c
                    );
            }

            // Returns a random direction vector inside a cone
            // Angle defined in radians
            // Example: direction=(0,1,0) and angle=pi returns ([-1,1],[0,1],[-1,1])
            float3 getConeSample(float randSeed, float3 direction, float coneAngle)
            {
                float cosAngle = cos(coneAngle);

                // Generate points on the spherical cap around the north pole [1].
                // [1] See https://math.stackexchange.com/a/205589/81266
                float z = randSeed * (1.0f - cosAngle) + cosAngle;
                float phi = randSeed * 2.0f * UNITY_PI;

                float x = sqrt(1.0f - z * z) * cos(phi);
                float y = sqrt(1.0f - z * z) * sin(phi);
                float3 north = float3(0.f, 0.f, 1.f);

                // Find the rotation axis `u` and rotation angle `rot` [1]
                float3 axis = normalize(cross(north, normalize(direction)));
                float angle = acos(dot(normalize(direction), north));

                // Convert rotation axis and angle to 3x3 rotation matrix [2]
                float3x3 R = angleAxis3x3(angle, axis);

                return mul(R, float3(x, y, z));
            }

            // Cosine-weighted hemisphere sampling
            float3 ImportanceSampleHemisphere(float2 uv, float3 normal)
            {
                // Convert UV to spherical coordinates
                float phi = 2.0 * UNITY_PI * uv.x;
                float cosTheta = sqrt(1.0 - uv.y);
                float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

                // Spherical to Cartesian coordinates, assuming Y up
                float3 direction = float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);

                // Align with the surface normal
                float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
                float3 tangent = normalize(cross(up, normal));
                float3 bitangent = cross(normal, tangent);

                return normalize(tangent * direction.x + normal * direction.y + bitangent * direction.z);
            }

            float3 cosineSampleHemisphere(float3 n, float2 uv)
            {
                //float2 u = rand(seed);

                float r = sqrt(uv.x);
                float theta = 2.0 * UNITY_PI * uv.y;

                float3  B = normalize(cross(n, float3(0.0, 1.0, 1.0)));
                float3  T = cross(B, n);

                return normalize(r * sin(theta) * B + sqrt(1.0 - uv.x) * n + r * cos(theta) * T);
            }

            float3 cosineSampleHemisphere(float r1, float r2)
            {
                //float r1 = rnd(seed);
                //float r2 = rnd(seed);

                float3 dir;
                float r = sqrt(r1);
                float phi = 2.0 * UNITY_PI * r2;

                dir.x = r * cos(phi);
                dir.y = r * sin(phi);
                dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

                return dir;
            }

            // Assumes that (0 <= x <= Pi).
            float SinFromCos(float cosX)
            {
                return sqrt(saturate(1 - cosX * cosX));
            }

            // Transforms the unit vector from the spherical to the Cartesian (right-handed, Z up) coordinate.
            float3 SphericalToCartesian(float cosPhi, float sinPhi, float cosTheta)
            {
                float sinTheta = SinFromCos(cosTheta);

                return float3(float2(cosPhi, sinPhi) * sinTheta, cosTheta);
            }

            float3 SphericalToCartesian(float phi, float cosTheta)
            {
                float sinPhi, cosPhi;
                sincos(phi, sinPhi, cosPhi);

                return SphericalToCartesian(cosPhi, sinPhi, cosTheta);
            }

            float3 SampleSphereUniform(float u1, float u2)
            {
                float phi = UNITY_TWO_PI * u2;
                float cosTheta = 1.0 - 2.0 * u1;

                return SphericalToCartesian(phi, cosTheta);
            }

            // Cosine-weighted sampling without the tangent frame.
            // Ref: http://www.amietia.com/lambertnotangent.html
            float3 SampleHemisphereCosine(float u1, float u2, float3 normal)
            {
                // This function needs to used safenormalize because there is a probability
                // that the generated direction is the exact opposite of the normal and that would lead
                // to a nan vector otheriwse.
                float3 pointOnSphere = SampleSphereUniform(u1, u2);
                return normalize(normal + pointOnSphere);
            }

            // weightOverPdf return the weight (without the diffuseAlbedo term) over pdf. diffuseAlbedo term must be apply by the caller.
            void ImportanceSampleLambert(float2   u,
                float3x3 localToWorld,
                out float3   L,
                out float    NdotL,
                out float    weightOverPdf)
            {
#if 0
                float3 localL = SampleHemisphereCosine(u.x, u.y);

                NdotL = localL.z;

                L = mul(localL, localToWorld);
#else
                float3 N = localToWorld[2];

                L = SampleHemisphereCosine(u.x, u.y, N);
                NdotL = saturate(dot(N, L));
#endif

                // Importance sampling weight for each sample
                // pdf = N.L / PI
                // weight = fr * (N.L) with fr = diffuseAlbedo / PI
                // weight over pdf is:
                // weightOverPdf = (diffuseAlbedo / PI) * (N.L) / (N.L / PI)
                // weightOverPdf = diffuseAlbedo
                // diffuseAlbedo is apply outside the function

                weightOverPdf = 1.0;
            }

            //||||||||||||||||||||||||||||| UTILITY FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UTILITY FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UTILITY FUNCTIONS |||||||||||||||||||||||||||||

            //https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl
            uint BitFieldInsert(uint mask, uint src, uint dst)
            {
                return (src & mask) | (dst & ~mask);
            }

            float CopySign(float x, float s, bool ignoreNegZero = true)
            {
                if (ignoreNegZero)
                {
                    return (s >= 0) ? abs(x) : -abs(x);
                }
                else
                {
                    uint negZero = 0x80000000u;
                    uint signBit = negZero & asuint(s);
                    return asfloat(BitFieldInsert(negZero, signBit, asuint(x)));
                }
            }

            float FastSign(float s, bool ignoreNegZero = true)
            {
                return CopySign(1.0, s, ignoreNegZero);
            }

            //https://github.com/Unity-Technologies/Graphics/blob/master/Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonLighting.hlsl
            float3x3 GetLocalFrame(float3 localZ)
            {
                float x = localZ.x;
                float y = localZ.y;
                float z = localZ.z;
                float sz = FastSign(z);
                float a = 1 / (sz + z);
                float ya = y * a;
                float b = x * ya;
                float c = x * sz;

                float3 localX = float3(c * x * a - 1, sz * b, c);
                float3 localY = float3(b, y * ya - sz, y);

                return float3x3(localX, localY, localZ);
            }

            //||||||||||||||||||||||||||||| PBR SHADING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| PBR SHADING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| PBR SHADING FUNCTIONS |||||||||||||||||||||||||||||

            void SampleAnisoGGXVisibleNormal(float2 u,
                float3 V,
                float3x3 localToWorld,
                float roughnessX,
                float roughnessY,
                out float3 localV,
                out float3 localH,
                out float  VdotH)
            {
                localV = mul(V, transpose(localToWorld));

                // Construct an orthonormal basis around the stretched view direction
                float3 N = normalize(float3(roughnessX * localV.x, roughnessY * localV.y, localV.z));
                float3 T = (N.z < 0.9999) ? normalize(cross(float3(0, 0, 1), N)) : float3(1, 0, 0);
                float3 B = cross(N, T);

                // Compute a sample point with polar coordinates (r, phi)
                float r = sqrt(u.x);
                float phi = 2.0 * UNITY_PI * u.y;
                float t1 = r * cos(phi);
                float t2 = r * sin(phi);
                float s = 0.5 * (1.0 + N.z);
                t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

                // Reproject onto hemisphere
                localH = t1 * T + t2 * B + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * N;

                // Transform the normal back to the ellipsoid configuration
                localH = normalize(float3(roughnessX * localH.x, roughnessY * localH.y, max(0.0, localH.z)));

                VdotH = saturate(dot(localV, localH));
            }

            // GGX VNDF via importance sampling
            half3 ImportanceSampleGGX_VNDF(float2 random, half3 normalWS, half3 viewDirWS, half roughness, out bool valid)
            {
                half3x3 localToWorld = GetLocalFrame(normalWS);

                half VdotH;
                half3 localV, localH;
                SampleAnisoGGXVisibleNormal(random, viewDirWS, localToWorld, roughness, roughness, localV, localH, VdotH);

                // Compute the reflection direction
                half3 localL = 2.0 * VdotH * localH - localV;
                half3 outgoingDir = mul(localL, localToWorld);

                half NdotL = dot(normalWS, outgoingDir);

                valid = (NdotL >= 0.001);

                return outgoingDir;
            }

            //https://google.github.io/filament/Filament.html#materialsystem
            //u = (View Direction) dot (Half Direction)
            //f0 = Reflectance at normal incidence
            float3 F_Schlick(float u, float3 f0)
            {
                float f = pow(1.0 - u, 5.0);
                float3 f90 = saturate(50.0 * f0); // cheap luminance approximation
                return f0 + (f90 - f0) * f;
            }

            float3 SampleCube(float3 samplePosition, float mip)
            {
                #if defined (_USE_CUBEMASK)
                    return texCUBElod(_CubeMask, float4(samplePosition, mip));
                #else
                    return DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, samplePosition, mip), unity_SpecCube0_HDR);
                #endif
            }

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 texcoord : TEXCOORD0;
                float4 tangent : TANGENT;
                //float4 texcoord1 : TEXCOORD1;
                //float4 texcoord2 : TEXCOORD2;
                //float4 color : COLOR;

                //instancing
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct vertexToFragment
            {
                float4 vertexCameraClipPosition : SV_POSITION;
                float4 vertexWorldPosition : TEXCOORD1;
                float3 vertexWorldNormal : TEXCOORD2;
                float4 screenPosition : TEXCOORD3;
                //float4 color : TEXCOORD4;

                //instancing
                UNITY_VERTEX_OUTPUT_STEREO
            };

            vertexToFragment vertex_forward_base(appdata v)
            {
                vertexToFragment o;

                //instancing
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(vertexToFragment, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.vertexCameraClipPosition = UnityObjectToClipPos(v.vertex);
                o.vertexWorldPosition = mul(unity_ObjectToWorld, v.vertex);
                o.vertexWorldNormal = UnityObjectToWorldNormal(normalize(v.normal));
                o.screenPosition = UnityStereoTransformScreenSpaceTex(ComputeScreenPos(o.vertexCameraClipPosition));

                return o;
            }

            float4 fragment_forward_base(vertexToFragment i) : SV_Target
            {
                // Screen UVs (used for noise).
                float2 vector_screenUV = i.screenPosition.xy / i.screenPosition.w;

                // Object world position.
                float3 vector_worldPosition = i.vertexWorldPosition.xyz;

                // The position of the current pixel relative to the camera.
                float3 vector_viewPosition = _WorldSpaceCameraPos.xyz - vector_worldPosition;

                // The view direction of the current pixel.
                float3 vector_viewDirection = normalize(vector_viewPosition);

                // The normal direction of a surface at the current fragment
                float3 vector_normalDirection = i.vertexWorldNormal.xyz;

                // The direction to the light at the current fragment (for a directional light)
                float3 vector_lightDirection = normalize(_WorldSpaceLightPos0.xyz);

                // The reflection direction at the current pixel fragment
                float3 vector_reflectionDirection = reflect(-vector_viewDirection, vector_normalDirection);

                // Normal dot View Direction (for fresnel calculations later)
                float NdotV = abs(dot(vector_normalDirection, vector_viewDirection));

                // Cube max point in world space
                float3 vector_worldCubeMax = (unity_SpecCube0_BoxMax - vector_worldPosition);

                // Cube min point in world space
                float3 vector_worldCubeMin = (unity_SpecCube0_BoxMin - vector_worldPosition);

                // Looking only for intersections in the forward direction of the ray.
                float3 vector_largestRayParameters = max(vector_worldCubeMax / vector_lightDirection, vector_worldCubeMin / vector_lightDirection);

                // Smallest value of the ray parameters gives us the intersection.
                float distanceToIntersection = min(min(vector_largestRayParameters.x, vector_largestRayParameters.y), vector_largestRayParameters.z);

                // Find the position of the intersection point. (This is the point where the light intersects with the box)
                float3 vector_worldSpaceIntersectPosition = vector_worldPosition + vector_lightDirection * distanceToIntersection;

                // The final color that will accumulate all lighting.
                float3 finalColor = float3(0, 0, 0);

                // Compute the direct lighting term.
                #if defined (_DIRECT_LIGHT)
                    // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                    float3 vector_correctedPosition = vector_worldSpaceIntersectPosition - unity_SpecCube0_ProbePosition;

                    // Calculate a mip level which we use as an approximation for contact hardening shadows
                    float directLightSampleMipLevel = _DirectLightingShadowSoftness * distanceToIntersection / UNITY_SPECCUBE_LOD_STEPS;

                    // Sample direct lighting
                    float3 directLightSample = texCUBElod(_CubeMask, float4(vector_correctedPosition, directLightSampleMipLevel)).rgb;

                    // Multiply direct lighting by light color, and also an normal dot light product
                    finalColor += directLightSample * _LightColor0.rgb * max(0.0, dot(vector_normalDirection, vector_lightDirection));
                #endif

                // Compute the direct bounce lighting term.
                #if defined (_BOUNCE_LIGHT)
                    float3 bounceLighting = float3(0, 0, 0);

                    for (int i = 1; i <= _BounceSamples; i++)
                    {
                        // Get random values.
                        //float2 noise = float2(GenerateRandomFloat(vector_screenUV), GenerateRandomFloat(vector_screenUV));
                        //float2 noise = float2(SampleJitter(vector_screenUV, vector_viewDirection), SampleJitter(vector_screenUV, vector_viewDirection));
                        float2 noise = Hammersley2dSeq(i, _BounceSamples);

                        // Randomized ray direction (Importance Sample Lambert)
                        float3 vector_bounceRayDirection = SampleHemisphereCosine(noise.x, noise.y, vector_normalDirection);

                        // Looking only for intersections in the forward direction of the ray.
                        float3 vector_bounce_largestRayParameters = max(vector_worldCubeMax / vector_bounceRayDirection, vector_worldCubeMin / vector_bounceRayDirection);

                        // Smallest value of the ray parameters gives us the intersection.
                        float bounce_distanceToIntersection = min(min(vector_bounce_largestRayParameters.x, vector_bounce_largestRayParameters.y), vector_bounce_largestRayParameters.z);

                        // Find the position of the intersection point. (This is the point where the light intersects with the box)
                        float3 vector_bounce_worldSpaceIntersectPosition = vector_worldSpaceIntersectPosition + vector_bounceRayDirection * bounce_distanceToIntersection;

                        // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                        float3 vector_bounce_correctedPosition = vector_bounce_worldSpaceIntersectPosition - unity_SpecCube0_ProbePosition;

                        // Sample the direct light.
                        float3 bounceLightSample = texCUBElod(_CubeMask, float4(vector_bounce_correctedPosition, 0)).rgb;

                        // Multiply by the light color.
                        bounceLightSample *= _LightColor0.rgb;

                        // Apply the Inverse Square Falloff
                        // NOTE TO SELF: Added 1 because energy then appears brighter than what it actually should be. This needs to be investigated.
                        bounceLightSample /= (1 + bounce_distanceToIntersection * bounce_distanceToIntersection);

                        // Apply a local "NdotL" term with the ray direction.
                        bounceLightSample *= max(0.0, dot(vector_normalDirection, vector_bounceRayDirection));

                        // Accumulate our sample to the overall bounce lighting.
                        bounceLighting += bounceLightSample;
                    }

                    // Make sure our samples are averaged.
                    bounceLighting /= _BounceSamples;

                    // Add it to the final result.
                    finalColor += max(0.0, bounceLighting);
                #endif

                // Compute the environment lighting term.
                #if defined (_ENVIORMENT_LIGHT)
                    float3 environmentLighting = float3(0, 0, 0);

                    for (int i = 1; i <= _EnviormentSamples; i++)
                    {
                        // Get random values.
                        //float2 noise = float2(GenerateRandomFloat(vector_screenUV), GenerateRandomFloat(vector_screenUV));
                        //float2 noise = float2(SampleJitter(vector_screenUV, vector_viewDirection), SampleJitter(vector_screenUV, vector_viewDirection));
                        float2 noise = Hammersley2dSeq(i, _EnviormentSamples);

                        // Randomized ray direction (Importance Sample Lambert)
                        float3 vector_enviormentRayDirection = SampleHemisphereCosine(noise.x, noise.y, vector_normalDirection);

                        // Looking only for intersections in the forward direction of the ray.
                        float3 vector_enviorment_largestRayParameters = max(vector_worldCubeMax / vector_enviormentRayDirection, vector_worldCubeMin / vector_enviormentRayDirection);

                        // Smallest value of the ray parameters gives us the intersection.
                        float enviorment_distanceToIntersection = min(min(vector_enviorment_largestRayParameters.x, vector_enviorment_largestRayParameters.y), vector_enviorment_largestRayParameters.z);

                        // Find the position of the intersection point. (from the 
                        float3 vector_enviorment_worldSpaceIntersectPosition = vector_worldPosition + vector_enviormentRayDirection * enviorment_distanceToIntersection;

                        // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                        float3 vector_enviorment_correctedPosition = vector_enviorment_worldSpaceIntersectPosition - unity_SpecCube0_ProbePosition;

                        // Sample the enviornment light.
                        float3 environmentLightSample = SampleCube(vector_enviorment_correctedPosition, 0);

                        // Apply a local "NdotL" term with the ray direction.
                        environmentLightSample *= max(0.0, dot(vector_normalDirection, vector_enviormentRayDirection));

                        // Accumulate our sample to the overall environment lighting.
                        environmentLighting += environmentLightSample;
                    }

                    environmentLighting /= _EnviormentSamples;

                    finalColor += max(0.0, environmentLighting);
                #endif

                // Compute the reflection term.
                #if defined (_REFLECTIONS)
                    // Calculate our PBR material parameters (based on filamented)
                    float smoothness = _Smoothness;
                    float perceptualRoughness = 1.0 - smoothness;
                    float roughness = perceptualRoughness * perceptualRoughness;
                    float metallic = _Metallic;
                    float reflectance = _Reflectance;

                    // Specular reflectance at normal incidence angle.
                    // In english... this basically means how reflective a material is when looking at it directly.
                    // This term also uses the albedo color for metallics as part of their specularity color.
                    // NOTE: float3(1,1,1) is albedo.
                    float3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + float3(1, 1, 1) * metallic;

                    // Since the GGX function outputs a variable for valid samples, we can't calculate how many accumulated samples we have as easily...
                    // So we will just keep track of it manually.
                    int accumulatedSamples = 0;

                    // The final acculmulated reflection term.
                    float3 reflectionTerm = float3(0, 0, 0);

                    for (int i = 0; i < _ReflectionSamples; i++)
                    {
                        // Get random values.
                        //float2 noise = float2(GenerateRandomFloat(vector_screenUV), GenerateRandomFloat(vector_screenUV));
                        //float2 noise = float2(SampleJitter(vector_screenUV, vector_viewDirection), SampleJitter(vector_screenUV, vector_viewDirection));
                        float2 noise = Hammersley2dSeq(i, _ReflectionSamples);

                        bool valid;

                        // Compute the reflection according to GGX
                        half3 vector_reflectionDirection = ImportanceSampleGGX_VNDF(noise, vector_normalDirection, vector_viewDirection, roughness, valid);

                        // If the direction is valid...
                        if (valid)
                        {
                            // Project it against the box.
                            vector_reflectionDirection = BoxProjectedCubemapDirection(vector_reflectionDirection, vector_worldPosition, unity_SpecCube0_ProbePosition, unity_SpecCube0_BoxMin, unity_SpecCube0_BoxMax);

                            // Sample the enviornment reflection,
                            float3 enviormentReflectionSample = SampleCube(vector_reflectionDirection, 0);

                            // Accumulate the reflection sample,
                            reflectionTerm += enviormentReflectionSample;

                            // Accumulate the amount of samples manually.
                            accumulatedSamples++;
                        }
                    }

                    // Average out our accumulated samples.
                    reflectionTerm /= accumulatedSamples;

                    // According to filament, we also need to remap the albedo since metallic surfaces actually use the albedo color as part of their specular color.
                    // "finalColor" is albedo.
                    finalColor *= (1.0 - metallic);

                    // Compute a PBR Fresnel according to the defined material parameters.
                    float3 reflectionFresnel = F_Schlick(NdotV, f0);
                    reflectionTerm.rgb *= reflectionFresnel;

                    // Add it to the final color.
                    finalColor += max(0.0, reflectionTerm);
                #endif

                return float4(finalColor, 1);
            }

            ENDCG
        }
    }
}
