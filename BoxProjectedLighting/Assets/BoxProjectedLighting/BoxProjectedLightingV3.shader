/*
* [BoxProjectedLightingV3]
*
* This version is again, a little more advanced than the last.
*
* It's an extension of the second version...
* - We calculate direct lighting with a cubemap that represents the shadow mask. (just like the arm article)
* - We also calculate the first bounce from the direct lighting for GI just like in the second version.
* - Now in this version we calculate a first bounce enviornment lighting term.
*
* For sampling once again, a cosine-weighted hemisphere oriented with the surface normal is used for better ray convergence.
* 
* Ideally, more work should be done to improve quality at lower samples (we have access to mips)
*/

Shader "Unlit/BoxProjectedLightingV3"
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
        [IntRange] _BounceSamples("Bounce Samples", Range(1, 512)) = 64

        [Header(Environment Lighting)]
        [Toggle(_ENVIORMENT_LIGHT)] _EnviormentLighting("Enviorment Lighting", Float) = 1
        [IntRange] _EnviormentSamples("Enviorment Samples", Range(1, 512)) = 64

        [Header(Rendering)]
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
            float _DirectLightingShadowSoftness;
            float _BounceSamples;
            float _EnviormentSamples;
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

            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SAMPLING FUNCTIONS |||||||||||||||||||||||||||||

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

                //instancing
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct vertexToFragment
            {
                float4 vertexCameraClipPosition : SV_POSITION;
                float4 vertexWorldPosition : TEXCOORD1;
                float3 vertexWorldNormal : TEXCOORD2;
                float4 screenPosition : TEXCOORD3;

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

                // The normal direction of a surface at the current fragment
                float3 vector_normalDirection = i.vertexWorldNormal.xyz;

                // The direction to the light at the current fragment (for a directional light)
                float3 vector_lightDirection = normalize(_WorldSpaceLightPos0.xyz);

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
                        float2 noise = float2(GenerateRandomFloat(vector_screenUV), GenerateRandomFloat(vector_screenUV));

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
                    finalColor += bounceLighting;
                #endif

                // Compute the environment lighting term.
                #if defined (_ENVIORMENT_LIGHT)
                    float3 environmentLighting = float3(0, 0, 0);

                    for (int i = 1; i <= _EnviormentSamples; i++)
                    {
                        // Get random values.
                        float2 noise = float2(GenerateRandomFloat(vector_screenUV), GenerateRandomFloat(vector_screenUV));

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

                    finalColor += environmentLighting;
                #endif

                return float4(finalColor, 1);
            }

            ENDCG
        }
    }
}
