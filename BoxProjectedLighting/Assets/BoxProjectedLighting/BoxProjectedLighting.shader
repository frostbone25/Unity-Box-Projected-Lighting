// Upgrade NOTE: replaced 'defined _REFLECTIONS' with 'defined (_REFLECTIONS)'

// Upgrade NOTE: replaced 'defined _USE_CUBEMASK' with 'defined (_USE_CUBEMASK)'

/*
* REFERENCES
* Based On - https://developer.arm.com/documentation/102179/0100/Dynamic-soft-shadows-based-on-local-cubemaps
* Random Functions - https://github.com/LWJGL/lwjgl3-demos/blob/main/res/org/lwjgl/demo/opengl/raytracing/randomCommon.glsl
*/

Shader "Unlit/BoxProjectedLighting"
{
    Properties
    {
        [Header(Mask)]
        [Toggle(_USE_CUBEMASK)] _UseCubeMask("Use Cube Mask", Float) = 0
        _CubeMask("Cube Mask", Cube) = "white" {}

        [Toggle(_DIRECT_LIGHT)] _DirectLighting("Direct Lighting", Float) = 1
        [Toggle(_BOUNCE_LIGHT)] _BounceLighting("Bounce Lighting", Float) = 1
        [Toggle(_ENVIORMENT_LIGHT)] _EnviormentLighting("Enviorment Lighting", Float) = 1
        [Toggle(_REFLECTIONS)] _REFLECTIONS("Reflections", Float) = 0

        _DirectLightingShadowSoftness("Direct Lighting Shadow Softness", Float) = 0
        _EnviormentMipOffset("Enviornment Mip Offset", Float) = 0
        _ReflectionAngle("ReflectionAngle", Range(0, 1)) = 0
        _JitterTexture("Jitter Texture", 2D) = "white" {}
        [IntRange] _BounceSamples("Bounce Samples", Range(1, 256)) = 32
        [IntRange] _EnviormentSamples("Enviorment Samples", Range(1, 256)) = 32
        [IntRange] _ReflectionSamples("Reflection Samples", Range(1, 256)) = 32
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

            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||

            //BUILT IN RENDER PIPELINE
            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "AutoLight.cginc"
            #include "UnityLightingCommon.cginc"
            #include "UnityStandardBRDF.cginc"

            #pragma vertex vertex_base
            #pragma fragment fragment_forward_base

            #pragma shader_feature_local _USE_CUBEMASK
            #pragma shader_feature_local _DIRECT_LIGHT
            #pragma shader_feature_local _BOUNCE_LIGHT
            #pragma shader_feature_local _ENVIORMENT_LIGHT
            #pragma shader_feature_local _REFLECTIONS

            samplerCUBE _CubeMask;
            float _DirectLightingShadowSoftness;
            float _EnviormentMipOffset;
            float _ReflectionAngle;

            float _BounceSamples;
            float _EnviormentSamples;
            float _ReflectionSamples;

            sampler2D _JitterTexture;
            float4 _JitterTexture_TexelSize;

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
                float4 pos : SV_POSITION;
                float4 posWorld : TEXCOORD1;
                float3 worldNormal : TEXCOORD2;
                float4 screenPos : TEXCOORD3;
                //float4 color : TEXCOORD4;

                //instancing
                UNITY_VERTEX_OUTPUT_STEREO
            };

            float hash(float2 p)  // replace this by something better
            {
                p = 50.0 * frac(p * 0.3183099 + float2(0.71, 0.113));
                return -1.0 + 2.0 * frac(p.x * p.y * (p.x + p.y));
            }

            float rand(float co) { return frac(sin(co * (91.3458)) * 47453.5453); }
            float rand(float2 co) { return frac(sin(dot(co.xy, float2(12.9898, 78.233))) * 43758.5453); }
            float rand(float3 co) { return rand(co.xy + rand(co.z)); }

            float noise(float2 uv)
            {
                return tex2Dlod(_JitterTexture, float4(uv * _ScreenParams.xy * _JitterTexture_TexelSize.xy, 0, 0));
            }

            float SampleJitter(float2 screenUV, float3 vector_worldPosition)
            {
                float3 localViewDir = normalize(UnityWorldSpaceViewDir(vector_worldPosition));
                float jitter = noise(screenUV + length(localViewDir));
                jitter = jitter * 2.0f - 1.0f;

                return jitter;
            }

            inline float3 UnityBoxProjectedCubemapDirection(float3 worldRefl, float3 worldPos, float4 cubemapCenter, float4 boxMin, float4 boxMax, out float distanceToHitPoint)
            {
                float3 nrdir = normalize(worldRefl);
                float3 rbmax = (boxMax.xyz - worldPos) / nrdir;
                float3 rbmin = (boxMin.xyz - worldPos) / nrdir;
                float3 rbminmax = (nrdir > 0.0f) ? rbmax : rbmin;

                distanceToHitPoint = min(min(rbminmax.x, rbminmax.y), rbminmax.z);

                worldPos -= cubemapCenter.xyz;
                worldRefl = worldPos + nrdir * distanceToHitPoint;

                return worldRefl;
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

            vertexToFragment vertex_base(appdata v)
            {
                vertexToFragment o;

                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(vertexToFragment, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);

                o.pos = UnityObjectToClipPos(v.vertex);
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(normalize(v.normal));
                o.screenPos = UnityStereoTransformScreenSpaceTex(ComputeScreenPos(o.pos));

                return o;
            }

            float3 SampleCube(float3 samplePosition, float mip)
            {
#if defined (_USE_CUBEMASK)
                return texCUBElod(_CubeMask, float4(samplePosition, mip));
#else
                return DecodeHDR(UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, samplePosition, mip), unity_SpecCube0_HDR);
#endif
            }

            float4 fragment_forward_base(vertexToFragment i) : SV_Target
            {
                float2 screenUV = i.screenPos.xy / i.screenPos.w;
                float3 vector_worldPosition = i.posWorld.xyz; //object world position
                float3 vector_viewPosition = _WorldSpaceCameraPos.xyz - vector_worldPosition; //object view position
                float3 vector_viewDirection = normalize(vector_viewPosition); //view direction
                float3 vector_normalDirection = normalize(i.worldNormal.xyz); //the normal direction of the object we are currently on
                float3 vector_lightDirection = normalize(_WorldSpaceLightPos0.xyz); //the direction to the light at the current fragment (for a directional light)
                float3 vector_reflectionDirection = reflect(-vector_viewDirection, vector_normalDirection); //reflection direction at the current object fragment

                float3 directLighting = float3(0, 0, 0);
                float3 bounceLighting = float3(0, 0, 0);
                float3 skyLighting = float3(0, 0, 0);
                float3 reflection = float3(0, 0, 0);

                float3 vector_worldCubeMax = (unity_SpecCube0_BoxMax - vector_worldPosition);
                float3 vector_worldCubeMin = (unity_SpecCube0_BoxMin - vector_worldPosition);

#if defined (_DIRECT_LIGHT) || (_BOUNCE_LIGHT)
                // Looking only for intersections in the forward direction of the ray.
                float3 largestRayParams = max(vector_worldCubeMax / vector_lightDirection, vector_worldCubeMin / vector_lightDirection);

                // Smallest value of the ray parameters gives us the intersection.
                float dist = min(min(largestRayParams.x, largestRayParams.y), largestRayParams.z);

                // Find the position of the intersection point. (from the 
                float3 intersectPositionWS = vector_worldPosition + vector_lightDirection * dist;

#if defined (_DIRECT_LIGHT)
                // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                float3 correctedPosition = intersectPositionWS - unity_SpecCube0_ProbePosition;

                // Sample direct lighting
                float3 directLightSample = texCUBElod(_CubeMask, float4(correctedPosition, _DirectLightingShadowSoftness * dist / UNITY_SPECCUBE_LOD_STEPS)).rgb;

                // Multiply direct lighting by light color, and also an normal dot light product
                directLighting += directLightSample * _LightColor0.rgb * max(0.0, dot(vector_normalDirection, vector_lightDirection));
#endif

#if defined (_BOUNCE_LIGHT)
                for (int i = 1; i <= _BounceSamples; i++)
                {
                    float3 rayDirection = randomHemispherePoint(float3(SampleJitter(screenUV + i, vector_worldPosition), SampleJitter(screenUV - i, vector_worldPosition), SampleJitter(screenUV - (i * _JitterTexture_TexelSize), vector_worldPosition)), vector_normalDirection);

                    // Looking only for intersections in the forward direction of the ray.
                    float3 bounce_largestRayParams = max(vector_worldCubeMax / rayDirection, vector_worldCubeMin / rayDirection);

                    // Smallest value of the ray parameters gives us the intersection.
                    float bounce_dist = min(min(bounce_largestRayParams.x, bounce_largestRayParams.y), bounce_largestRayParams.z);

                    float3 bounce_intersectPositionWS = intersectPositionWS + rayDirection * bounce_dist;

                    // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                    float3 bounce_correctedPosition = bounce_intersectPositionWS - unity_SpecCube0_ProbePosition;

                    float3 bounceLightSample = texCUBElod(_CubeMask, float4(bounce_correctedPosition, _DirectLightingShadowSoftness * bounce_dist / UNITY_SPECCUBE_LOD_STEPS)).rgb;
                    //float3 bounceLightSample = SampleCube(bounce_correctedPosition, _DirectLightingShadowSoftness * bounce_dist / UNITY_SPECCUBE_LOD_STEPS);
                    bounceLightSample *= _LightColor0.rgb * 0.5f;
                    bounceLightSample *= max(0.0, dot(vector_normalDirection, rayDirection));

                    bounceLighting += bounceLightSample;
                }

                bounceLighting /= _BounceSamples;
#endif

#endif

#if defined (_ENVIORMENT_LIGHT)
                for (int i = 1; i <= _EnviormentSamples; i++)
                {
                    float3 rayDirection = randomHemispherePoint(float3(SampleJitter(screenUV + i, vector_worldPosition), SampleJitter(screenUV - i, vector_worldPosition), SampleJitter(screenUV, vector_worldPosition)), vector_normalDirection);

                    // Looking only for intersections in the forward direction of the ray.
                    float3 sky_largestRayParams = max(vector_worldCubeMax / rayDirection, vector_worldCubeMin / rayDirection);

                    // Smallest value of the ray parameters gives us the intersection.
                    float sky_dist = min(min(sky_largestRayParams.x, sky_largestRayParams.y), sky_largestRayParams.z);

                    // Find the position of the intersection point. (from the 
                    float3 sky_intersectPositionWS = vector_worldPosition + rayDirection * sky_dist;

                    // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                    float3 sky_correctedPosition = sky_intersectPositionWS - unity_SpecCube0_ProbePosition;

                    float3 skyLightSample = SampleCube(sky_correctedPosition, _EnviormentMipOffset);
                    skyLightSample *= max(0.0, dot(vector_normalDirection, rayDirection));

                    skyLighting += skyLightSample;
                }

                skyLighting /= _EnviormentSamples;
#endif

#if defined (_REFLECTIONS)
                for (int i = 1; i <= _ReflectionSamples; i++)
                {
                    //float3 rayDirection = randomHemispherePoint(float3(SampleJitter(screenUV + i, vector_worldPosition), SampleJitter(screenUV - i, vector_worldPosition), SampleJitter(screenUV, vector_worldPosition)), vector_normalDirection);
                    //float3 rayDirection = getConeSample(SampleJitter(screenUV + i, vector_worldPosition), vector_reflectionDirection, _ReflectionAngle);
                    //float3 rayDirection = getConeSample(rand(screenUV * i), vector_reflectionDirection, _ReflectionAngle);
                    //float3 rayDirection = SampleCone(_ReflectionAngle, rand(screenUV + i), vector_reflectionDirection);
                    float3 rayDirection = 
                        lerp(
                            vector_reflectionDirection,
                            randomHemispherePoint(float3(SampleJitter(screenUV + i, vector_worldPosition), SampleJitter(screenUV - i, vector_worldPosition), SampleJitter(screenUV - (i * _JitterTexture_TexelSize), vector_worldPosition)), vector_reflectionDirection),
                            //randomSpherePoint(float3(SampleJitter(screenUV + i, vector_worldPosition), SampleJitter(screenUV - i, vector_worldPosition), SampleJitter(screenUV - (i * _JitterTexture_TexelSize), vector_worldPosition))),
                            _ReflectionAngle);
                    rayDirection = normalize(rayDirection);

                    // Looking only for intersections in the forward direction of the ray.
                    float3 sky_largestRayParams = max(vector_worldCubeMax / rayDirection, vector_worldCubeMin / rayDirection);

                    // Smallest value of the ray parameters gives us the intersection.
                    float sky_dist = min(min(sky_largestRayParams.x, sky_largestRayParams.y), sky_largestRayParams.z);

                    // Find the position of the intersection point. (from the 
                    float3 sky_intersectPositionWS = vector_worldPosition + rayDirection * sky_dist;

                    // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                    float3 sky_correctedPosition = sky_intersectPositionWS - unity_SpecCube0_ProbePosition;

                    float3 skyLightSample = SampleCube(sky_correctedPosition, _EnviormentMipOffset);
                    skyLightSample *= max(0.0, dot(vector_normalDirection, rayDirection));

                    reflection += skyLightSample;
                }

                reflection /= _ReflectionSamples;
                reflection *= 1 - dot(vector_normalDirection, vector_viewDirection);
#endif

                float3 result = directLighting + bounceLighting + skyLighting + reflection;

                return float4(result, 1);
            }

            ENDCG
        }
    }
}
