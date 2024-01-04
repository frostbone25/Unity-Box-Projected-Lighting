/*
* [BoxProjectedLightingV1]
* 
* This version computes just simple directional lighting from a cubemap shadow mask.
* It's a direct implementation of the effect from the ARM article.
*/

Shader "Unlit/BoxProjectedLightingV1"
{
    Properties
    {
        [Header(Mask)]
        [Toggle(_USE_CUBEMASK)] _UseCubeMask("Use Cube Mask", Float) = 0
        _CubeMask("Cube Mask", Cube) = "white" {}

        [Header(Direct Lighting)]
        _DirectLightingShadowSoftness("Direct Lighting Shadow Softness", Float) = 0
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

            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| UNITY3D INCLUDES |||||||||||||||||||||||||||||

            //BUILT IN RENDER PIPELINE
            #include "UnityCG.cginc"
            #include "Lighting.cginc"

            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||
            //||||||||||||||||||||||||||||| SHADER PARAMETERS |||||||||||||||||||||||||||||

            samplerCUBE _CubeMask;
            float _DirectLightingShadowSoftness;

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

                return o;
            }

            float4 fragment_forward_base(vertexToFragment i) : SV_Target
            {
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

                // Get the local corrected vector, corresponding to the cubemap mask where we sample visible light from
                float3 vector_correctedPosition = vector_worldSpaceIntersectPosition - unity_SpecCube0_ProbePosition;

                // Calculate a mip level which we use as an approximation for contact hardening shadows
                float directLightSampleMipLevel = _DirectLightingShadowSoftness * distanceToIntersection / UNITY_SPECCUBE_LOD_STEPS;

                // Sample direct lighting
                float3 directLightSample = texCUBElod(_CubeMask, float4(vector_correctedPosition, directLightSampleMipLevel)).rgb;

                // Multiply direct lighting by light color, and also an normal dot light product
                float3 directLighting = directLightSample * _LightColor0.rgb * max(0.0, dot(vector_normalDirection, vector_lightDirection));

                return float4(directLighting, 1);
            }

            ENDCG
        }
    }
}