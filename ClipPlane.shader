//  — Cross-section through anatomy
Shader "Custom/ClipPlane"
{
    Properties {
        _Color ("Color", Color) = (1,1,1,1)
        _PlanePoint ("Plane Point", Vector) = (0,0,0,0)
        _PlaneNormal ("Plane Normal", Vector) = (0,1,0,0)
    }
    SubShader {
        Tags { "RenderType"="Opaque" }
        Cull Off   // Render both faces (see inside)

        CGPROGRAM
        #pragma surface surf Standard

        float3 _PlanePoint;
        float3 _PlaneNormal;
        fixed4 _Color;
 struct Input { float3 worldPos; };

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            // Discard fragment if on the "cut" side of the plane
            clip(dot(IN.worldPos - _PlanePoint, _PlaneNormal));
            o.Albedo = _Color.rgb;
        }
        ENDCG
    }
}