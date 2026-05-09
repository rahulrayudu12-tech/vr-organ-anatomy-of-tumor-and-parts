// ClippingPlaneController.cs — MediVR Unity VR
// Controls the anatomy slicing plane. Left controller moves the plane.
// Works with the ClipPlane.shader below.

using UnityEngine;
using UnityEngine.XR;
using System.Collections.Generic;

namespace MediVR
{
    public class ClippingPlaneController : MonoBehaviour
    {
        [Header("Clip Plane Settings")]
        [Tooltip("All renderers that use ClipPlane.shader — will be updated each frame.")]
        public Renderer[] clippableRenderers;

        [Tooltip("Visual representation of the cutting plane (a flat quad).")]
        public Transform planeVisual;

        [Range(-1f, 1f)] public float planeHeight = 0f;   // normalized Y position
        public Vector3 planeNormal = Vector3.up;

        [Header("Controller Input")]
        public XRNode controllerNode  = XRNode.LeftHand;
        [Range(0.1f, 2f)] public float moveSpeed = 0.5f;
        public bool  requireGrip = true;    // only move while grip held

        // Shader property IDs (cached for performance)
        private static readonly int PlanePointID  = Shader.PropertyToID("_PlanePoint");
        private static readonly int PlaneNormalID = Shader.PropertyToID("_PlaneNormal");

        private List<InputDevice> _devices = new List<InputDevice>();
        private bool _isGripHeld;

        // ── Unity lifecycle ───────────────────────────────────────────────

        private void Start()
        {
            UpdateShaders();
        }

        private void Update()
        {
            HandleControllerInput();
            UpdateShaders();
            UpdateVisual();
        }

        // ── Input handling ────────────────────────────────────────────────

        private void HandleControllerInput()
        {
            InputDevices.GetDevicesAtXRNode(controllerNode, _devices);
            if (_devices.Count == 0) return;

            var device = _devices[0];

            // Check grip
            if (requireGrip)
            {
                device.TryGetFeatureValue(CommonUsages.gripButton, out _isGripHeld);
                if (!_isGripHeld) return;
            }

            // Thumbstick Y-axis to move the plane
            device.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 thumbstick);
            planeHeight += thumbstick.y * moveSpeed * Time.deltaTime;
            planeHeight  = Mathf.Clamp(planeHeight, -1f, 1f);

            // Trigger: toggle between Y-axis and Z-axis normal
            device.TryGetFeatureValue(CommonUsages.triggerButton, out bool trigger);
            if (trigger)
                planeNormal = (planeNormal == Vector3.up) ? Vector3.forward : Vector3.up;
        }

        // ── Shader update ─────────────────────────────────────────────────

        private void UpdateShaders()
        {
            // World-space point on the plane
            Vector3 worldPoint = transform.position + planeNormal * planeHeight;
            Vector3 worldNorm  = transform.TransformDirection(planeNormal).normalized;

            foreach (var r in clippableRenderers)
            {
                if (r == null) continue;
                foreach (var mat in r.materials)
                {
                    mat.SetVector(PlanePointID,  worldPoint);
                    mat.SetVector(PlaneNormalID, worldNorm);
                }
            }
        }

        // ── Visual ────────────────────────────────────────────────────────

        private void UpdateVisual()
        {
            if (planeVisual == null) return;
            planeVisual.position = transform.position + planeNormal * planeHeight;
            planeVisual.up       = transform.TransformDirection(planeNormal);
        }

        // ── Public API ────────────────────────────────────────────────────

        public void SetPlaneHeight(float t) { planeHeight = Mathf.Clamp(t, -1f, 1f); }
        public void SetPlaneNormal(Vector3 n) { planeNormal = n.normalized; }

        public void SetAxialCut()   { planeNormal = Vector3.up; }
        public void SetCoronalCut() { planeNormal = Vector3.forward; }
        public void SetSagittalCut(){ planeNormal = Vector3.right; }

        public void ResetPlane()
        {
            planeHeight = 0f;
            planeNormal = Vector3.up;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ClipPlane.shader  — Unity HLSL Surface Shader
// Place in: Assets/Shaders/ClipPlane.shader
// ═══════════════════════════════════════════════════════════════════════════
// 
// Shader "MediVR/ClipPlane"
/*
Shader "MediVR/ClipPlane"
{
    Properties
    {
        _Color       ("Color",      Color)     = (1,1,1,1)
        _MainTex     ("Albedo",     2D)        = "white" {}
        _Glossiness  ("Smoothness", Range(0,1))= 0.5
        _Metallic    ("Metallic",   Range(0,1))= 0.0
        _PlanePoint  ("Plane Point",  Vector)  = (0,0,0,0)
        _PlaneNormal ("Plane Normal", Vector)  = (0,1,0,0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Off    // render both sides so cut surfaces are visible
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        sampler2D _MainTex;
        half  _Glossiness;
        half  _Metallic;
        fixed4 _Color;
        float3 _PlanePoint;
        float3 _PlaneNormal;

        struct Input { float2 uv_MainTex; float3 worldPos; };

        void surf(Input IN, inout SurfaceOutputStandard o)
        {
            // Discard fragment if it is on the positive side of the plane
            clip(dot(IN.worldPos - _PlanePoint, _PlaneNormal));

            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo     = c.rgb;
            o.Metallic   = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha      = c.a;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
*/
