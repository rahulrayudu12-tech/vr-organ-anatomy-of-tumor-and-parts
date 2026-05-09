// NeuronRenderer.cs — MediVR Unity VR
// Renders a neuron network using LineRenderer for axons/dendrites.
// Supports GPU instancing for 10,000+ neurons.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MediVR
{   
    [System.Serializable]
    public class NeuronData
    {
        public Vector3   position;
        public float     size         = 0.005f;
        public Color     color        = new Color(0.2f, 0.8f, 1.0f, 0.9f);
        public List<int> connections  = new List<int>();  // indices of connected neurons
    }

    public class NeuronRenderer : MonoBehaviour
    {
        [Header("Network Settings")]
        [Range(10, 10000)]  public int   neuronCount     = 500;
        [Range(1,  20)]     public int   connectionsPerNeuron = 3;
        [Range(0.1f, 2f)]   public float networkRadius   = 0.3f;
        public bool generateOnStart = true;
        public bool animateFiring   = true;

        [Header("Visual")]
        public Material neuronMaterial;   // supports GPU instancing
        public Material axonMaterial;
        public float    axonWidth       = 0.0005f;
        public float    neuronBaseSize  = 0.005f;
        public Color    neuronColor     = new Color(0.2f, 0.8f, 1.0f);
        public Color    activeColor     = new Color(1.0f, 0.8f, 0.0f);
        public Color    axonColor       = new Color(0.1f, 0.4f, 0.6f, 0.6f);

        [Header("Firing Animation")]
        [Range(0.01f, 2f)] public float fireProbabilityPerSec = 0.1f;
        [Range(0.05f, 1f)] public float fireDuration          = 0.2f;

        // Internals
        private List<NeuronData>      _neurons    = new List<NeuronData>();
        private List<LineRenderer>    _axons      = new List<LineRenderer>();
        private List<GameObject>      _spheres    = new List<GameObject>();
        private List<Coroutine>       _coroutines = new List<Coroutine>();

        // GPU instancing buffers
        private Matrix4x4[]  _matrices;
        private MaterialPropertyBlock _mpb;
        private Mesh         _sphereMesh;

        // ── Lifecycle ─────────────────────────────────────────────────────

        private void Start()
        {
            CreateMaterials();
            if (generateOnStart)
                GenerateNetwork();
        }

        private void Update()
        {
            if (animateFiring && _neurons.Count > 0)
                TryFireRandom();

            // Draw instanced spheres
            if (_matrices != null && neuronMaterial != null && _sphereMesh != null)
                Graphics.DrawMeshInstanced(_sphereMesh, 0, neuronMaterial, _matrices,
                                           _matrices.Length, _mpb);
        }

        // ── Network Generation ────────────────────────────────────────────

        public void GenerateNetwork()
        {
            ClearNetwork();
            _neurons.Clear();

            // 1. Create neuron positions (random in sphere)
            for (int i = 0; i < neuronCount; i++)
            {
                _neurons.Add(new NeuronData
                {
                    position = Random.insideUnitSphere * networkRadius,
                    size     = neuronBaseSize * Random.Range(0.5f, 2.0f),
                    color    = Color.Lerp(neuronColor, new Color(0.5f, 0.9f, 1.0f), Random.value),
                });
            }

            // 2. Connect nearby neurons
            for (int i = 0; i < _neurons.Count; i++)
            {
                var sorted = SortByDistance(i);
                int k = Mathf.Min(connectionsPerNeuron, sorted.Count);
                for (int j = 0; j < k; j++)
                    _neurons[i].connections.Add(sorted[j]);
            }

            // 3. Render
            if (neuronCount <= 1000)
                RenderWithGameObjects();
            else
                RenderWithGPUInstancing();

            Debug.Log($"[NeuronRenderer] Generated {neuronCount} neurons, " +
                      $"{_axons.Count} axons");
        }

        // ── Rendering backends ────────────────────────────────────────────

        private void RenderWithGameObjects()
        {
            var root = new GameObject("NeuronRoot");
            root.transform.SetParent(transform, false);

            for (int i = 0; i < _neurons.Count; i++)
            {
                var n   = _neurons[i];
                var go  = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"Neuron_{i}";
                go.transform.SetParent(root.transform);
                go.transform.localPosition = n.position;
                go.transform.localScale    = Vector3.one * n.size;

                if (neuronMaterial != null)
                {
                    var r   = go.GetComponent<Renderer>();
                    var mat = new Material(neuronMaterial) { color = n.color };
                    r.material = mat;
                }
                _spheres.Add(go);
            }

            // Axons (LineRenderer per connection pair)
            var axonRoot = new GameObject("AxonRoot");
            axonRoot.transform.SetParent(transform, false);

            foreach (var n in _neurons)
            {
                foreach (int connIdx in n.connections)
                {
                    var lr = CreateAxon(axonRoot.transform, n.position, _neurons[connIdx].position);
                    _axons.Add(lr);
                }
            }
        }

        private void RenderWithGPUInstancing()
        {
            // Prepare sphere mesh
            _sphereMesh = CreateSphereMesh(8, 6);
            _matrices   = new Matrix4x4[_neurons.Count];
            _mpb        = new MaterialPropertyBlock();

            var colors = new Vector4[_neurons.Count];
            for (int i = 0; i < _neurons.Count; i++)
            {
                _matrices[i] = Matrix4x4.TRS(
                    transform.TransformPoint(_neurons[i].position),
                    Quaternion.identity,
                    Vector3.one * _neurons[i].size
                );
                colors[i] = _neurons[i].color;
            }
            _mpb.SetVectorArray("_Color", colors);

            // Still use LineRenderer for axons (instanced lines need custom shader)
            var axonRoot = new GameObject("AxonRoot");
            axonRoot.transform.SetParent(transform, false);
            // Only render a subset of axons to keep draw calls manageable
            int maxAxons = Mathf.Min(2000, _neurons.Count * connectionsPerNeuron);
            int drawn    = 0;
            foreach (var n in _neurons)
            {
                if (drawn >= maxAxons) break;
                foreach (int ci in n.connections)
                {
                    if (drawn >= maxAxons) break;
                    _axons.Add(CreateAxon(axonRoot.transform, n.position, _neurons[ci].position));
                    drawn++;
                }
            }
        }

        private LineRenderer CreateAxon(Transform parent, Vector3 from, Vector3 to)
        {
            var go = new GameObject("Axon");
            go.transform.SetParent(parent);
            var lr = go.AddComponent<LineRenderer>();
            lr.positionCount = 2;
            lr.SetPosition(0, transform.TransformPoint(from));
            lr.SetPosition(1, transform.TransformPoint(to));
            lr.startWidth = axonWidth;
            lr.endWidth   = axonWidth * 0.5f;
            lr.material   = axonMaterial ?? new Material(Shader.Find("Sprites/Default"));
            lr.startColor = axonColor;
            lr.endColor   = new Color(axonColor.r, axonColor.g, axonColor.b, 0.1f);
            lr.useWorldSpace = true;
            return lr;
        }

        // ── Firing Animation ──────────────────────────────────────────────

        private void TryFireRandom()
        {
            if (Random.value < fireProbabilityPerSec * Time.deltaTime)
            {
                int idx = Random.Range(0, _spheres.Count);
                if (idx < _spheres.Count && _spheres[idx] != null)
                    StartCoroutine(FireNeuron(idx));
            }
        }

        private IEnumerator FireNeuron(int idx)
        {
            if (idx >= _spheres.Count || _spheres[idx] == null) yield break;
            var r   = _spheres[idx].GetComponent<Renderer>();
            if (r == null) yield break;
            var origColor = r.material.color;
            r.material.color = activeColor;
            yield return new WaitForSeconds(fireDuration);
            if (r != null) r.material.color = origColor;
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private List<int> SortByDistance(int fromIdx)
        {
            var from = _neurons[fromIdx].position;
            var distances = new List<(float d, int i)>();
            for (int i = 0; i < _neurons.Count; i++)
            {
                if (i == fromIdx) continue;
                distances.Add(((from - _neurons[i].position).sqrMagnitude, i));
            }
            distances.Sort((a, b) => a.d.CompareTo(b.d));
            var result = new List<int>(distances.Count);
            foreach (var (_, i) in distances) result.Add(i);
            return result;
        }

        private void ClearNetwork()
        {
            foreach (var go in _spheres) if (go != null) Destroy(go);
            _spheres.Clear();
            foreach (var lr in _axons) if (lr != null) Destroy(lr.gameObject);
            _axons.Clear();
        }

        private void CreateMaterials()
        {
            if (neuronMaterial == null)
            {
                neuronMaterial = new Material(Shader.Find("Standard"));
                neuronMaterial.enableInstancing = true;
                neuronMaterial.color = neuronColor;
                neuronMaterial.SetFloat("_Metallic",   0.0f);
                neuronMaterial.SetFloat("_Smoothness", 0.7f);
                var c = neuronColor; c.a = 0.85f;
                neuronMaterial.SetFloat("_Mode", 3);   // Transparent
                neuronMaterial.color = c;
            }
            if (axonMaterial == null)
            {
                axonMaterial = new Material(Shader.Find("Sprites/Default"));
                axonMaterial.color = axonColor;
            }
        }

        private static Mesh CreateSphereMesh(int lon, int lat)
        {
            var mesh   = new Mesh();
            var verts  = new List<Vector3>();
            var tris   = new List<int>();

            for (int i = 0; i <= lat; i++)
            {
                float theta = i * Mathf.PI / lat;
                for (int j = 0; j <= lon; j++)
                {
                    float phi = j * 2 * Mathf.PI / lon;
                    verts.Add(new Vector3(
                        Mathf.Sin(theta) * Mathf.Cos(phi),
                        Mathf.Cos(theta),
                        Mathf.Sin(theta) * Mathf.Sin(phi)));
                }
            }
            for (int i = 0; i < lat; i++)
            {
                for (int j = 0; j < lon; j++)
                {
                    int a = i * (lon+1) + j;
                    int b = a + 1;
                    int c = a + lon + 1;
                    int d = c + 1;
                    tris.AddRange(new[]{ a,c,b, b,c,d });
                }
            }
            mesh.vertices  = verts.ToArray();
            mesh.triangles = tris.ToArray();
            mesh.RecalculateNormals();
            return mesh;
        }
    }
}