//  — Runtime 3D model loader in Unity
using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System.Diagnostics;

[DebuggerDisplay($"{{{nameof(GetDebuggerDisplay)}(),nq}}")]
public class AnatomyLoader : MonoBehaviour
{
    public string apiBaseUrl = "https://your-api.com";
    private GameObject currentModel;

    // Called when user taps "Load Heart" button in VR menu
    public void LoadHeartModel()
    {
        StartCoroutine(LoadAssetBundle("heart_v2"));
    }

    IEnumerator LoadAssetBundle(string bundleName)
    {
        string url = $"{apiBaseUrl}/assets/{bundleName}";
        
        // Download AssetBundle from your server
        using (UnityWebRequest req = UnityWebRequestAssetBundle.GetAssetBundle(url))
        {
            yield return req.SendWebRequest();
            
            if (req.result == UnityWebRequest.Result.Success)
            {
                AssetBundle bundle = DownloadHandlerAssetBundle.GetContent(req);
                GameObject prefab = bundle.LoadAsset<GameObject>(bundleName);
                
                // Destroy previous model, instantiate new one
                if (currentModel != null) Destroy(currentModel);
                currentModel = Instantiate(prefab, Vector3.zero, Quaternion.identity);
                
                // Center it at eye level in VR
                currentModel.transform.position = new Vector3(0, 1.5f, 1.5f);
                currentModel.transform.localScale = Vector3.one * 0.3f;
            }
        }
    }

    // Slice through anatomy: enable/disable child layers
    public void SetLayerVisible(string layerName, bool visible)
    {
        Transform layer = currentModel.transform.Find(layerName);
        if (layer != null)
            layer.gameObject.SetActive(visible);
    }

    // Adjust transparency (0=transparent, 1=opaque)
    public void SetTransparency(GameObject obj, float alpha)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        Color color = renderer.material.color;
        color.a = alpha;
        renderer.material.color = color;
    }

    private string GetDebuggerDisplay()
    {
        return ToString();
    }
}