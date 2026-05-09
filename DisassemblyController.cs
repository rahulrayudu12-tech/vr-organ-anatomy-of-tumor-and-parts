//  — Exploded view + part isolation
using UnityEngine;
using System.Collections.Generic;
using System.Diagnostics;

public interface IDisassemblyController
{
    void IsolatePart(GameObject selectedPart);
    void ToggleExplodedView();
}

public interface IDisassemblyController1
{
    void IsolatePart(GameObject selectedPart);
    void ToggleExplodedView();
}

[DebuggerDisplay($"{{{nameof(GetDebuggerDisplay)}(),nq}}")]
public class DisassemblyController : MonoBehaviour, IDisassemblyController, IDisassemblyController1
{
    [System.Serializable]
    public class MachinePart
    {
        public string partName;
        public GameObject partObject;
        public Vector3 explodedPosition;     // Where it moves to in exploded view
        public string description;           // Shown in VR tooltip
    }

    public List<MachinePart> parts;
    public float animationSpeed = 2f;
    private bool isExploded = false;

    private Dictionary<GameObject, Vector3> originalPositions = new();

    void Start()
    {
        // Record all original (assembled) positions
        foreach (var part in parts)
            originalPositions[part.partObject] = part.partObject.transform.localPosition;
    }

    // Toggle between assembled ↔ exploded view
    public void ToggleExplodedView()
    {
        isExploded = !isExploded;
        foreach (var part in parts)
        {
            Vector3 target = isExploded ? part.explodedPosition : originalPositions[part.partObject];
            StartCoroutine(MovePart(part.partObject, target));
        }
    }

    IEnumerator MovePart(GameObject obj, Vector3 target)
    {
        while (Vector3.Distance(obj.transform.localPosition, target) > 0.001f)
        {
            obj.transform.localPosition = Vector3.MoveTowards(
                obj.transform.localPosition, target, animationSpeed * Time.deltaTime
            );
            yield return null;
        }
    }
    // X-ray / internal view — make all other parts transparent
    public void IsolatePart(GameObject selectedPart)
    {
        foreach (var part in parts)
        {
            Renderer r = part.partObject.GetComponent<Renderer>();
            bool isSelected = part.partObject == selectedPart;
            r.material.color = isSelected ? Color.white :
                new Color(0.5f, 0.5f, 1f, 0.1f);   // Ghost blue
        }
    }

    private string GetDebuggerDisplay()
    {
        return ToString();
    }
}