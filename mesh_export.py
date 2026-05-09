"""
mesh_export.py — MediVR AI Backend
Convert tumor segmentation mask → 3D mesh → STL/OBJ/GLB export
"""

import numpy as np
import struct
import os
from pathlib import Path

try:
    from skimage.measure import marching_cubes
    from skimage.filters import gaussian
    SKIMAGE = True
except ImportError:
    SKIMAGE = False
    print("⚠️  scikit-image not found. pip install scikit-image")

try:
    import trimesh
    from trimesh.smoothing import filter_laplacian
    TRIMESH = True
except ImportError:
    TRIMESH = False

def predict_tumor_mask(volume, model=None):
    """Fallback placeholder for tumor mask prediction."""
    print("⚠️  model.predict_tumor_mask not imported. Using threshold-based fallback.")
    return (volume > volume.mean()).astype(np.uint8)

try:
    from model import predict_tumor_mask
except ImportError:
    print("⚠️  model module not found. Using fallback implementation.")


# ── Mask → Mesh ───────────────────────────────────────────────────────────

def mask_to_mesh(
    mask: np.ndarray,
    voxel_size_mm: float = 1.0,
    smooth_sigma: float = 1.0,
    level: float = 0.3,
    laplacian_iters: int = 5,
) -> dict:
    """
    Convert binary 3D mask → mesh via Marching Cubes.

    Args:
        mask:           uint8 binary volume
        voxel_size_mm:  physical voxel spacing in mm
        smooth_sigma:   Gaussian blur sigma before marching cubes
        level:          iso-surface threshold
        laplacian_iters: mesh smoothing iterations (0 = no smoothing)

    Returns:
        dict with: verts, faces, normals, stats
    """
    if not SKIMAGE:
        raise ImportError("pip install scikit-image")

    # 1. Smooth mask (reduces staircase artifacts)
    smooth = gaussian(mask.astype(float), sigma=smooth_sigma)

    if smooth.max() < level:
        raise ValueError(f"Mask empty after smoothing (max={smooth.max():.4f}). "
                         "The model found no tumor region.")

    # 2. Marching Cubes
    spacing = (voxel_size_mm,) * 3
    verts, faces, normals, values = marching_cubes(smooth, level=level, spacing=spacing)

    stats = {
        "vertices": len(verts),
        "faces":    len(faces),
        "volume_mm3": float(mask.sum() * (voxel_size_mm ** 3)),
        "surface_area_mm2": float(len(faces) * 0.5),  # rough estimate
    }

    print(f"[mask_to_mesh] verts={stats['vertices']:,}  faces={stats['faces']:,}  "
          f"volume≈{stats['volume_mm3']:.1f}mm³")

    # 3. Laplacian smoothing (if trimesh available)
    if TRIMESH and laplacian_iters > 0:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        filter_laplacian(mesh, lamb=0.5, iterations=laplacian_iters)
        verts   = np.array(mesh.vertices)
        faces   = np.array(mesh.faces)
        normals = np.array(mesh.vertex_normals)
        print(f"[mask_to_mesh] Laplacian smoothed ({laplacian_iters} iters)")

    return {"verts": verts, "faces": faces, "normals": normals, "stats": stats}


# ── Export formats ────────────────────────────────────────────────────────

def export_stl_ascii(verts: np.ndarray, faces: np.ndarray, path: str):
    """Write ASCII STL — readable by Unity, Blender, MeshLab."""
    path = str(path)
    with open(path, 'w') as f:
        f.write("solid tumor\n")
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            e1, e2 = v1 - v0, v2 - v0
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            f.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            f.write("    outer loop\n")
            for v in [v0, v1, v2]:
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid tumor\n")
    size_kb = os.path.getsize(path) / 1024
    print(f"[export_stl_ascii]  {path}  ({size_kb:.1f} KB)")


def export_stl_binary(verts: np.ndarray, faces: np.ndarray, path: str):
    """Write binary STL — ~6× smaller than ASCII. Better for large meshes."""
    path = str(path)
    n_faces = len(faces)
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)                # 80-byte header
        f.write(struct.pack('<I', n_faces))   # face count
        for face in faces:
            v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
            e1, e2 = v1 - v0, v2 - v0
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            f.write(struct.pack('<fff', *n))
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))     # attribute byte count
    size_kb = os.path.getsize(path) / 1024
    print(f"[export_stl_binary] {path}  ({size_kb:.1f} KB)")


def export_obj(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray,
               path: str, material_name: str = "tumor"):
    """Write Wavefront OBJ + MTL — supports materials for Unity/Blender."""
    path = Path(path)
    mtl_path = path.with_suffix('.mtl')

    # Write OBJ
    with open(path, 'w') as f:
        f.write(f"# MediVR Tumor Mesh\n")
        f.write(f"mtllib {mtl_path.name}\n")
        f.write(f"usemtl {material_name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for face in faces:
            i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
            f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")

    # Write MTL
    with open(mtl_path, 'w') as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 0.2 0.0 0.0\n")   # ambient: dark red
        f.write("Kd 0.8 0.1 0.1\n")   # diffuse: red
        f.write("Ks 0.3 0.3 0.3\n")   # specular: white
        f.write("Ns 50\n")             # shininess
        f.write("d 0.85\n")            # alpha (slight transparency)

    size_kb = os.path.getsize(path) / 1024
    print(f"[export_obj]  {path}  ({size_kb:.1f} KB)  +  {mtl_path.name}")


def export_glb(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, path: str):
    """Write binary GLTF (.glb) — optimal for Unity AssetBundle pipeline."""
    if not TRIMESH:
        raise ImportError("pip install trimesh")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        diffuse=[204, 26, 26, 217]   # red, 85% opaque
    )
    mesh.export(str(path))
    size_kb = os.path.getsize(path) / 1024
    print(f"[export_glb]  {path}  ({size_kb:.1f} KB)")


# ── High-level pipeline function ──────────────────────────────────────────

def generate_tumor_mesh(
    volume: np.ndarray,
    output_dir: str = "./outputs",
    job_id: str = "test",
    model=None,
    formats: list = None,
) -> dict:
    """
    End-to-end: volume → predict mask → mesh → export.

    Returns dict with output file paths and stats.
    """
    formats = formats or ["stl_binary", "obj"]
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Predict mask
    mask = predict_tumor_mask(volume, model=model)

    if mask.sum() == 0:
        return {"error": "No tumor detected", "mask_voxels": 0}

    # 2. Build mesh
    mesh_data = mask_to_mesh(mask, voxel_size_mm=1.0)
    verts   = mesh_data["verts"]
    faces   = mesh_data["faces"]
    normals = mesh_data["normals"]
    stats   = mesh_data["stats"]

    # 3. Export
    outputs = {}
    if "stl_ascii" in formats:
        p = out_dir / f"tumor_{job_id}.stl"
        export_stl_ascii(verts, faces, str(p))
        outputs["stl"] = str(p)

    if "stl_binary" in formats:
        p = out_dir / f"tumor_{job_id}_bin.stl"
        export_stl_binary(verts, faces, str(p))
        outputs["stl_binary"] = str(p)

    if "obj" in formats:
        p = out_dir / f"tumor_{job_id}.obj"
        export_obj(verts, faces, normals, str(p))
        outputs["obj"] = str(p)

    if "glb" in formats and TRIMESH:
        p = out_dir / f"tumor_{job_id}.glb"
        export_glb(verts, faces, normals, str(p))
        outputs["glb"] = str(p)

    return {
        "job_id":  job_id,
        "outputs": outputs,
        "stats":   stats,
        "mask_voxels": int(mask.sum()),
    }


if __name__ == "__main__":
    from preprocessing import full_pipeline

    print("mesh_export.py — running end-to-end test...")
    result = full_pipeline("synthetic", target_size=64)
    vol = result["processed"]
    gt  = result["ground_truth"]

    # Predict
    from model import ThresholdSegmentor
    mask = ThresholdSegmentor().predict(vol)
    print(f"Mask voxels: {mask.sum()}")

    # Override mask with ground truth for clean mesh test
    mesh_result = generate_tumor_mesh(
        gt,    # use GT for a clean test mesh
        output_dir="./test_outputs",
        job_id="demo",
        model=None,
        formats=["stl_ascii", "stl_binary", "obj"],
    )
    print("\nOutput files:")
    for k, v in mesh_result["outputs"].items():
        print(f"  {k}: {v}")
    print("\nStats:", mesh_result["stats"])
    print("\nmesh_export.py — OK")