"""
blender_export.py — MediVR Blender Asset Pipeline
Automated 3D model optimization and export for Unity VR.

Run in Blender's Script editor (Text → Run Script)
OR from command line:
    blender --background model.blend --python blender_export.py -- \
        --output ./exports --lod --format GLB
"""

import bpy # pyright: ignore[reportMissingImports]
import sys
import os
import math
from pathlib import Path

# Parse command-line arguments (after --)
argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []

# Default config
CONFIG = {
    "output_dir":      "./medivr_exports",
    "lod_levels":      [1.0, 0.30, 0.08],   # poly ratios: full, 30%, 8%
    "max_tris_quest2": 150_000,
    "max_tris_quest3": 300_000,
    "max_tris_cardboard": 50_000,
    "texture_size":    1024,
    "export_formats":  ["GLB", "FBX", "STL"],
    "auto_smooth":     True,
    "smooth_angle":    60.0,   # degrees
}

# Parse basic CLI args
for i, a in enumerate(args):
    if a == "--output" and i+1 < len(args): CONFIG["output_dir"] = args[i+1]
    if a == "--format" and i+1 < len(args): CONFIG["export_formats"] = [args[i+1]]
    if a == "--lod":    CONFIG["do_lod"] = True
    if a == "--notex":  CONFIG["texture_size"] = 0


# ── Helpers ────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[MediVR Blender] {msg}")

def select_only(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def get_mesh_stats(obj) -> dict:
    """Return triangle and vertex counts for a mesh object."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    return {
        "tris":   len(mesh.loop_triangles),
        "verts":  len(mesh.vertices),
        "faces":  len(mesh.polygons),
    }

def apply_auto_smooth(obj, angle_deg: float = 60.0):
    """Apply smooth shading with auto-smooth angle."""
    select_only(obj)
    bpy.ops.object.shade_smooth()
    obj.data.use_auto_smooth = True
    obj.data.auto_smooth_angle = math.radians(angle_deg)
    log(f"  Auto-smooth applied ({angle_deg}°)")

def remove_doubles(obj, threshold: float = 0.0001):
    """Merge duplicate vertices."""
    select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"  Duplicate vertices removed (threshold={threshold})")

def recalculate_normals(obj):
    """Fix flipped normals."""
    select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    log("  Normals recalculated")

def triangulate(obj):
    """Convert quads/ngons to triangles (required for Unity)."""
    select_only(obj)
    mod = obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    mod.quad_method = 'BEAUTY'
    bpy.ops.object.modifier_apply(modifier="Triangulate")
    log("  Triangulated")


# ── LOD Generation ─────────────────────────────────────────────────────────

def generate_lod_levels(obj, ratios: list) -> list:
    """
    Create LOD variants using Decimate modifier.
    Returns list of (ratio, lod_object) tuples.
    """
    lods = []
    stats_orig = get_mesh_stats(obj)
    log(f"  Original: {stats_orig['tris']:,} tris")

    for i, ratio in enumerate(ratios):
        if ratio >= 1.0:
            lods.append((ratio, obj))
            continue

        # Duplicate object
        select_only(obj)
        bpy.ops.object.duplicate()
        lod_obj = bpy.context.object
        lod_obj.name = f"{obj.name}_LOD{i}"

        # Apply Decimate modifier
        mod = lod_obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")

        stats = get_mesh_stats(lod_obj)
        log(f"  LOD{i} ({ratio:.0%}): {stats['tris']:,} tris")
        lods.append((ratio, lod_obj))

    return lods


# ── UV & Texture ───────────────────────────────────────────────────────────

def unwrap_uv(obj):
    """Smart UV project for medical models."""
    select_only(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    log("  UV unwrapped (smart project)")

def bake_ao(obj, tex_size: int = 1024):
    """Bake ambient occlusion to a texture."""
    if tex_size == 0: return
    image = bpy.data.images.new(f"{obj.name}_AO", tex_size, tex_size)
    # Setup material for baking
    mat = obj.data.materials[0] if obj.data.materials else None
    if mat is None:
        mat = bpy.data.materials.new(f"{obj.name}_mat")
        obj.data.materials.append(mat)
    mat.use_nodes = True
    tree = mat.node_tree
    tex_node = tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = image
    tree.nodes.active = tex_node

    # Bake
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    select_only(obj)
    bpy.ops.object.bake(type='AO', use_clear=True)
    image.save_render(filepath=f"{CONFIG['output_dir']}/{obj.name}_ao.png")
    log(f"  AO baked → {obj.name}_ao.png")


# ── Export functions ────────────────────────────────────────────────────────

def export_glb(obj, path: str):
    """Export as binary glTF 2.0 (.glb) — optimal for Unity."""
    select_only(obj)
    bpy.ops.export_scene.gltf(
        filepath=path,
        use_selection=True,
        export_format='GLB',
        export_draco_mesh_compression_enable=False,
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_yup=True,
    )
    size_kb = os.path.getsize(path) / 1024
    log(f"  GLB exported: {path} ({size_kb:.1f} KB)")

def export_fbx(obj, path: str):
    """Export as FBX — best for Unity material import."""
    select_only(obj)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_NONE',
        axis_forward='-Z',
        axis_up='Y',
        bake_space_transform=True,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
        use_tspace=True,
        embed_textures=False,
        path_mode='COPY',
    )
    size_kb = os.path.getsize(path) / 1024
    log(f"  FBX exported: {path} ({size_kb:.1f} KB)")

def export_stl(obj, path: str):
    """Export as STL — for the tumor mesh import to Unity."""
    select_only(obj)
    bpy.ops.export_mesh.stl(
        filepath=path,
        use_selection=True,
        global_scale=1.0,
        use_scene_unit=False,
        ascii=False,
        use_mesh_modifiers=True,
    )
    size_kb = os.path.getsize(path) / 1024
    log(f"  STL exported: {path} ({size_kb:.1f} KB)")


# ── Main pipeline ───────────────────────────────────────────────────────────

def process_object(obj):
    """Full optimization pipeline for a single mesh object."""
    if obj.type != 'MESH': return
    log(f"\n{'─'*50}")
    log(f"Processing: {obj.name}")

    out = Path(CONFIG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # 1. Cleanup
    remove_doubles(obj)
    recalculate_normals(obj)
    triangulate(obj)
    if CONFIG.get("auto_smooth", True):
        apply_auto_smooth(obj, CONFIG["smooth_angle"])

    stats = get_mesh_stats(obj)
    log(f"  After cleanup: {stats['tris']:,} tris  {stats['verts']:,} verts")

    # 2. Warn about polygon budget
    t = stats["tris"]
    if   t > CONFIG["max_tris_quest3"]:   log(f"  ⚠️  OVER Quest3 budget ({CONFIG['max_tris_quest3']:,})")
    elif t > CONFIG["max_tris_quest2"]:   log(f"  ⚠️  OVER Quest2 budget ({CONFIG['max_tris_quest2']:,})")
    elif t > CONFIG["max_tris_cardboard"]:log(f"  ✅ OK for Quest2/3  ⚠️  OVER Cardboard budget")
    else:                                  log(f"  ✅ Within all platform budgets")

    # 3. UV unwrap
    unwrap_uv(obj)

    # 4. LOD levels
    lods = generate_lod_levels(obj, CONFIG["lod_levels"])

    # 5. Export each LOD in each format
    for i, (ratio, lod_obj) in enumerate(lods):
        lod_suffix = "" if i == 0 else f"_LOD{i}"
        for fmt in CONFIG["export_formats"]:
            path = str(out / f"{obj.name}{lod_suffix}.{fmt.lower()}")
            if   fmt == "GLB": export_glb(lod_obj, path)
            elif fmt == "FBX": export_fbx(lod_obj, path)
            elif fmt == "STL": export_stl(lod_obj, path)

    log(f"  Done: {obj.name}")


def main():
    log("MediVR Blender Export Pipeline starting...")
    log(f"Output: {CONFIG['output_dir']}")
    log(f"Formats: {CONFIG['export_formats']}")

    # Process all selected mesh objects (or all if none selected)
    objects = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not objects:
        objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']

    if not objects:
        log("No mesh objects found in scene!")
        return

    log(f"Found {len(objects)} mesh object(s)")
    for obj in objects:
        process_object(obj)

    log(f"\n{'='*50}")
    log("Export complete!")
    log(f"Files saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()