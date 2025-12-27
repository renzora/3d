//! Wavefront OBJ file loader.

use super::{LoadError, LoadedGeometry, LoadedMaterial, LoadedMesh, LoadedNode, LoadedScene};
use std::collections::HashMap;

/// Wavefront OBJ file loader.
pub struct ObjLoader;

impl ObjLoader {
    /// Create a new OBJ loader.
    pub fn new() -> Self {
        Self
    }

    /// Load an OBJ file from string content.
    pub fn load_from_str(&self, content: &str) -> Result<LoadedScene, LoadError> {
        let mut scene = LoadedScene::new("OBJ Scene");

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();

        let mut current_geometry = LoadedGeometry::new();
        let mut current_name = String::from("default");
        let mut current_material: Option<usize> = None;

        // Vertex index cache for deduplication
        let mut vertex_cache: HashMap<(usize, usize, usize), u32> = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "v" => {
                    // Vertex position
                    if parts.len() >= 4 {
                        let x = parts[1].parse::<f32>().unwrap_or(0.0);
                        let y = parts[2].parse::<f32>().unwrap_or(0.0);
                        let z = parts[3].parse::<f32>().unwrap_or(0.0);
                        positions.push([x, y, z]);
                    }
                }
                "vn" => {
                    // Vertex normal
                    if parts.len() >= 4 {
                        let x = parts[1].parse::<f32>().unwrap_or(0.0);
                        let y = parts[2].parse::<f32>().unwrap_or(0.0);
                        let z = parts[3].parse::<f32>().unwrap_or(0.0);
                        normals.push([x, y, z]);
                    }
                }
                "vt" => {
                    // Texture coordinate
                    if parts.len() >= 3 {
                        let u = parts[1].parse::<f32>().unwrap_or(0.0);
                        let v = parts[2].parse::<f32>().unwrap_or(0.0);
                        uvs.push([u, v]);
                    }
                }
                "f" => {
                    // Face
                    if parts.len() >= 4 {
                        let face_vertices: Vec<(usize, usize, usize)> = parts[1..]
                            .iter()
                            .filter_map(|p| self.parse_face_vertex(p))
                            .collect();

                        // Triangulate (fan triangulation for convex polygons)
                        for i in 1..face_vertices.len() - 1 {
                            let indices = [
                                face_vertices[0],
                                face_vertices[i],
                                face_vertices[i + 1],
                            ];

                            for (vi, ti, ni) in indices {
                                let key = (vi, ti, ni);
                                let index = if let Some(&idx) = vertex_cache.get(&key) {
                                    idx
                                } else {
                                    let idx = current_geometry.positions.len() as u32;

                                    // Position (1-indexed, 0 means not present)
                                    if vi > 0 && vi <= positions.len() {
                                        current_geometry.positions.push(positions[vi - 1]);
                                    }

                                    // UV
                                    if ti > 0 && ti <= uvs.len() {
                                        current_geometry.uvs.push(uvs[ti - 1]);
                                    } else if !uvs.is_empty() {
                                        current_geometry.uvs.push([0.0, 0.0]);
                                    }

                                    // Normal
                                    if ni > 0 && ni <= normals.len() {
                                        current_geometry.normals.push(normals[ni - 1]);
                                    }

                                    vertex_cache.insert(key, idx);
                                    idx
                                };

                                current_geometry.indices.push(index);
                            }
                        }
                    }
                }
                "o" | "g" => {
                    // Object or group - start new mesh
                    if !current_geometry.positions.is_empty() {
                        // Save current geometry
                        if current_geometry.normals.is_empty() {
                            current_geometry.compute_flat_normals();
                        }
                        let mut mesh = LoadedMesh::new(&current_name, current_geometry);
                        mesh.material_index = current_material;
                        scene.meshes.push(mesh);

                        current_geometry = LoadedGeometry::new();
                        vertex_cache.clear();
                    }

                    if parts.len() >= 2 {
                        current_name = parts[1..].join(" ");
                    }
                }
                "usemtl" => {
                    // Material reference
                    if parts.len() >= 2 {
                        let mat_name = parts[1];
                        // Find or create material
                        if let Some(idx) = scene.materials.iter().position(|m| m.name == mat_name) {
                            current_material = Some(idx);
                        } else {
                            let material = LoadedMaterial::new(mat_name);
                            current_material = Some(scene.materials.len());
                            scene.materials.push(material);
                        }
                    }
                }
                "mtllib" => {
                    // Material library reference (not loaded here)
                }
                _ => {
                    // Unknown command, ignore
                }
            }
        }

        // Save final geometry
        if !current_geometry.positions.is_empty() {
            if current_geometry.normals.is_empty() {
                current_geometry.compute_flat_normals();
            }
            let mut mesh = LoadedMesh::new(&current_name, current_geometry);
            mesh.material_index = current_material;
            scene.meshes.push(mesh);
        }

        // Create a single root node containing all meshes
        if !scene.meshes.is_empty() {
            let mut root_node = LoadedNode::new("Root");
            for i in 0..scene.meshes.len() {
                root_node.mesh_indices.push(i);
            }
            scene.nodes.push(root_node);
            scene.root_nodes.push(0);
        }

        Ok(scene)
    }

    /// Load an OBJ file from bytes.
    pub fn load_from_bytes(&self, data: &[u8]) -> Result<LoadedScene, LoadError> {
        let content = std::str::from_utf8(data)
            .map_err(|e| LoadError::new(format!("Invalid UTF-8: {}", e)))?;
        self.load_from_str(content)
    }

    /// Parse a face vertex specification (v/vt/vn or v//vn or v/vt or v).
    fn parse_face_vertex(&self, s: &str) -> Option<(usize, usize, usize)> {
        let parts: Vec<&str> = s.split('/').collect();

        let v = parts.get(0).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let vt = parts.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        let vn = parts.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);

        if v > 0 {
            Some((v, vt, vn))
        } else {
            None
        }
    }

    /// Load MTL material library from string content.
    pub fn load_mtl_from_str(&self, content: &str) -> Vec<LoadedMaterial> {
        let mut materials = Vec::new();
        let mut current: Option<LoadedMaterial> = None;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "newmtl" => {
                    if let Some(mat) = current.take() {
                        materials.push(mat);
                    }
                    let name = if parts.len() >= 2 { parts[1] } else { "Unnamed" };
                    current = Some(LoadedMaterial::new(name));
                }
                "Kd" => {
                    // Diffuse color
                    if let Some(ref mut mat) = current {
                        if parts.len() >= 4 {
                            let r = parts[1].parse::<f32>().unwrap_or(1.0);
                            let g = parts[2].parse::<f32>().unwrap_or(1.0);
                            let b = parts[3].parse::<f32>().unwrap_or(1.0);
                            mat.base_color = [r, g, b, mat.base_color[3]];
                        }
                    }
                }
                "d" | "Tr" => {
                    // Dissolve/transparency
                    if let Some(ref mut mat) = current {
                        if parts.len() >= 2 {
                            let a = parts[1].parse::<f32>().unwrap_or(1.0);
                            mat.base_color[3] = if parts[0] == "Tr" { 1.0 - a } else { a };
                        }
                    }
                }
                "Ns" => {
                    // Specular exponent -> roughness
                    if let Some(ref mut mat) = current {
                        if parts.len() >= 2 {
                            let ns = parts[1].parse::<f32>().unwrap_or(100.0);
                            // Convert specular exponent to roughness (rough approximation)
                            mat.roughness = (1.0 - (ns / 1000.0).min(1.0)).max(0.04);
                        }
                    }
                }
                "map_Kd" => {
                    // Diffuse texture
                    if let Some(ref mut mat) = current {
                        if parts.len() >= 2 {
                            mat.base_color_texture = Some(parts[1..].join(" "));
                        }
                    }
                }
                "map_Bump" | "bump" => {
                    // Normal/bump map
                    if let Some(ref mut mat) = current {
                        if parts.len() >= 2 {
                            mat.normal_texture = Some(parts[1..].join(" "));
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(mat) = current {
            materials.push(mat);
        }

        materials
    }
}

impl Default for ObjLoader {
    fn default() -> Self {
        Self::new()
    }
}
