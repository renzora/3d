//! GLTF/GLB model loader.

use super::{
    AlphaMode, LoadError, LoadedGeometry, LoadedMaterial, LoadedMesh, LoadedNode, LoadedScene,
    LoadedTexture,
};

/// GLTF/GLB file loader.
pub struct GltfLoader;

impl GltfLoader {
    /// Create a new GLTF loader.
    pub fn new() -> Self {
        Self
    }

    /// Load a GLTF/GLB file from bytes.
    pub fn load_from_bytes(&self, data: &[u8]) -> Result<LoadedScene, LoadError> {
        let gltf = gltf::Gltf::from_slice(data)
            .map_err(|e| LoadError::new(format!("Failed to parse GLTF: {}", e)))?;

        let mut scene = LoadedScene::new("GLTF Scene");

        // Get buffer data
        let buffers = self.load_buffers(&gltf, data)?;

        // Load materials
        for material in gltf.materials() {
            scene.materials.push(self.load_material(&material));
        }

        // Load meshes and build mapping from GLTF mesh index to our flattened indices
        let mut mesh_index_map: Vec<Vec<usize>> = Vec::new();
        for mesh in gltf.meshes() {
            let mut indices_for_this_mesh = Vec::new();
            for primitive in mesh.primitives() {
                let geometry = self.load_primitive(&primitive, &buffers)?;
                let mut loaded_mesh = LoadedMesh::new(
                    mesh.name().unwrap_or("Unnamed"),
                    geometry,
                );
                loaded_mesh.material_index = primitive.material().index();
                indices_for_this_mesh.push(scene.meshes.len());
                scene.meshes.push(loaded_mesh);
            }
            mesh_index_map.push(indices_for_this_mesh);
        }

        // Load nodes with correct mesh index mapping
        for node in gltf.nodes() {
            scene.nodes.push(self.load_node_with_mesh_map(&node, &mesh_index_map));
        }

        // Find root nodes
        if let Some(gltf_scene) = gltf.default_scene().or_else(|| gltf.scenes().next()) {
            for node in gltf_scene.nodes() {
                scene.root_nodes.push(node.index());
            }
        }

        // Load embedded textures
        for texture in gltf.textures() {
            if let gltf::image::Source::View { view, mime_type: _ } = texture.source().source() {
                let start = view.offset();
                let end = start + view.length();
                if let Some(buffer_data) = buffers.get(view.buffer().index()) {
                    if end <= buffer_data.len() {
                        let image_bytes = &buffer_data[start..end];
                        // Decode the image using the image crate
                        if let Ok(img) = image::load_from_memory(image_bytes) {
                            let rgba = img.to_rgba8();
                            let (width, height) = (rgba.width(), rgba.height());
                            let data = rgba.into_raw();
                            let name = format!("texture_{}", texture.index());
                            scene.textures.insert(name, LoadedTexture::new(width, height, data));
                        }
                    }
                }
            }
        }

        Ok(scene)
    }

    /// Load buffer data from GLTF.
    fn load_buffers(&self, gltf: &gltf::Gltf, _data: &[u8]) -> Result<Vec<Vec<u8>>, LoadError> {
        let mut buffers = Vec::new();

        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Bin => {
                    // GLB embedded binary
                    if let Some(blob) = gltf.blob.as_ref() {
                        buffers.push(blob.clone());
                    } else {
                        return Err(LoadError::new("GLB missing binary data"));
                    }
                }
                gltf::buffer::Source::Uri(uri) => {
                    // Data URI or external file
                    if uri.starts_with("data:") {
                        // Base64 encoded data URI
                        if let Some(base64_start) = uri.find(",") {
                            let base64_data = &uri[base64_start + 1..];
                            let decoded = base64_decode(base64_data)
                                .map_err(|e| LoadError::new(format!("Failed to decode base64: {}", e)))?;
                            buffers.push(decoded);
                        } else {
                            return Err(LoadError::new("Invalid data URI"));
                        }
                    } else {
                        // External file - not supported in WASM without fetch
                        return Err(LoadError::new(format!(
                            "External buffer files not supported: {}",
                            uri
                        )));
                    }
                }
            }
        }

        Ok(buffers)
    }

    /// Load a primitive as geometry.
    fn load_primitive(
        &self,
        primitive: &gltf::Primitive,
        buffers: &[Vec<u8>],
    ) -> Result<LoadedGeometry, LoadError> {
        let mut geometry = LoadedGeometry::new();
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|v| v.as_slice()));

        // Positions (required)
        if let Some(positions) = reader.read_positions() {
            geometry.positions = positions.collect();
        } else {
            return Err(LoadError::new("Primitive missing positions"));
        }

        // Normals
        if let Some(normals) = reader.read_normals() {
            geometry.normals = normals.collect();
        }

        // Texture coordinates
        if let Some(tex_coords) = reader.read_tex_coords(0) {
            geometry.uvs = tex_coords.into_f32().collect();
        }

        // Indices
        if let Some(indices) = reader.read_indices() {
            geometry.indices = indices.into_u32().collect();
        } else {
            // Generate sequential indices if not present
            geometry.indices = (0..geometry.positions.len() as u32).collect();
        }

        // Colors
        if let Some(colors) = reader.read_colors(0) {
            geometry.colors = Some(colors.into_rgba_f32().collect());
        }

        // Tangents
        if let Some(tangents) = reader.read_tangents() {
            geometry.tangents = Some(tangents.collect());
        }

        // Generate normals if missing
        if geometry.normals.is_empty() && !geometry.positions.is_empty() {
            geometry.compute_flat_normals();
        }

        Ok(geometry)
    }

    /// Load a material.
    fn load_material(&self, material: &gltf::Material) -> LoadedMaterial {
        let pbr = material.pbr_metallic_roughness();

        let mut loaded = LoadedMaterial::new(material.name().unwrap_or("Unnamed"));

        loaded.base_color = pbr.base_color_factor();
        loaded.metallic = pbr.metallic_factor();
        loaded.roughness = pbr.roughness_factor();
        loaded.emissive = material.emissive_factor();
        loaded.double_sided = material.double_sided();

        // Alpha mode
        loaded.alpha_mode = match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        };
        loaded.alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);

        // Texture references (store texture index as string for now)
        if let Some(info) = pbr.base_color_texture() {
            loaded.base_color_texture = Some(format!("texture_{}", info.texture().index()));
        }
        if let Some(info) = material.normal_texture() {
            loaded.normal_texture = Some(format!("texture_{}", info.texture().index()));
        }
        if let Some(info) = pbr.metallic_roughness_texture() {
            loaded.metallic_roughness_texture = Some(format!("texture_{}", info.texture().index()));
        }
        if let Some(info) = material.emissive_texture() {
            loaded.emissive_texture = Some(format!("texture_{}", info.texture().index()));
        }
        if let Some(info) = material.occlusion_texture() {
            loaded.occlusion_texture = Some(format!("texture_{}", info.texture().index()));
        }

        loaded
    }

    /// Load a node with correct mesh index mapping.
    fn load_node_with_mesh_map(&self, node: &gltf::Node, mesh_index_map: &[Vec<usize>]) -> LoadedNode {
        let mut loaded = LoadedNode::new(node.name().unwrap_or("Node"));

        let (translation, rotation, scale) = node.transform().decomposed();
        loaded.translation = translation;
        loaded.rotation = rotation;
        loaded.scale = scale;

        if let Some(mesh) = node.mesh() {
            // Use the mapping to get correct flattened mesh indices
            if let Some(indices) = mesh_index_map.get(mesh.index()) {
                loaded.mesh_indices.extend(indices.iter().copied());
            }
        }

        for child in node.children() {
            loaded.children.push(child.index());
        }

        loaded
    }
}

impl Default for GltfLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple base64 decoder.
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn decode_char(c: u8) -> Result<u8, String> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(format!("Invalid base64 character: {}", c as char)),
        }
    }

    let input = input.as_bytes();
    let mut output = Vec::with_capacity(input.len() * 3 / 4);

    for chunk in input.chunks(4) {
        if chunk.len() < 4 {
            break;
        }

        let b0 = decode_char(chunk[0])?;
        let b1 = decode_char(chunk[1])?;
        let b2 = decode_char(chunk[2])?;
        let b3 = decode_char(chunk[3])?;

        output.push((b0 << 2) | (b1 >> 4));
        if chunk[2] != b'=' {
            output.push((b1 << 4) | (b2 >> 2));
        }
        if chunk[3] != b'=' {
            output.push((b2 << 6) | b3);
        }
    }

    Ok(output)
}
