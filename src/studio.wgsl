struct PreviewLight { position: vec4<f32>, color: vec4<f32> }
struct Globals { projection: mat4x4<f32>, eye: vec4<f32>, ambient: vec4<f32>, counts: vec4<u32>, lights: array<PreviewLight,64> }
struct Material { color: vec4<f32>, surface: vec4<f32>, emission: vec4<f32>, settings: vec4<f32>, decal: vec4<f32>, decal_blend: vec4<f32>, decal_layers: vec4<f32>, secondary_color: vec4<f32>, uv_transform: vec4<f32> }
struct Instance { model: mat4x4<f32>, normals: mat4x4<f32>, orientation: vec4<f32> }
@group(2) @binding(0) var<uniform> instance: Instance;
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_map: texture_2d<f32>;
@group(1) @binding(2) var surface_map: texture_2d<f32>;
@group(1) @binding(3) var normal_map: texture_2d<f32>;
@group(1) @binding(4) var emissive_map: texture_2d<f32>;
@group(1) @binding(5) var map_sampler: sampler;
@group(1) @binding(6) var flow_map: texture_2d<f32>;
@group(1) @binding(7) var gradient_map: texture_2d<f32>;
@group(1) @binding(8) var secondary_map: texture_2d<f32>;
@group(3) @binding(0) var receiving_normal: texture_2d<f32>;
@group(3) @binding(1) var receiving_surface: texture_2d<f32>;
@group(3) @binding(2) var receiving_depth: texture_depth_2d;
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
}
@vertex fn vertex_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>,
                      @location(2) uv: vec2<f32>, @location(3) tangent: vec4<f32>, @location(4) color: vec4<f32>) -> VertexOutput {
    var output: VertexOutput;
    let world = instance.model * vec4(position, 1.0);
    output.clip = globals.projection * world;
    output.position = world.xyz;
    output.normal = (instance.normals * vec4(normal, 0.0)).xyz;
    output.uv = uv;
    output.tangent = vec4((instance.model * vec4(tangent.xyz, 0.0)).xyz, tangent.w * instance.orientation.x);
    output.color = color;
    return output;
}
// A directional fiber lobe avoids treating thin hair cards as glossy flat sheets.
// This remains a studio approximation rather than REDengine's scattering model.
fn hair_lighting(n: vec3<f32>, t: vec3<f32>, v: vec3<f32>, l: vec3<f32>, albedo: vec3<f32>, roughness: f32, sheen: f32) -> vec3<f32> {
    let diffuse = max(dot(n, l), 0.0);
    let h = normalize(v + l);
    let exponent = clamp(2.0 / (roughness * roughness) - 2.0, 2.0, 128.0);
    let specular = pow(sqrt(max(0.0, 1.0 - pow(dot(t, h), 2.0))), exponent);
    return (albedo / 3.14159265 + vec3(0.025 * specular * sheen)) * diffuse;
}
fn lighting(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, albedo: vec3<f32>, roughness: f32, metal: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let nl = max(dot(n, l), 0.0); let nv = max(dot(n, v), 0.001);
    let nh = max(dot(n, h), 0.0); let vh = max(dot(v, h), 0.0);
    let a2 = pow(roughness, 4.0);
    let d = a2 / max(3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0), 0.00001);
    let k = pow(roughness + 1.0, 2.0) / 8.0;
    let g = nv / (nv * (1.0 - k) + k) * nl / max(nl * (1.0 - k) + k, 0.0001);
    let f0 = mix(vec3(0.04), albedo, metal);
    let f = f0 + (vec3(1.0) - f0) * pow(1.0 - vh, 5.0);
    let specular = d * g * f / max(4.0 * nv * nl, 0.001);
    let diffuse = (vec3(1.0) - f) * (1.0 - metal) * albedo / 3.14159265;
    return (diffuse + specular) * nl;
}
struct ShadingSurface {
    color: vec4<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metal: f32,
    emission: vec3<f32>,
    tangent_normal: vec3<f32>,
    coverage: f32,
    normal_coverage: f32,
    flow: vec3<f32>,
}
fn read_surface(input: VertexOutput, front: bool) -> ShadingSurface {
    let uv = input.uv * material.uv_transform.xy + material.uv_transform.zw;
    let texel = textureSample(base_map, map_sampler, uv);
    var tint = material.color.rgb;
    var secondary_tint = material.secondary_color.rgb;
    if material.decal_layers.x > 0.5 {
        let dimensions = textureDimensions(gradient_map);
        let position = clamp(material.decal_layers.z, 0.0, 1.0) * f32(dimensions.x - 1u);
        let left = i32(floor(position));
        let right = min(left + 1, i32(dimensions.x) - 1);
        let a = textureLoad(gradient_map, vec2(left, i32(dimensions.y / 2u)), 0).rgb;
        let b = textureLoad(gradient_map, vec2(right, i32(dimensions.y / 2u)), 0).rgb;
        tint = mix(a, b, fract(position)) * material.decal_layers.y;
        secondary_tint = tint;
    }
    var color = vec4(texel.rgb * tint, texel.a * material.color.a);
    var coverage = texel.a;
    if material.decal_layers.w > 0.0 {
        let secondary = textureSample(secondary_map, map_sampler, input.uv);
        let secondary_alpha = secondary.a * material.decal_layers.w * (1.0 - texel.a);
        coverage = texel.a + secondary_alpha;
        if coverage > 0.00001 {
            color = vec4((texel.rgb * tint * texel.a + secondary.rgb * secondary_tint * secondary_alpha) / coverage, coverage * material.color.a);
        }
    }
    let surface = textureSample(surface_map, map_sampler, uv);
    let normal_texel = textureSample(normal_map, map_sampler, uv);
    // Native normal maps store XY (including BC5); their blue channel is not Z.
    let normal_xy = normal_texel.xy * 2.0 - 1.0;
    let normal_sample = vec3(normal_xy, sqrt(max(0.0, 1.0-dot(normal_xy,normal_xy))));
    let emission = textureSample(emissive_map, map_sampler, input.uv).rgb * material.emission.rgb * material.emission.a;
    let flow = textureSample(flow_map, map_sampler, input.uv).xyz * 2.0 - 1.0;
    if material.surface.w == 1.0 && color.a < material.surface.z { discard; }
    if material.surface.w == 2.0 && color.a < 0.001 && material.decal.x == 0.0 { discard; }
    if material.decal.x > 0.0 && coverage < 0.001 { discard; }
    // Depth-writing stochastic coverage resolves intersecting hair cards without
    // relying on a single draw order for an entire mesh chunk. Depth decorrelates
    // overlapping layers; the existing supersampled output integrates coverage.
    if material.surface.w == 3.0 {
        let seed = input.clip.xy + vec2(input.clip.z * 65536.0);
        let threshold = fract(52.9829189 * fract(dot(seed, vec2(0.06711056, 0.00583715))));
        if color.a <= threshold { discard; }
    }
    var n = normalize(input.normal) * select(-1.0, 1.0, front);
    if material.settings.x > 0.0 && dot(input.tangent.xyz, input.tangent.xyz) > 0.1 {
        let t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
        let b = cross(n, t) * input.tangent.w;
        n = normalize(mat3x3(t, b, n) * normal_sample);
    }
    let roughness = clamp(surface.g * material.surface.x, 0.08, 1.0);
    let metal = clamp(surface.b * material.surface.y, 0.0, 1.0);
    return ShadingSurface(color, n, roughness, metal, emission, normal_sample, coverage, normal_texel.a, flow);
}
fn shade_surface(input: VertexOutput, surface: ShadingSurface) -> vec4<f32> {
    let color = surface.color;
    let n = surface.normal;
    let roughness = surface.roughness;
    let metal = surface.metal;
    let emission = surface.emission;
    let flow = surface.flow;
    let v = normalize(globals.eye.xyz - input.position);
    let is_hair = material.settings.y > 0.0 && dot(input.tangent.xyz, input.tangent.xyz) > 0.1;
    var strand = vec3(0.0,1.0,0.0);
    var albedo = color.rgb;
    var strand_roughness = roughness;
    var sheen = 1.0;
    if is_hair {
        let t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
        let b = cross(n, t) * input.tangent.w;
        let direction = mix(vec3(0.0, 1.0, 0.0), flow, material.settings.w);
        strand = normalize(t * direction.x + b * direction.y + n * direction.z);
        let occlusion = clamp(input.color.r * material.settings.z, 0.0, 1.0);
        albedo = color.rgb * (1.0 - 0.7 * occlusion);
        strand_roughness = mix(roughness, 1.0, occlusion);
        sheen = 1.0 - occlusion;
    }
    var result = albedo * (globals.ambient.rgb + vec3(select(globals.ambient.w * max(n.y,0.0),0.0,is_hair))) + emission;
    // The studio has broad reflected illumination as well as direct lights.
    // Smooth dark plastics/metals otherwise only catch a subpixel point highlight.
    let reflected = reflect(-v,n);
    let f0 = mix(vec3(0.04),albedo,metal);
    // Double-sided surfaces can expose the authored back normal; Fresnel uses
    // the angle to the visible side rather than turning that side fully white.
    let environment_fresnel = f0 + (max(vec3(1.0-roughness),f0)-f0)*pow(1.0-clamp(abs(dot(n,v)),0.0,1.0),5.0);
    if !is_hair {
        result += environment_fresnel * (globals.ambient.rgb + vec3(globals.ambient.w * (0.5+0.5*max(reflected.y,0.0))));
    }
    for (var i=0u; i<globals.counts.x; i+=1u) {
        let light=globals.lights[i];
        var direction=light.position.xyz;
        var attenuation=1.0;
        if light.position.w>0.5 {
            direction=light.position.xyz-input.position;
            let distance_squared=dot(direction,direction);
            let relative_distance_squared=distance_squared/(light.color.w*light.color.w);
            let cutoff=max(0.0,1.0-relative_distance_squared*relative_distance_squared);
            attenuation=cutoff*cutoff/max(distance_squared,0.04);
        }
        let l=direction/max(length(direction),0.00001);
        if is_hair {
            result+=hair_lighting(n,strand,v,l,albedo,strand_roughness,sheen)*light.color.rgb*attenuation;
        } else {
            result+=lighting(n,v,l,albedo,roughness,metal)*light.color.rgb*attenuation;
            if light.position.w < 0.5 {
                let softbox=pow(max(dot(reflected,l),0.0),mix(64.0,2.0,roughness));
                result+=environment_fresnel*light.color.rgb*softbox*0.25;
            }
        }
    }
    result = result * (2.51 * result + vec3(0.03)) / (result * (2.43 * result + vec3(0.59)) + vec3(0.14));
    let alpha = select(1.0, color.a, material.surface.w == 2.0);
    return vec4(clamp(result, vec3(0.0), vec3(1.0)), alpha);
}
struct OpaqueOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal_roughness: vec4<f32>,
    @location(2) albedo_metalness: vec4<f32>,
}
@fragment fn forward_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return shade_surface(input, read_surface(input, front));
}
@fragment fn opaque_main(input: VertexOutput, @builtin(front_facing) front: bool) -> OpaqueOutput {
    let surface = read_surface(input, front);
    return OpaqueOutput(shade_surface(input, surface), vec4(surface.normal, surface.roughness), vec4(surface.color.rgb, surface.metal));
}
@fragment fn fragment_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var surface = read_surface(input, front);
    if material.decal.x > 0.0 {
        let pixel = vec2<i32>(input.clip.xy);
        let base_normal = textureLoad(receiving_normal, pixel, 0);
        let base_surface = textureLoad(receiving_surface, pixel, 0);
        let base_depth = textureLoad(receiving_depth, pixel, 0);
        if dot(base_normal.xyz, base_normal.xyz) > 0.5 {
            // Native DepthThreshold is not expressed in normalized device depth.
            // Mesh decals already use raster depth testing against the receiver.
            // Comparing these values discarded every brow with threshold zero.
            let n = normalize(base_normal.xyz);
            // A flat decal normal contributes no perturbation. Using the
            // overlay mesh normal directly exposes its polygon boundaries on
            // curved receivers, even where its diffuse coverage is nearly zero.
            let overlay_normal = normalize(input.normal) * select(-1.0, 1.0, front);
            var decal_normal = normalize(n + surface.normal - overlay_normal);
            if material.decal_blend.x > 0.5 {
                decal_normal = n;
                if dot(input.tangent.xyz, input.tangent.xyz) > 0.1 {
                    let t = normalize(input.tangent.xyz - n * dot(n, input.tangent.xyz));
                    let b = cross(n, t) * input.tangent.w;
                    decal_normal = normalize(mat3x3(t, b, n) * surface.tangent_normal);
                }
            }
            let normal_mask = select(1.0, surface.normal_coverage, material.decal_blend.y > 0.5);
            let normal_alpha = clamp(material.decal.y * surface.coverage * normal_mask, 0.0, 1.0);
            let surface_alpha = clamp(material.decal.z * surface.coverage, 0.0, 1.0);
            surface.normal = normalize(mix(n, decal_normal, normal_alpha));
            surface.roughness = mix(base_normal.w, surface.roughness, surface_alpha);
            surface.metal = mix(base_surface.w, surface.metal, surface_alpha);
            surface.color = vec4(mix(base_surface.rgb, surface.color.rgb, surface.color.a), 1.0);
        }
    }
    return shade_surface(input, surface);
}
