@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba32float, write>;

fn cubic(value: f32) -> f32 {
    let x = abs(value);
    if x < 1.0 { return (1.5 * x - 2.5) * x * x + 1.0; }
    if x < 2.0 { return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0; }
    return 0.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = textureDimensions(destination);
    if any(id.xy >= output_size) { return; }
    let input_size = textureDimensions(source);
    let scale = f32(input_size.x) / f32(output_size.x);
    let center = (f32(id.x) + 0.5) * scale;
    let support = max(scale, 1.0);
    let start = max(i32(floor(center - 2.0 * support)), 0);
    let end = min(i32(ceil(center + 2.0 * support)), i32(input_size.x));
    var sum = vec4<f32>(0.0);
    var weight = 0.0;
    for (var x = start; x < end; x++) {
        let w = cubic((f32(x) + 0.5 - center) / support);
        sum += textureLoad(source, vec2<i32>(x, i32(id.y)), 0) * w;
        weight += w;
    }
    textureStore(destination, vec2<i32>(id.xy), sum / weight);
}
